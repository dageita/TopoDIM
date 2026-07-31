"""Efficiency metrics for multi-agent topology experiments (B0–B5).

Per-LLM-call: TTFT, node E2E, tokens/chars, cost.
Per-question: question E2E, active agents, edge counts, LLM call count,
              peer-context volume, decision latency.
Aggregates: p50/p95 TTFT, questions/hour, correct/hour, tokens/s,
            accuracy / calls, accuracy / wall-time.
"""

from __future__ import annotations

import contextvars
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# Context for the question currently being solved in this asyncio Task.
_active_question: contextvars.ContextVar[Optional["QuestionMetrics"]] = contextvars.ContextVar(
    "topodim_active_question_metrics", default=None
)
# Peer / debate / query context chars injected into the next agen() call.
_pending_peer_chars: contextvars.ContextVar[int] = contextvars.ContextVar(
    "topodim_pending_peer_chars", default=0
)


def set_pending_peer_chars(n: int) -> None:
    _pending_peer_chars.set(max(0, int(n)))


def consume_pending_peer_chars() -> int:
    n = _pending_peer_chars.get()
    _pending_peer_chars.set(0)
    return n


def get_active_question_metrics() -> Optional["QuestionMetrics"]:
    return _active_question.get()


@dataclass
class LLMCallMetrics:
    """One ``llm.agen`` / Claude Code session."""

    role: str = ""
    node_id: str = ""
    mode: int = 0  # 0 normal / 1 query-eval / 2 debate / 3 evaluation-reply
    ttft_s: float = 0.0  # prompt sent → first assistant text
    e2e_s: float = 0.0  # full agen wall time
    prompt_chars: int = 0
    completion_chars: int = 0
    peer_context_chars: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    is_decision: bool = False


@dataclass
class QuestionMetrics:
    """One question / one Graph.arun()."""

    sample_id: int = -1
    question_preview: str = ""
    correct: Optional[bool] = None
    question_e2e_s: float = 0.0
    decision_e2e_s: float = 0.0
    active_agents: int = 0
    spatial_edges: int = 0
    query_edges: int = 0
    debate_edges: int = 0
    llm_calls: List[LLMCallMetrics] = field(default_factory=list)
    topo_summary: str = ""

    def begin(self) -> "QuestionMetrics":
        _active_question.set(self)
        self._t0 = time.perf_counter()
        return self

    def end(self) -> "QuestionMetrics":
        self.question_e2e_s = time.perf_counter() - getattr(self, "_t0", time.perf_counter())
        _active_question.set(None)
        return self

    def add_call(self, call: LLMCallMetrics) -> None:
        self.llm_calls.append(call)

    @property
    def num_llm_calls(self) -> int:
        return len(self.llm_calls)

    @property
    def total_edges(self) -> int:
        return self.spatial_edges + self.query_edges + self.debate_edges

    @property
    def peer_context_chars(self) -> int:
        return sum(c.peer_context_chars for c in self.llm_calls)

    @property
    def first_ttft_s(self) -> float:
        return self.llm_calls[0].ttft_s if self.llm_calls else 0.0

    @property
    def mean_call_e2e_s(self) -> float:
        if not self.llm_calls:
            return 0.0
        return sum(c.e2e_s for c in self.llm_calls) / len(self.llm_calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.llm_calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.llm_calls)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_t0", None)
        d["num_llm_calls"] = self.num_llm_calls
        d["total_edges"] = self.total_edges
        d["peer_context_chars_total"] = self.peer_context_chars
        d["first_ttft_s"] = self.first_ttft_s
        return d


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[int(k)])
    return float(xs[f] * (c - k) + xs[c] * (k - f))


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


@dataclass
class EfficiencySummary:
    """Aggregated metrics for one experimental condition (e.g. B0)."""

    condition: str = ""
    n_questions: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    wall_time_s: float = 0.0  # wall clock of whole eval (batch-parallel)

    # Latency
    question_e2e_mean_s: float = 0.0
    question_e2e_p50_s: float = 0.0
    question_e2e_p95_s: float = 0.0
    ttft_mean_s: float = 0.0  # all calls
    ttft_p50_s: float = 0.0
    ttft_p95_s: float = 0.0
    first_ttft_p50_s: float = 0.0  # first call of each question
    node_e2e_mean_s: float = 0.0
    decision_e2e_mean_s: float = 0.0

    # Throughput
    questions_per_hour: float = 0.0
    correct_per_hour: float = 0.0
    completion_tokens_per_sec: float = 0.0

    # Communication
    llm_calls_mean: float = 0.0
    active_agents_mean: float = 0.0
    edges_mean: float = 0.0
    spatial_edges_mean: float = 0.0
    query_edges_mean: float = 0.0
    debate_edges_mean: float = 0.0
    peer_context_chars_mean: float = 0.0

    # Efficiency ratios
    accuracy_per_llm_call: float = 0.0
    accuracy_per_question_e2e_s: float = 0.0

    total_cost_usd: float = 0.0
    total_completion_tokens: int = 0
    total_prompt_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def print_report(self) -> None:
        print("\n" + "=" * 72)
        print(f"Efficiency Report — condition={self.condition or '(unnamed)'}")
        print("=" * 72)
        print(f"  Accuracy:              {self.accuracy:.4f} ({self.n_correct}/{self.n_questions})")
        print(f"  Wall time:             {self.wall_time_s:.1f}s ({self.wall_time_s/60:.1f} min)")
        print("-" * 72)
        print("  Latency")
        print(f"    Question E2E mean/p50/p95: {self.question_e2e_mean_s:.2f} / "
              f"{self.question_e2e_p50_s:.2f} / {self.question_e2e_p95_s:.2f} s")
        print(f"    TTFT (all calls) mean/p50/p95: {self.ttft_mean_s:.2f} / "
              f"{self.ttft_p50_s:.2f} / {self.ttft_p95_s:.2f} s")
        print(f"    First-call TTFT p50:  {self.first_ttft_p50_s:.2f} s")
        print(f"    Node E2E mean:        {self.node_e2e_mean_s:.2f} s")
        print(f"    Decision E2E mean:    {self.decision_e2e_mean_s:.2f} s")
        print("-" * 72)
        print("  Throughput")
        print(f"    Questions / hour:     {self.questions_per_hour:.2f}")
        print(f"    Correct  / hour:      {self.correct_per_hour:.2f}")
        print(f"    Completion tok / s:   {self.completion_tokens_per_sec:.2f}")
        print("-" * 72)
        print("  Communication")
        print(f"    LLM calls / question: {self.llm_calls_mean:.2f}")
        print(f"    Active agents mean:   {self.active_agents_mean:.2f}")
        print(f"    Edges mean (s/q/d):   {self.edges_mean:.2f} "
              f"({self.spatial_edges_mean:.2f}/{self.query_edges_mean:.2f}/{self.debate_edges_mean:.2f})")
        print(f"    Peer context chars:   {self.peer_context_chars_mean:.0f}")
        print("-" * 72)
        print("  Efficiency ratios")
        print(f"    Acc / LLM call:       {self.accuracy_per_llm_call:.4f}")
        print(f"    Acc / question-E2E s: {self.accuracy_per_question_e2e_s:.4f}")
        print(f"  Cost: ${self.total_cost_usd:.4f} | "
              f"tokens prompt={self.total_prompt_tokens:,} completion={self.total_completion_tokens:,}")
        print("=" * 72)


def aggregate_question_metrics(
    questions: List[QuestionMetrics],
    *,
    condition: str = "",
    wall_time_s: float = 0.0,
    total_cost_usd: float = 0.0,
) -> EfficiencySummary:
    n = len(questions)
    n_correct = sum(1 for q in questions if q.correct)
    accuracy = n_correct / n if n else 0.0

    q_e2e = [q.question_e2e_s for q in questions]
    first_ttfts = [q.first_ttft_s for q in questions if q.llm_calls]
    all_ttfts = [c.ttft_s for q in questions for c in q.llm_calls]
    all_node_e2e = [c.e2e_s for q in questions for c in q.llm_calls if not c.is_decision]
    decision_e2e = [c.e2e_s for q in questions for c in q.llm_calls if c.is_decision]
    if not decision_e2e:
        decision_e2e = [q.decision_e2e_s for q in questions if q.decision_e2e_s > 0]

    total_completion = sum(q.completion_tokens for q in questions)
    total_prompt = sum(q.prompt_tokens for q in questions)
    wall = wall_time_s if wall_time_s > 0 else sum(q_e2e)

    llm_calls = [float(q.num_llm_calls) for q in questions]
    summary = EfficiencySummary(
        condition=condition,
        n_questions=n,
        n_correct=n_correct,
        accuracy=accuracy,
        wall_time_s=wall,
        question_e2e_mean_s=_mean(q_e2e),
        question_e2e_p50_s=_percentile(q_e2e, 50),
        question_e2e_p95_s=_percentile(q_e2e, 95),
        ttft_mean_s=_mean(all_ttfts),
        ttft_p50_s=_percentile(all_ttfts, 50),
        ttft_p95_s=_percentile(all_ttfts, 95),
        first_ttft_p50_s=_percentile(first_ttfts, 50),
        node_e2e_mean_s=_mean(all_node_e2e),
        decision_e2e_mean_s=_mean(decision_e2e),
        questions_per_hour=(n / wall * 3600.0) if wall > 0 else 0.0,
        correct_per_hour=(n_correct / wall * 3600.0) if wall > 0 else 0.0,
        completion_tokens_per_sec=(total_completion / wall) if wall > 0 else 0.0,
        llm_calls_mean=_mean(llm_calls),
        active_agents_mean=_mean([float(q.active_agents) for q in questions]),
        edges_mean=_mean([float(q.total_edges) for q in questions]),
        spatial_edges_mean=_mean([float(q.spatial_edges) for q in questions]),
        query_edges_mean=_mean([float(q.query_edges) for q in questions]),
        debate_edges_mean=_mean([float(q.debate_edges) for q in questions]),
        peer_context_chars_mean=_mean([float(q.peer_context_chars) for q in questions]),
        accuracy_per_llm_call=(accuracy / _mean(llm_calls)) if _mean(llm_calls) > 0 else 0.0,
        accuracy_per_question_e2e_s=(accuracy / _mean(q_e2e)) if _mean(q_e2e) > 0 else 0.0,
        total_cost_usd=total_cost_usd,
        total_completion_tokens=total_completion,
        total_prompt_tokens=total_prompt,
    )
    return summary


def save_metrics(
    path: str | Path,
    summary: EfficiencySummary,
    questions: Optional[List[QuestionMetrics]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary.to_dict(),
        "questions": [q.to_dict() for q in questions] if questions else [],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Efficiency metrics saved to: {path}")
    return path


def load_metrics_summary(path: str | Path) -> EfficiencySummary:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    s = data["summary"] if isinstance(data, dict) and "summary" in data else data
    return EfficiencySummary(**{k: v for k, v in s.items() if k in EfficiencySummary.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Success criteria for B0–B5 (relative to named baseline conditions)
# ---------------------------------------------------------------------------

# Interpreted by ``evaluate_success_criteria``. Keys are criterion ids.
# ``vs`` names must match ``EfficiencySummary.condition`` in baseline files.
DEFAULT_SUCCESS_CRITERIA: List[Dict[str, Any]] = [
    {
        "id": "comm_efficiency",
        "description": "Communication volume down ≥30% vs B1 FullConnected, accuracy drop ≤2pp",
        "metric": "llm_calls_mean",
        "vs": "B1",
        "min_reduction_pct": 30.0,
        "max_accuracy_drop_pp": 2.0,
    },
    {
        "id": "peer_context",
        "description": "Peer-context chars down ≥30% vs B1",
        "metric": "peer_context_chars_mean",
        "vs": "B1",
        "min_reduction_pct": 30.0,
    },
    {
        "id": "ttft",
        "description": "First-call TTFT p50 not worse than B1 (≤ B1 × 1.05)",
        "metric": "first_ttft_p50_s",
        "vs": "B1",
        "max_increase_pct": 5.0,
    },
    {
        "id": "question_e2e",
        "description": "Question E2E p50 down vs B1",
        "metric": "question_e2e_p50_s",
        "vs": "B1",
        "min_reduction_pct": 0.0,  # any reduction counts; tighten later
        "require_strict_lower": True,
    },
    {
        "id": "throughput_vs_b1",
        "description": "Correct answers / hour higher than B1",
        "metric": "correct_per_hour",
        "vs": "B1",
        "require_strict_higher": True,
    },
    {
        "id": "throughput_vs_b3",
        "description": "Correct answers / hour higher than B3 (untrained RGCN)",
        "metric": "correct_per_hour",
        "vs": "B3",
        "require_strict_higher": True,
    },
    {
        "id": "quality_vs_b3",
        "description": "Accuracy ≥ B3 (untrained)",
        "metric": "accuracy",
        "vs": "B3",
        "require_not_lower": True,
    },
]


def evaluate_success_criteria(
    candidate: EfficiencySummary,
    baselines: Dict[str, EfficiencySummary],
    criteria: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return per-criterion verdicts. ``baselines`` keyed by condition name (B1, B3, …)."""
    criteria = criteria or DEFAULT_SUCCESS_CRITERIA
    results = []
    for crit in criteria:
        vs_name = crit["vs"]
        baseline = baselines.get(vs_name)
        metric = crit["metric"]
        row: Dict[str, Any] = {
            "id": crit["id"],
            "description": crit.get("description", ""),
            "metric": metric,
            "vs": vs_name,
            "candidate": getattr(candidate, metric, None),
            "baseline": getattr(baseline, metric, None) if baseline else None,
            "status": "SKIP",
            "detail": "",
        }
        if baseline is None:
            row["detail"] = f"baseline condition '{vs_name}' not provided"
            results.append(row)
            continue

        cand_v = float(getattr(candidate, metric))
        base_v = float(getattr(baseline, metric))
        cand_acc = float(candidate.accuracy)
        base_acc = float(baseline.accuracy)

        passed = True
        details = []

        if "min_reduction_pct" in crit:
            if base_v <= 0:
                passed = False
                details.append("baseline metric is 0; cannot measure reduction")
            else:
                reduction = (base_v - cand_v) / base_v * 100.0
                need = float(crit["min_reduction_pct"])
                ok = reduction >= need
                passed = passed and ok
                details.append(f"reduction={reduction:.1f}% (need ≥{need}%)")

        if crit.get("require_strict_lower"):
            ok = cand_v < base_v
            passed = passed and ok
            details.append(f"candidate {cand_v:.4f} < baseline {base_v:.4f}: {ok}")

        if crit.get("require_strict_higher"):
            ok = cand_v > base_v
            passed = passed and ok
            details.append(f"candidate {cand_v:.4f} > baseline {base_v:.4f}: {ok}")

        if crit.get("require_not_lower"):
            ok = cand_v + 1e-12 >= base_v
            passed = passed and ok
            details.append(f"candidate {cand_v:.4f} ≥ baseline {base_v:.4f}: {ok}")

        if "max_increase_pct" in crit:
            if base_v <= 0:
                ok = cand_v <= 0
            else:
                increase = (cand_v - base_v) / base_v * 100.0
                ok = increase <= float(crit["max_increase_pct"])
                details.append(f"increase={increase:.1f}% (max {crit['max_increase_pct']}%)")
            passed = passed and ok

        if "max_accuracy_drop_pp" in crit:
            drop_pp = (base_acc - cand_acc) * 100.0
            ok = drop_pp <= float(crit["max_accuracy_drop_pp"])
            passed = passed and ok
            details.append(f"acc_drop={drop_pp:.2f}pp (max {crit['max_accuracy_drop_pp']}pp)")

        row["status"] = "PASS" if passed else "FAIL"
        row["detail"] = "; ".join(details)
        results.append(row)
    return results


def print_success_criteria(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("Success Criteria (B0–B5 efficiency goals)")
    print("=" * 72)
    for r in results:
        print(f"  [{r['status']}] {r['id']}: {r['description']}")
        print(f"         metric={r['metric']} vs {r['vs']}: "
              f"candidate={r['candidate']} baseline={r['baseline']}")
        if r["detail"]:
            print(f"         {r['detail']}")
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    print("-" * 72)
    print(f"  Summary: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP")
    print("=" * 72)
