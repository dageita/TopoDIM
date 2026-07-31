import os
import json
import math
import time
import asyncio
from typing import Union, Literal, Optional, Iterator, List, Any, Dict, Tuple
from tqdm import tqdm
import copy

from Topodim.graph.graph import Graph
from experiments.accuracy import Accuracy
from Topodim.utils.globals import Cost, PromptTokens, CompletionTokens
from Topodim.utils.efficiency_metrics import (
    QuestionMetrics,
    aggregate_question_metrics,
    save_metrics,
    load_metrics_summary,
    evaluate_success_criteria,
    print_success_criteria,
    DEFAULT_SUCCESS_CRITERIA,
)


async def evaluate(
        graph: Graph,
        dataset,
        num_rounds: int = 1,
        limit_questions: Optional[int] = None,
        eval_batch_size: int = 4,
        condition: str = "",
        metrics_out: Optional[str] = None,
        baseline_metrics: Optional[Dict[str, str]] = None,
        ) -> Tuple[float, Any]:
    """Evaluate and collect efficiency metrics.

    Returns ``(accuracy, EfficiencySummary)``.
    ``baseline_metrics`` maps condition name → path of a previously saved
    metrics JSON (e.g. ``{"B1": "result/metrics_B1.json"}``) for success-criteria checks.
    """

    print(f"Evaluating gdesigner on {dataset.__class__.__name__} split {dataset.split}")
    if condition:
        print(f"Condition: {condition}")
    
    graph.rgcn.eval()
    accuracy = Accuracy()
    all_question_metrics: List[QuestionMetrics] = []
    eval_wall_t0 = time.perf_counter()

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None:
                if i_record >= limit_questions:
                    break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if len(records) > 0:
            yield records
        return
    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / eval_batch_size))

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
        print(80*'-')

        start_ts = time.time()
        answer_log_probs = []
        realized_graphs = []
        input_dicts = []
        
        for record in record_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.rgcn = graph.rgcn
            input_dict = dataset.record_to_input(record)
            input_dicts.append(input_dict)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
            realized_graphs.append(realized_graph)
            
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs,*_ = zip(*raw_results)
        batch_wall = time.time() - start_ts
        print(f"Batch time {batch_wall:.3f}")
        
        print("\n" + "="*80)
        print("📊 Detailed Evaluation Results:")
        print("="*80)
        
        for idx, (raw_answer, record, realized_graph, input_dict) in enumerate(zip(
            raw_answers, record_batch, realized_graphs, input_dicts
        )):
            answer = dataset.postprocess_answer(raw_answer)
            correct_answer = dataset.record_to_target_answer(record)
            is_correct = answer == correct_answer
            status = "✅" if is_correct else "❌"
            sample_id = i_batch * eval_batch_size + idx
            
            print(f"\n{status} Sample {sample_id}:")
            print(f"  Question: {input_dict['task'][:150]}...")
            
            print(f"  Topology Execution Order:")
            topo_summary = realized_graph.get_execution_summary()
            for line in topo_summary.split('\n'):
                print(f"    {line}")
            
            print(f"  Predicted Answer: '{answer}'")
            print(f"  Correct Answer:   '{correct_answer}'")

            qm = getattr(realized_graph, "efficiency_metrics", None)
            if qm is not None:
                qm.sample_id = sample_id
                qm.correct = bool(is_correct)
                print(
                    f"  Efficiency: q_e2e={qm.question_e2e_s:.2f}s "
                    f"ttft0={qm.first_ttft_s:.2f}s "
                    f"calls={qm.num_llm_calls} "
                    f"agents={qm.active_agents} "
                    f"edges={qm.total_edges} "
                    f"(s={qm.spatial_edges}/q={qm.query_edges}/d={qm.debate_edges}) "
                    f"peer_chars={qm.peer_context_chars}"
                )
                all_question_metrics.append(qm)
            
            accuracy.update(answer, correct_answer)
        
        print("\n" + "="*80)
        print(f"Current Accuracy: {accuracy.get():.4f} ({accuracy._num_correct}/{accuracy._num_total})")
        print(f"Cost: ${Cost.instance().value:.4f}")
        print(f"Tokens: {int(PromptTokens.instance().value):,} prompt + {int(CompletionTokens.instance().value):,} completion")
        print("="*80)

    wall_time_s = time.perf_counter() - eval_wall_t0
    accuracy.print()

    summary = aggregate_question_metrics(
        all_question_metrics,
        condition=condition,
        wall_time_s=wall_time_s,
        total_cost_usd=float(Cost.instance().value),
    )
    summary.print_report()

    # Always print the criteria definition; compare when baselines are supplied.
    print("\n" + "=" * 72)
    print("Configured Success Criteria (for B0–B5)")
    print("=" * 72)
    for crit in DEFAULT_SUCCESS_CRITERIA:
        print(f"  • [{crit['id']}] {crit['description']}")
    print("=" * 72)

    if baseline_metrics:
        baselines = {}
        for name, path in baseline_metrics.items():
            if path and os.path.exists(path):
                baselines[name] = load_metrics_summary(path)
            else:
                print(f"Warning: baseline metrics for {name} not found at {path}")
        verdicts = evaluate_success_criteria(summary, baselines)
        print_success_criteria(verdicts)
        if metrics_out:
            verdict_path = os.path.splitext(metrics_out)[0] + "_criteria.json"
            with open(verdict_path, "w", encoding="utf-8") as f:
                json.dump(verdicts, f, ensure_ascii=False, indent=2)
            print(f"Success criteria verdicts saved to: {verdict_path}")

    if metrics_out:
        save_metrics(metrics_out, summary, all_question_metrics)

    return accuracy.get(), summary


def dump_eval_results(self, dct: Dict[str, Any]) -> None:
    if self._art_dir_name is not None:
        eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
        with open(eval_json_name, "w") as f:
            json.dump(dct, f)
