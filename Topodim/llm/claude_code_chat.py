"""
Claude Code backend.

Instead of a single non-agentic chat completion, each topology node becomes a
real **Claude Code agent session**: it can use tools (Bash/Read/Edit/Write/…),
reason over multiple turns, and return a final answer. This drops in where the
"toy agent" used to call ``self.llm.agen(messages)`` — the R-GCN topology
generation, Kahn DAG ordering, peer-context prompt assembly (in AnalyzeAgent),
and REINFORCE training all stay unchanged; only the *executor* behind ``agen``
is swapped.

Selected by including "claude" in the TopoDIM ``--llm_name`` CLI flag
(e.g. ``--llm_name claude-code``); see ``Topodim/llm/llm_registry.py``.

The Claude Agent SDK (``claude-agent-sdk``) shells out to the installed
``claude`` CLI, which inherits the ambient Anthropic-protocol env
(``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` / ``ANTHROPIC_MODEL``).

Configuration (all optional env vars):

    CLAUDE_CODE_PERMISSION_MODE  default "bypassPermissions"  (full autonomy)
    CLAUDE_CODE_ALLOWED_TOOLS    default unset (all tools)    comma-sep allowlist, e.g. "Bash,Read"
    CLAUDE_CODE_WORKDIR          default fresh tempdir        pin a shared sandbox dir
    CLAUDE_CODE_MAX_TURNS        default 8                    agentic loop cap per node
    CLAUDE_CODE_MODEL            default ANTHROPIC_MODEL      override model for the session

Each ``agen`` call runs in a **sandboxed working directory** (a fresh tempdir
unless ``CLAUDE_CODE_WORKDIR`` is set) so node agents can never clobber the
repo — safe under the concurrent ``asyncio.gather`` batches used in training.
"""

import os
import shutil
import tempfile
import time
from typing import List, Union, Optional, Dict, Any

from dotenv import load_dotenv

from Topodim.llm.format import Message
from Topodim.llm.llm import LLM
from Topodim.llm.llm_registry import LLMRegistry
from Topodim.utils.globals import (
    PromptTokens,
    CompletionTokens,
    Cost,
)
from Topodim.utils.efficiency_metrics import (
    consume_pending_peer_chars,
    get_active_question_metrics,
)

load_dotenv()


def _split_messages(messages: List[Union[Message, dict]]) -> tuple[str, str]:
    """Return (system_text, user_text) from a list of Message/dict.

    Mirrors the system/conversation split done in sglang_chat.py. Claude Code
    takes a single prompt; the role constraint is passed as an appended system
    prompt and the rest as the user prompt.
    """
    system_parts: List[str] = []
    user_parts: List[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = msg.role
            content = msg.content
        if role not in ("system", "user", "assistant"):
            role = "user"
        if role == "system":
            system_parts.append(content)
        else:
            user_parts.append(f"[{role}]\n{content}" if role == "assistant" else content)
    return ("\n\n".join(system_parts), "\n\n".join(user_parts))


def _build_options(workdir: str):
    """Construct ClaudeAgentOptions from env config.

    Imported lazily so environments without ``claude-agent-sdk`` can still
    import this module (and use other backends).

    Note: we intentionally do NOT set ``system_prompt`` — that would *replace*
    Claude Code's default system prompt (which carries the tool-use / agent
    instructions) and break agentic behavior. The role constraint is instead
    folded into the user prompt (see ``_run_claude_code``), preserving the
    default system prompt.
    """
    from claude_agent_sdk import ClaudeAgentOptions

    kwargs: Dict[str, Any] = {
        "permission_mode": os.getenv("CLAUDE_CODE_PERMISSION_MODE", "bypassPermissions"),
        "cwd": workdir,
        "max_turns": int(os.getenv("CLAUDE_CODE_MAX_TURNS", "8")),
    }

    allowed = os.getenv("CLAUDE_CODE_ALLOWED_TOOLS", "").strip()
    if allowed:
        # Non-empty allowlist restricts tools; leaving it unset (= []) means
        # the full default tool set is available.
        kwargs["allowed_tools"] = [t.strip() for t in allowed.split(",") if t.strip()]

    model = os.getenv("CLAUDE_CODE_MODEL", "").strip()
    if model:
        kwargs["model"] = model

    # The runtime container runs as root, so Claude Code refuses
    # ``--dangerously-skip-permissions`` (and thus bypassPermissions) unless
    # IS_SANDBOX=1 is set. TopoDIM runs in containerized sandboxes, so this is
    # the intended signal. Disable via CLAUDE_CODE_IS_SANDBOX=0 if undesired.
    env_overrides: Dict[str, str] = {}
    if os.getenv("CLAUDE_CODE_IS_SANDBOX", "1") not in ("0", "", "false", "False"):
        env_overrides["IS_SANDBOX"] = "1"
    # Offline/LAN-gateway defaults: spawned sessions must not hang on
    # GitHub-hosted plugin marketplace sync when github.com is unreachable.
    # Shell env takes precedence; CLAUDE_CODE_ENV_JSON (merged below) wins last.
    offline_defaults = {
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
        "CLAUDE_CODE_PLUGIN_KEEP_MARKETPLACE_ON_FAILURE": "1",
        "CLAUDE_CODE_PLUGIN_GIT_TIMEOUT_MS": "15000",
    }
    for key, value in offline_defaults.items():
        if key not in os.environ:
            env_overrides.setdefault(key, value)
    extra_env = os.getenv("CLAUDE_CODE_ENV_JSON", "").strip()
    if extra_env:
        import json
        env_overrides.update(json.loads(extra_env))
    if env_overrides:
        kwargs["env"] = env_overrides

    return ClaudeAgentOptions(**kwargs)


async def _run_claude_code(system_text: str, user_text: str, workdir: str) -> tuple[str, dict, float, float]:
    """Run one Claude Code agent session.

    Returns (result_text, usage_dict, cost_usd, ttft_s).
    ``ttft_s`` is seconds from query start to the first assistant text block
    (best-effort TTFT proxy for the Claude Code agentic session).
    """
    from claude_agent_sdk import query, ResultMessage, AssistantMessage, TextBlock

    options = _build_options(workdir)

    # Fold the role/system constraint into the user prompt so Claude Code's
    # default system prompt (tool/agent instructions) stays intact.
    prompt = f"{system_text}\n\n{user_text}" if system_text else user_text

    result_text = ""
    usage: dict = {}
    cost = 0.0
    ttft_s = 0.0
    t0 = time.perf_counter()
    saw_first_text = False

    try:
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, ResultMessage):
                result_text = getattr(msg, "result", "") or ""
                usage = getattr(msg, "usage", {}) or {}
                cost = float(getattr(msg, "total_cost_usd", 0.0) or 0.0)
            elif isinstance(msg, AssistantMessage):
                # Fallback: accumulate text blocks in case no result message arrives.
                for block in (getattr(msg, "content", []) or []):
                    text = ""
                    if isinstance(block, TextBlock):
                        text = getattr(block, "text", "") or ""
                    elif getattr(block, "text", None):
                        text = block.text or ""
                    if text:
                        if not saw_first_text:
                            ttft_s = time.perf_counter() - t0
                            saw_first_text = True
                        result_text += text
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never crash the eval
        # Claude Code can raise on a hard stop (e.g. "Reached maximum number of
        # turns (N)") or a transport error. We keep whatever text was produced
        # by the assistant turns so far, so a single over-long node degrades to
        # a partial answer instead of aborting the whole run.
        reason = str(exc)
        if "maximum number of turns" in reason:
            print(
                f"[ClaudeCodeChat] node hit max_turns; returning partial "
                f"answer ({len(result_text)} chars)."
            )
        else:
            print(
                f"[ClaudeCodeChat] query raised ({type(exc).__name__}); "
                f"returning partial answer ({len(result_text)} chars): {reason[:200]}"
            )

    if not saw_first_text:
        # No streamed text (result-only or empty); treat full wait as TTFT upper bound.
        ttft_s = time.perf_counter() - t0

    return result_text, usage, cost, ttft_s


@LLMRegistry.register("ClaudeCodeChat")
class ClaudeCodeChat(LLM):
    """LLM backend that runs each call as a sandboxed Claude Code agent session."""

    def __init__(self, model_name: str = ""):
        # The actual model is taken from ANTHROPIC_MODEL (or CLAUDE_CODE_MODEL);
        # the flag value is kept for diagnostics only.
        self.flag_name = model_name

    async def agen(
        self,
        messages: List[Union[Message, dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> str:
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        system_text, user_text = _split_messages(messages)
        peer_chars = consume_pending_peer_chars()
        prompt_chars = len(system_text) + len(user_text)

        # Sandboxed working directory: fresh tempdir per call (safe under
        # concurrent batches) unless CLAUDE_CODE_WORKDIR pins a shared one.
        workdir = os.getenv("CLAUDE_CODE_WORKDIR", "").strip()
        own_tempdir = False
        if not workdir:
            workdir = tempfile.mkdtemp(prefix="topodim_cc_")
            own_tempdir = True

        t0 = time.perf_counter()
        try:
            result_text, usage, cost, ttft_s = await _run_claude_code(
                system_text, user_text, workdir
            )
        finally:
            if own_tempdir:
                shutil.rmtree(workdir, ignore_errors=True)
        e2e_s = time.perf_counter() - t0

        prompt_tokens = 0
        completion_tokens = 0
        # Token / cost accounting (best-effort; usage may be empty on failure).
        if usage:
            prompt_tokens = (
                int(usage.get("input_tokens", 0))
                + int(usage.get("cache_read_input_tokens", 0))
                + int(usage.get("cache_creation_input_tokens", 0))
            )
            completion_tokens = int(usage.get("output_tokens", 0))
            PromptTokens.instance().value += prompt_tokens
            CompletionTokens.instance().value += completion_tokens
        Cost.instance().value += cost

        qm = get_active_question_metrics()
        if qm is not None:
            from Topodim.utils.efficiency_metrics import LLMCallMetrics
            qm.add_call(
                LLMCallMetrics(
                    ttft_s=ttft_s,
                    e2e_s=e2e_s,
                    prompt_chars=prompt_chars,
                    completion_chars=len(result_text or ""),
                    peer_context_chars=peer_chars,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost,
                )
            )

        return result_text

    def gen(self, *args, **kwargs):
        # Sync path (Graph.run). The quickstart uses the async path (arun), so
        # this is only for completeness. Guard against a running event loop.
        import asyncio
        try:
            asyncio.get_running_loop()
            # Already inside a loop — can't use asyncio.run; run in a new thread.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, self.agen(*args, **kwargs)).result()
        except RuntimeError:
            return asyncio.run(self.agen(*args, **kwargs))
