"""
SGLang backend over the **Anthropic-compatible** protocol.

SGLang can serve an Anthropic-compatible endpoint (the same one used by
Claude Code / the Anthropic SDK). TopoDIM's default backend is Ollama, but
when the model is hosted in a separate SGLang container we route here.

Configuration is read from the environment (mirrors the Claude Code env
vars used to point at a local SGLang gateway):

    ANTHROPIC_BASE_URL   e.g. http://10.121.129.19:30822
    ANTHROPIC_AUTH_TOKEN e.g. sk-local   (any non-empty value works if the
                                         gateway does not verify it)
    ANTHROPIC_MODEL      e.g. Hy3-GPTQ-Int4

The model is selected by including "sglang" in the TopoDIM --llm_name CLI
flag (see Topodim/llm/llm_registry.py). The actual served model name is
taken from ANTHROPIC_MODEL, NOT from the flag, because the flag is also
used to pick the backend branch.
"""

import os
from typing import List, Union, Optional, Dict, Any

import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from dotenv import load_dotenv

from Topodim.llm.format import Message
from Topodim.llm.llm import LLM
from Topodim.llm.llm_registry import LLMRegistry
from Topodim.utils.globals import (
    PromptTokens,
    CompletionTokens,
    Cost,
)

load_dotenv()

_ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "")
_ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "")


def _approx_tokens(text: str) -> int:
    """Rough token count via tiktoken (avoids depending on a specific
    model's HF tokenizer). Good enough for usage accounting only."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


@retry(
    reraise=True,
    stop=stop_after_attempt(8),
    wait=wait_exponential_jitter(exp_base=2, max=8),
)
async def achat_sglang(
    model: str,
    msg: List[Dict[str, str]],
    *,
    max_tokens: int = 3000,
    temperature: float = 0.2,
) -> str:
    """Call the SGLang Anthropic-compatible endpoint.

    Imported lazily so that environments without the `anthropic` package
    installed can still import this module (and use other backends).
    """
    from anthropic import AsyncAnthropic

    if not _ANTHROPIC_BASE_URL:
        raise RuntimeError(
            "ANTHROPIC_BASE_URL is not set. Point it at your SGLang "
            "Anthropic-compatible endpoint, e.g. http://HOST:PORT"
        )

    client = AsyncAnthropic(
        base_url=_ANTHROPIC_BASE_URL,
        api_key=_ANTHROPIC_AUTH_TOKEN or "sk-local",
    )

    system_messages = [m["content"] for m in msg if m["role"] == "system"]
    convo = [m for m in msg if m["role"] != "system"]

    system_text = "\n\n".join(system_messages) if system_messages else None

    completion = await client.messages.create(
        model=model or _ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_text,
        messages=convo,
    )

    content = "".join(
        block.text for block in completion.content if hasattr(block, "text")
    )

    prompt = "".join([m["content"] for m in msg])
    PromptTokens.instance().value += _approx_tokens(prompt)
    CompletionTokens.instance().value += _approx_tokens(content)
    # Local SGLang has no priced cost; keep Cost at 0.
    Cost.instance().value += 0.0

    return content


@LLMRegistry.register("SGLangChat")
class SGLangChat(LLM):
    """LLM backend for a SGLang server speaking the Anthropic protocol."""

    def __init__(self, model_name: str = ""):
        # The served model is taken from ANTHROPIC_MODEL, but we keep the
        # flag value around for diagnostics.
        self.flag_name = model_name
        self.model_name = _ANTHROPIC_MODEL or model_name

    async def agen(
        self,
        messages: List[Union[Message, dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> str:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        msg_dicts = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = msg.role
                content = msg.content
            if role not in ("system", "user", "assistant"):
                role = "user"
            msg_dicts.append({"role": role, "content": content})

        return await achat_sglang(
            self.model_name,
            msg_dicts,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def gen(self, *args, **kwargs):
        raise NotImplementedError("Use agen() for SGLangChat (async only).")
