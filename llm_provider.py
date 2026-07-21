"""
llm_provider.py: Open-source LLM backend for BioSimulateAI.

Replaces the previous Google Gemini dependency. Two engines, one interface:

  * "local": a GGUF model run **in-process** via llama-cpp-python, auto-downloaded
               from HuggingFace on first use. No separate server, no API key, runs
               on CPU. This is the "bundled" open-source model (Llama / Gemma / Qwen /
               gpt-oss).
  * "remote": any OpenAI-compatible chat endpoint (a self-hosted vLLM/TGI model on
               AWS, Ollama, LM Studio, Groq, OpenRouter, ...). Use this to point at a
               larger cloud-hosted model when accuracy matters most.

Both engines expose chat(messages, ...) and are consumed through generate_json().
Callers fall back to deterministic rule-based logic whenever the LLM is unavailable.
"""

import os
import re
import ast
import json
import threading
from typing import Dict, Any, List, Optional


class LLMError(Exception):
    """Raised when an LLM engine cannot fulfil a request."""


# =============================================================================
# Curated model registry: small, instruction-tuned GGUFs that run on CPU.
# Sizes are approximate on-disk footprints of the Q4_K_M quantization.
# =============================================================================
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "llama-3.2-3b": {
        "label": "Llama 3.2 3B Instruct",
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "params": "3B", "size_gb": 2.0, "recommended": True,
    },
    "gemma-2-2b": {
        "label": "Gemma 2 2B Instruct",
        "repo": "bartowski/gemma-2-2b-it-GGUF",
        "file": "gemma-2-2b-it-Q4_K_M.gguf",
        "params": "2B", "size_gb": 1.7,
    },
    "qwen2.5-3b": {
        "label": "Qwen2.5 3B Instruct",
        "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "file": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "params": "3B", "size_gb": 2.0,
    },
    "llama-3.1-8b": {
        "label": "Llama 3.1 8B Instruct",
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "params": "8B", "size_gb": 4.9,
    },
    "gpt-oss-20b": {
        "label": "gpt-oss 20B (heavy)",
        "repo": "bartowski/openai_gpt-oss-20b-GGUF",
        "file": "openai_gpt-oss-20b-MXFP4.gguf",
        "params": "20B", "size_gb": 12.1, "heavy": True,
    },
}

DEFAULT_MODEL_KEY = os.environ.get("BIOSIM_LLM_MODEL", "llama-3.2-3b")

# All native llama.cpp work (model load, eviction, and inference) touches
# process-global state that is NOT thread-safe. A SINGLE lock therefore serializes
# the entire load+generate sequence. (A split load/gen lock would allow a model to
# be constructed or freed on one thread while another thread is decoding, which is a
# native-level data race that can segfault the whole process.)
_infer_lock = threading.RLock()

_download_state: Dict[str, Dict[str, Any]] = {}
_download_lock = threading.Lock()
_active_download: Optional[str] = None  # at most one weight download at a time


# =============================================================================
# JSON extraction helper (tolerant of stray prose / markdown fences)
# =============================================================================
def _strip_json_comments(s: str) -> str:
    """Remove // line comments and /* */ block comments (leaving :// in URLs alone)."""
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r"(^|[^:])//[^\n\r]*", lambda m: m.group(1), s)
    return s


def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", s)


_SMART_QUOTES = {
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‘": "'", "’": "'", "‛": "'", "′": "'",
}


def _normalize_quotes(s: str) -> str:
    for bad, good in _SMART_QUOTES.items():
        s = s.replace(bad, good)
    return s


def _strip_reasoning(s: str) -> str:
    """Remove reasoning/thinking blocks that reasoning models (e.g. gpt-oss) emit."""
    s = re.sub(r"<(reasoning|think|thinking|analysis)\b[^>]*>.*?</\1>", " ",
               s, flags=re.DOTALL | re.IGNORECASE)
    # Leftover harmony channel tokens like <|channel|>, <|message|>, <|end|>.
    s = re.sub(r"<\|[^>]*\|>", " ", s)
    return s


def _balanced_spans(s: str) -> List[str]:
    """Return top-level balanced {...}/[...] substrings, ignoring braces inside strings."""
    spans: List[str] = []
    stack: List[str] = []
    start = None
    in_str, quote, esc = False, "", False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch == '"':
            # Only double quotes delimit JSON strings; a bare ' is usually a prose
            # apostrophe ("the model's cascade") and must not swallow later braces.
            in_str, quote = True, ch
        elif ch in "{[":
            if not stack:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    spans.append(s[start:i + 1])
                    start = None
    return spans


_JSON_LITERALS = ("true", "false", "null", "True", "False", "None")


def _repairs(s: str):
    """Yield progressively-repaired variants of a candidate JSON string."""
    yield s
    nc = _remove_trailing_commas(_strip_json_comments(s))
    yield nc
    # Quote bare (unquoted) keys:  { key:  ->  { "key":
    qk = re.sub(r"([{\[,]\s*)([A-Za-z_][\w\-]*)(\s*):", r'\1"\2"\3:', nc)
    yield qk
    # Normalize JSON literals so ast.literal_eval can parse lowercase true/false/null.
    py = re.sub(r"\btrue\b", "True", nc)
    py = re.sub(r"\bfalse\b", "False", py)
    py = re.sub(r"\bnull\b", "None", py)
    yield py
    # Last resort: quote bare word values ( : ODE -> : "ODE" ), skipping literals.
    def _qval(m):
        v = m.group(1).strip()
        return m.group(0) if v in _JSON_LITERALS else ': "' + v + '"' + m.group(2)
    yield re.sub(r":\s*([A-Za-z_][\w\- ]*?)\s*([,}\]])", _qval, qk)


def extract_json(text: str) -> Any:
    """
    Parse JSON out of an LLM response, tolerating how open models mangle it:
    <reasoning>/<think> blocks, markdown fences, a prose preamble/epilogue,
    // and /* */ comments, trailing commas, single-quoted or unquoted keys,
    smart/curly quotes, and Python True/False/None.
    """
    raw = (text or "").strip()
    work = _strip_reasoning(raw)
    work = re.sub(r"```[a-zA-Z]*", "", work).replace("```", "")
    work = _normalize_quotes(work).strip()

    # Prefer balanced JSON spans (the final answer usually comes after reasoning):
    # try the last span first, then the longest, then the whole text.
    spans = _balanced_spans(work)
    candidates: List[str] = []
    if spans:
        candidates.append(spans[-1])
        candidates.append(max(spans, key=len))
    candidates.append(work)

    seen, ordered = set(), []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)

    last_err: Optional[Exception] = None
    for cand in ordered:
        for rep in _repairs(cand):
            for parser in (json.loads, ast.literal_eval):
                try:
                    val = parser(rep)
                    if isinstance(val, (dict, list)):
                        return val
                except Exception as e:
                    last_err = e

    snippet = raw[:200].replace("\n", " ")
    raise LLMError(f"Model did not return valid JSON ({last_err}). Output began: {snippet!r}")


# =============================================================================
# Local engine: in-process GGUF via llama-cpp-python
# =============================================================================
class LocalLLM:
    # Only one model is kept resident at a time to conserve RAM.
    _loaded: Dict[str, Any] = {}
    _current_key: Optional[str] = None

    def __init__(self, model_key: str):
        if model_key not in MODEL_REGISTRY:
            raise LLMError(f"Unknown local model '{model_key}'.")
        self.model_key = model_key
        self.spec = MODEL_REGISTRY[model_key]

    @staticmethod
    def runtime_available() -> bool:
        # Both packages are required for the local engine: llama_cpp to run
        # inference and huggingface_hub to fetch weights.
        try:
            import llama_cpp        # noqa: F401
            import huggingface_hub  # noqa: F401
            return True
        except Exception:
            return False

    def is_downloaded(self) -> bool:
        try:
            from huggingface_hub import try_to_load_from_cache
            path = try_to_load_from_cache(self.spec["repo"], self.spec["file"])
            return isinstance(path, str) and os.path.exists(path)
        except Exception:
            return False

    def _cached_path(self) -> str:
        # local_files_only guarantees this NEVER performs network I/O, so it is
        # safe to call from the inference path (no HEAD/revision check, no fetch).
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=self.spec["repo"], filename=self.spec["file"],
            local_files_only=True,
        )

    def download(self) -> str:
        """Network fetch of the weights, used ONLY by the explicit download flow."""
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=self.spec["repo"], filename=self.spec["file"])

    def _load_locked(self):
        """Load (and evict others); the caller MUST hold _infer_lock."""
        from llama_cpp import Llama
        # Never trigger a multi-GB download from the inference path.
        if not self.is_downloaded():
            raise LLMError(
                f"Model '{self.model_key}' is not downloaded yet. "
                f"Download it in AI settings first."
            )
        if LocalLLM._current_key == self.model_key and self.model_key in LocalLLM._loaded:
            return LocalLLM._loaded[self.model_key]
        # Safe to evict here: we hold _infer_lock, so no inference is running.
        LocalLLM._loaded.clear()
        model = Llama(
            model_path=self._cached_path(),
            n_ctx=8192,
            n_threads=max(1, (os.cpu_count() or 4) - 1),
            verbose=False,
        )
        LocalLLM._loaded[self.model_key] = model
        LocalLLM._current_key = self.model_key
        return model

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2,
             max_tokens: int = 16384, json_mode: bool = False) -> str:
        # Local GGUF models are grammar-constrained (clean, short JSON) and bounded
        # by the context window, so cap the request to stay well inside n_ctx=8192.
        kwargs: Dict[str, Any] = dict(
            messages=messages, temperature=temperature, max_tokens=min(int(max_tokens), 4096),
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # Hold ONE lock across load + generation so native llama.cpp state is
        # never touched by two threads at once.
        with _infer_lock:
            model = self._load_locked()
            out = model.create_chat_completion(**kwargs)
        return out["choices"][0]["message"]["content"]


# =============================================================================
# Remote engine: OpenAI-compatible chat endpoint
# =============================================================================
class RemoteLLM:
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None,
                 timeout: int = 180):
        if not base_url:
            raise LLMError("Remote endpoint URL is required.")
        self.base_url = base_url.rstrip("/")
        self.model = model or "default"
        self.api_key = api_key or None
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2,
             max_tokens: int = 16384, json_mode: bool = False) -> str:
        import requests
        # Accept either a full ".../v1" base or a bare host; normalise to the endpoint.
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        base: Dict[str, Any] = {
            "model": self.model, "messages": messages,
            "temperature": temperature, "stream": False,
        }
        # Reasoning models (gpt-oss on Bedrock) honor ONLY max_completion_tokens and
        # a tiny default otherwise cuts them off mid-reasoning; classic OpenAI-compatible
        # servers (old vLLM/Ollama) use max_tokens. Sending both together can 400, so try
        # tiers in order, keeping max_completion_tokens as long as possible.
        def _tier(modern: bool, structured: bool) -> Dict[str, Any]:
            p = dict(base)
            if modern:
                p["max_completion_tokens"] = max_tokens
            else:
                p["max_tokens"] = max_tokens
            if structured and json_mode:
                p["response_format"] = {"type": "json_object"}
                p["reasoning_effort"] = "low"
            return p

        tiers = [
            _tier(modern=True, structured=True),    # full-featured (Bedrock, modern vLLM)
            _tier(modern=True, structured=False),   # Bedrock rejecting response_format/reasoning_effort
            _tier(modern=False, structured=False),  # classic OpenAI-compatible (max_tokens only)
        ]

        def _post(p):
            return requests.post(url, headers=headers, json=p, timeout=self.timeout)

        try:
            r = None
            for p in tiers:
                r = _post(p)
                if r.status_code != 400:
                    break  # 200 (or a non-param error we should surface) -> stop
        except requests.RequestException as e:
            raise LLMError(f"Remote LLM request failed: {e}")

        # Surface the endpoint's actual complaint instead of a generic message.
        if r is None or r.status_code >= 400:
            detail = ""
            try:
                detail = (r.text or "")[:400]
            except Exception:
                pass
            raise LLMError(f"Remote LLM HTTP {getattr(r, 'status_code', '?')}: {detail}")

        try:
            data = r.json()
            choice = data["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or ""
            # Some reasoning models put the answer in reasoning_content when content is empty.
            if not content.strip() and msg.get("reasoning_content"):
                content = msg["reasoning_content"]

            # Definitive truncation diagnostic: whenever the model was cut off by a
            # length limit, report it with the actual token count so we can tell
            # whether max_completion_tokens is being honored.
            finish = (choice.get("finish_reason") or "").lower()
            usage = data.get("usage", {}) or {}
            if finish == "length":
                ct = usage.get("completion_tokens", "?")
                raise LLMError(
                    f"Remote model output was TRUNCATED at {ct} completion tokens "
                    f"(finish_reason=length) even though max_completion_tokens={max_tokens} "
                    f"was requested — the endpoint is capping the completion. "
                    f"Try model gpt-oss-120b, a non-reasoning instruct model, or the local model."
                )
            return content
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Unexpected remote LLM response: {e}")


# =============================================================================
# AWS Bedrock engine: native Converse API (works with ANY Bedrock model, e.g.
# Mistral, Llama, etc.). Auth uses the standard AWS credential chain
# (env vars, ~/.aws/credentials, SSO cache, or an IAM role) — no API key in the app.
# =============================================================================
# Defaults come from the environment (.env / real env vars) when present, so a
# pre-configured .env fully drives the Bedrock engine with no UI input.
BEDROCK_DEFAULT_MODEL = os.getenv("BEDROCK_MODEL_ID", "mistral.mistral-large-3-675b-instruct")
BEDROCK_DEFAULT_REGION = (os.getenv("BEDROCK_REGION") or os.getenv("AWS_DEFAULT_REGION")
                          or os.getenv("AWS_REGION") or "us-east-2")


def _looks_like_placeholder(v: str) -> bool:
    s = (v or "").strip().upper()
    return (not s) or any(tok in s for tok in ("PASTE_", "REPLACE", "YOUR_", "XXXX", "EXAMPLE"))


def bedrock_env_ready() -> bool:
    """True when the environment already carries usable Bedrock credentials:
    a Bedrock API key (AWS_BEARER_TOKEN_BEDROCK), static access keys, or an
    AWS profile / SSO. Placeholder values in a freshly-copied .env are not-ready."""
    tok = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if tok and not _looks_like_placeholder(tok):
        return True
    ak, sk = os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY")
    if ak and sk and not _looks_like_placeholder(ak) and not _looks_like_placeholder(sk):
        return True
    if os.getenv("AWS_PROFILE") or os.getenv("AWS_ROLE_ARN"):
        return True
    return False


class BedrockLLM:
    def __init__(self, model: str, region: str = "", access_key: Optional[str] = None,
                 secret_key: Optional[str] = None, session_token: Optional[str] = None,
                 bearer_token: Optional[str] = None, timeout: int = 180):
        self.model = model or BEDROCK_DEFAULT_MODEL
        self.region = region or BEDROCK_DEFAULT_REGION
        self.timeout = timeout
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise LLMError("boto3 is not installed. Run: pip install boto3  (needed for the AWS Bedrock engine).")

        # A Bedrock API key (bearer token) supplied from the UI is exported to the
        # env var botocore reads at client creation. (When it comes from .env it is
        # already in the environment and this is a no-op.) Bearer auth then replaces
        # SigV4, so no access key / secret is needed.
        bearer_token = (bearer_token or "").strip()
        if bearer_token:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bearer_token

        client_kwargs: Dict[str, Any] = {
            "region_name": self.region,
            "config": Config(read_timeout=timeout, connect_timeout=20,
                             retries={"max_attempts": 2, "mode": "standard"}),
        }
        # Explicit keys (entered in the UI) take precedence; otherwise fall back to the
        # standard AWS credential chain (env vars, ~/.aws, SSO/login cache, IAM role).
        access_key = (access_key or "").strip()
        secret_key = (secret_key or "").strip()
        session_token = (session_token or "").strip()
        if access_key and secret_key:
            client_kwargs["aws_access_key_id"] = access_key
            client_kwargs["aws_secret_access_key"] = secret_key
            if session_token:
                client_kwargs["aws_session_token"] = session_token
        elif access_key or secret_key:
            raise LLMError("Enter BOTH an AWS Access Key ID and a Secret Access Key, "
                           "or leave both blank to use this machine's AWS credentials.")

        try:
            self._client = boto3.client("bedrock-runtime", **client_kwargs)
        except Exception as e:
            # Report only the exception type, never str(e), so no credential material
            # could ever appear in a client-construction error surfaced to the UI.
            raise LLMError(f"Could not create the AWS Bedrock client for region "
                           f"'{self.region}' ({type(e).__name__}).")

    @staticmethod
    def _error_detail(e: Exception) -> str:
        # botocore ClientError carries the useful message under .response["Error"]["Message"];
        # NoCredentials / expired-SSO / endpoint errors are plain strings.
        resp = getattr(e, "response", None)
        if isinstance(resp, dict):
            msg = resp.get("Error", {}).get("Message")
            if msg:
                return msg
        return str(e)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2,
             max_tokens: int = 16384, json_mode: bool = False) -> str:
        # Convert OpenAI-style messages to the Bedrock Converse schema.
        system_blocks: List[Dict[str, str]] = []
        conversation: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "system":
                system_blocks.append({"text": content})
            else:
                conversation.append({
                    "role": "assistant" if role == "assistant" else "user",
                    "content": [{"text": content}],
                })
        if json_mode:
            system_blocks.append({"text": "Respond with a single valid JSON object and nothing else."})

        # Bedrock caps output tokens per model; keep a safe ceiling well under model limits.
        max_out = max(256, min(int(max_tokens), 8192))
        kwargs: Dict[str, Any] = {
            "modelId": self.model,
            "messages": conversation,
            "inferenceConfig": {"maxTokens": max_out, "temperature": float(temperature)},
        }
        if system_blocks:
            kwargs["system"] = system_blocks

        try:
            resp = self._client.converse(**kwargs)
        except Exception as e:
            raise LLMError(f"AWS Bedrock request failed (model '{self.model}', region '{self.region}'): "
                           f"{self._error_detail(e)}")

        try:
            blocks = resp["output"]["message"]["content"]
            text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
            stop = (resp.get("stopReason") or "").lower()
            if stop == "max_tokens" and not text.strip():
                raise LLMError(f"Bedrock output was truncated at maxTokens={max_out} with no usable text.")
            return text
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Unexpected AWS Bedrock response shape: {e}")


# =============================================================================
# Public API
# =============================================================================
def build_client(config: Optional[Dict[str, Any]]):
    """
    Build an LLM client from a config dict:
        {"engine": "local"|"remote"|"off",
         "model": <registry key or remote model name>,
         "base_url": <remote only>, "api_key": <remote only, optional>}
    Raises LLMError if the requested engine is unavailable/misconfigured.
    """
    cfg = config or {}
    engine = (cfg.get("engine") or "off").lower()

    if engine == "local":
        if not LocalLLM.runtime_available():
            raise LLMError("llama-cpp-python is not installed for local inference.")
        key = cfg.get("model") or DEFAULT_MODEL_KEY
        if key not in MODEL_REGISTRY:
            key = DEFAULT_MODEL_KEY
        return LocalLLM(key)

    if engine == "remote":
        return RemoteLLM(cfg.get("base_url", ""), cfg.get("model", ""), cfg.get("api_key"))

    if engine == "bedrock":
        return BedrockLLM(cfg.get("model") or BEDROCK_DEFAULT_MODEL, cfg.get("region", ""),
                          access_key=cfg.get("aws_access_key_id"),
                          secret_key=cfg.get("aws_secret_access_key"),
                          session_token=cfg.get("aws_session_token"),
                          bearer_token=cfg.get("aws_bearer_token"))

    raise LLMError("LLM engine is off.")


def wants_llm(config: Optional[Dict[str, Any]]) -> bool:
    return bool(config) and (config.get("engine") or "off").lower() in ("local", "remote", "bedrock")


def generate_json(client, prompt: str, system: Optional[str] = None,
                  temperature: float = 0.1, max_tokens: int = 16384) -> Any:
    # Generous token budget by default so reasoning models finish thinking AND
    # emit the JSON (the local engine caps this to its context window internally).
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    text = client.chat(messages, temperature=temperature,
                        max_tokens=max_tokens, json_mode=True)
    return extract_json(text)


# =============================================================================
# Model management (listing / downloading) for the API layer
# =============================================================================
def list_models() -> Dict[str, Any]:
    available = LocalLLM.runtime_available()
    models = []
    for key, spec in MODEL_REGISTRY.items():
        downloaded = LocalLLM(key).is_downloaded() if available else False
        models.append({
            "key": key,
            "label": spec["label"],
            "params": spec["params"],
            "size_gb": spec["size_gb"],
            "heavy": spec.get("heavy", False),
            "recommended": spec.get("recommended", False),
            "downloaded": downloaded,
        })
    return {"runtime_available": available, "default": DEFAULT_MODEL_KEY, "models": models}


def start_download(model_key: str) -> Dict[str, Any]:
    global _active_download
    if model_key not in MODEL_REGISTRY:
        raise LLMError(f"Unknown model '{model_key}'.")
    if not LocalLLM.runtime_available():
        raise LLMError("llama-cpp-python / huggingface_hub is not installed.")

    spec = MODEL_REGISTRY[model_key]
    with _download_lock:
        st = _download_state.get(model_key)
        if st and st.get("status") == "downloading":
            return download_status(model_key)
        # Only one weight download at a time (prevents a burst of concurrent
        # multi-GB fetches from saturating bandwidth / filling the disk).
        if _active_download is not None and _active_download != model_key:
            raise LLMError(
                f"A model download is already in progress ({_active_download}). "
                f"Please wait for it to finish."
            )
        # Refuse if the disk clearly cannot hold the weights.
        try:
            import shutil
            free_gb = shutil.disk_usage(os.path.expanduser("~")).free / (1024 ** 3)
            need_gb = spec["size_gb"] * 1.2 + 1.0
            if free_gb < need_gb:
                raise LLMError(
                    f"Not enough free disk space (~{free_gb:.0f} GB free, "
                    f"need ~{need_gb:.0f} GB for {spec['label']})."
                )
        except LLMError:
            raise
        except Exception:
            pass  # if we can't check, proceed rather than block
        _active_download = model_key
        _download_state[model_key] = {"status": "downloading", "error": None}

    def _run():
        global _active_download
        try:
            LocalLLM(model_key).download()
            _download_state[model_key] = {"status": "done", "error": None}
        except Exception as e:  # pragma: no cover - network dependent
            _download_state[model_key] = {"status": "error", "error": str(e)}
        finally:
            with _download_lock:
                _active_download = None

    threading.Thread(target=_run, daemon=True).start()
    return download_status(model_key)


def download_status(model_key: str) -> Dict[str, Any]:
    downloaded = (
        LocalLLM(model_key).is_downloaded()
        if (model_key in MODEL_REGISTRY and LocalLLM.runtime_available())
        else False
    )
    st = _download_state.get(model_key, {"status": "idle", "error": None})
    if downloaded:
        st = {"status": "done", "error": None}
    return {"model": model_key, "downloaded": downloaded, **st}
