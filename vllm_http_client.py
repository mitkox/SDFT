"""
Lightweight HTTP client for external vLLM servers (OpenAI-compatible API).

This avoids installing vLLM locally when you already run a teacher model via `vllm serve`.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch


class VLLMClient:
    """HTTP client for an external vLLM server that mimics TRL's `VLLMClient` interface."""

    def __init__(
        self,
        base_url: str,
        connection_timeout: float = 600.0,
        tokenizer=None,
        default_model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_concurrent_requests: int = 1,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(connection_timeout)
        self.session = requests.Session()
        self.tokenizer = tokenizer
        self.default_model = default_model
        self.api_key = api_key
        self.max_concurrent_requests = int(max_concurrent_requests) if max_concurrent_requests else 1

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def wait_until_ready(self, timeout_s: float = 240.0, poll_interval_s: float = 1.0) -> None:
        """Poll `/v1/models` until the server responds or timeout is reached."""
        deadline = time.time() + float(timeout_s)
        last_exc: Optional[BaseException] = None
        while time.time() < deadline:
            try:
                resp = self.session.get(
                    f"{self.base_url}/v1/models",
                    headers=self._headers(),
                    timeout=min(self.timeout, poll_interval_s + 5.0),
                )
                resp.raise_for_status()
                return
            except Exception as exc:  # noqa: BLE001 - surface as ConnectionError below
                last_exc = exc
                time.sleep(poll_interval_s)
        raise ConnectionError(f"Timed out waiting for vLLM server at {self.base_url}") from last_exc

    def init_communicator(self, device: Optional[torch.device] = None, startup_timeout_s: float = 240.0):
        """Initialize communicator (no-op) but optionally waits for server readiness."""
        self.wait_until_ready(timeout_s=startup_timeout_s)

    def update_named_param(self, name: str, param: torch.Tensor):
        """Update model parameters (no-op for external server)."""
        return

    def reset_prefix_cache(self):
        """Reset prefix cache (no-op for external server)."""
        return

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def list_models(self) -> List[str]:
        resp = self.session.get(f"{self.base_url}/v1/models", headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", []) if isinstance(data, dict) else []
        return [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]

    def _maybe_truncate_prompt(self, prompt_text: str, truncate_prompt_tokens: Optional[int]) -> Tuple[str, List[int]]:
        if self.tokenizer is None:
            return prompt_text, []
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if truncate_prompt_tokens is not None and len(prompt_ids) > truncate_prompt_tokens:
            prompt_ids = prompt_ids[-truncate_prompt_tokens:]
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        return prompt_text, prompt_ids

    def _post_completions(
        self,
        model: str,
        prompt_text: str,
        n: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        max_tokens: int,
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], Optional[List[List[float]]]]:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt_text,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": n,
            "echo": False,
            "logprobs": 0,
        }
        if top_k > 0:
            payload["top_k"] = top_k
        if min_p and min_p > 0:
            payload["min_p"] = min_p
        if repetition_penalty != 1.0:
            payload["repetition_penalty"] = repetition_penalty
        if generation_kwargs:
            payload.update(generation_kwargs)

        resp = self.session.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", []) if isinstance(data, dict) else []

        texts: List[str] = []
        token_logprobs: Optional[List[List[float]]] = []
        any_logprobs = False

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            texts.append((choice.get("text") or "").strip())
            lp = choice.get("logprobs") if isinstance(choice.get("logprobs"), dict) else None
            if lp and isinstance(lp.get("token_logprobs"), list):
                any_logprobs = True
                token_logprobs.append([x for x in lp["token_logprobs"] if isinstance(x, (int, float))])
            else:
                token_logprobs.append([])

        if not any_logprobs:
            token_logprobs = None
        return texts, token_logprobs

    def _post_chat_completions(
        self,
        model: str,
        prompt_text: str,
        n: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        max_tokens: int,
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], Optional[List[List[float]]]]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": n,
        }
        if top_k > 0:
            payload["top_k"] = top_k
        if min_p and min_p > 0:
            payload["min_p"] = min_p
        if repetition_penalty != 1.0:
            payload["repetition_penalty"] = repetition_penalty
        if generation_kwargs:
            payload.update(generation_kwargs)

        resp = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", []) if isinstance(data, dict) else []

        texts: List[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            msg = choice.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                texts.append(msg["content"].strip())
            else:
                texts.append((choice.get("text") or "").strip())

        # Chat logprobs are not standardized across servers; keep None unless we detect a usable format.
        return texts, None

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[Dict[str, Any]] = None,
        images: Optional[List] = None,
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 256,
        truncate_prompt_tokens: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate completions using the external vLLM server.

        Return format matches TRL's server-mode expectation:
        - `prompt_ids`: list[list[int]] with one entry per prompt (unique prompts).
        - `completion_ids`: list[list[int]] with one entry per completion (len(prompts) * n).
        - `logprobs`: optional list[list[float]] with per-token logprobs for each completion.
        """
        if images is not None:
            raise NotImplementedError("Image inputs are not supported in server mode via this HTTP client yet.")

        model_name = kwargs.get("model") or self.default_model or "glm-4.7"

        prompt_ids_list: List[List[int]] = []
        completion_ids_list: List[List[int]] = []
        logprobs_list: List[List[float]] = []
        any_logprobs = False

        def run_one(prompt_text: str) -> Tuple[List[str], Optional[List[List[float]]]]:
            # Prefer /v1/completions since it more reliably supports `logprobs` on vLLM.
            try:
                return self._post_completions(
                    model=model_name,
                    prompt_text=prompt_text,
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    generation_kwargs=generation_kwargs,
                )
            except Exception:
                return self._post_chat_completions(
                    model=model_name,
                    prompt_text=prompt_text,
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    generation_kwargs=generation_kwargs,
                )

        prompt_texts: List[str] = []
        for prompt in prompts:
            prompt_text, prompt_ids = self._maybe_truncate_prompt(prompt, truncate_prompt_tokens)
            prompt_texts.append(prompt_text)
            prompt_ids_list.append(prompt_ids)

        results: List[Tuple[List[str], Optional[List[List[float]]]]] = [([], None) for _ in prompt_texts]
        if self.max_concurrent_requests > 1 and len(prompt_texts) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as ex:
                fut_to_idx = {ex.submit(run_one, txt): i for i, txt in enumerate(prompt_texts)}
                for fut in concurrent.futures.as_completed(fut_to_idx):
                    idx = fut_to_idx[fut]
                    results[idx] = fut.result()
        else:
            for i, txt in enumerate(prompt_texts):
                results[i] = run_one(txt)

        for texts, token_logprobs in results:
            for i, completion_text in enumerate(texts):
                if not completion_text:
                    completion_ids_list.append(
                        [getattr(self.tokenizer, "eos_token_id", 0) or 0] if self.tokenizer else [0]
                    )
                    logprobs_list.append([])
                    continue

                if self.tokenizer is not None:
                    completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)
                else:
                    completion_ids = [0] * min(len(completion_text.split()), max_tokens)

                completion_ids_list.append(completion_ids)

                if token_logprobs is not None and i < len(token_logprobs):
                    any_logprobs = True
                    logprobs_list.append(token_logprobs[i])
                else:
                    logprobs_list.append([])

        if not any_logprobs:
            logprobs_list = [[] for _ in logprobs_list]

        return {"prompt_ids": prompt_ids_list, "completion_ids": completion_ids_list, "logprobs": logprobs_list}
