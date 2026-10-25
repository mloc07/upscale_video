"""Gemini translation backend (google-genai SDK).

Behaviour (see plan section 3/Phase 3):
* batches with read-only context from neighbouring segments;
* structured JSON output, output count validated against input count;
* exponential backoff for rate-limit / network errors;
* legitimate handling of blocked/refused responses: retry with smaller batches,
  optional fallback backend, finally keep the original text flagged as failed;
* per-batch on-disk cache so interrupted runs resume without re-translating;
* the API key is read from an environment variable and is redacted from any message.

The translation prompt is an ordinary, faithful-translation prompt. Nothing here tries to
bypass the provider's safety filters; only the API's own documented ``safety_settings`` are exposed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..config import TranslateConfig
from ..segments import Segment, TranslatedSegment
from .base import TranslatorBackend

LANGUAGE_NAMES = {
    "vi": "Vietnamese", "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "ru": "Russian", "th": "Thai", "id": "Indonesian",
}

_RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {"id": {"type": "INTEGER"}, "text": {"type": "STRING"}},
        "required": ["id", "text"],
    },
}

_BLOCK_FINISH = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY", "OTHER"}
_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


class TranslationError(RuntimeError):
    """Non-recoverable translation failure (e.g. invalid API key)."""


class BlockedError(RuntimeError):
    """The API blocked or refused the response (safety/policy)."""


class CountMismatchError(RuntimeError):
    """Model returned a different number of items than requested."""


class RetryableError(RuntimeError):
    """Transient error (rate limit / network)."""


def _redact(text: str, secret: Optional[str]) -> str:
    if secret and secret in text:
        text = text.replace(secret, "***")
    return text


def _lang_name(code: Optional[str]) -> str:
    if not code:
        return "the source language (auto-detected)"
    return LANGUAGE_NAMES.get(code.lower().split("-")[0], code)


class GeminiTranslator:
    def __init__(
        self,
        cfg: TranslateConfig,
        client=None,
        sleep: Callable[[float], None] = time.sleep,
        fallback: Optional[TranslatorBackend] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.cfg = cfg
        self._client = client
        self._sleep = sleep
        self.fallback = fallback
        self.should_cancel = should_cancel
        self._key: Optional[str] = None
        self._model = cfg.model
        self.stats = {
            "requests": 0, "retries": 0, "cache_hits": 0,
            "blocked_batches": 0, "count_mismatches": 0, "failed_segments": 0, "fallback_segments": 0,
        }
        self.cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ client
    def _get_client(self):
        if self._client is None:
            key = os.environ.get(self.cfg.api_key_env, "").strip()
            if not key:
                raise TranslationError(
                    f"Environment variable {self.cfg.api_key_env} is not set (API key is never read from files in the repo)."
                )
            self._key = key
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover
                raise TranslationError("google-genai is not installed: pip install 'ecoup[subtitle]'") from exc
            self._client = genai.Client(api_key=key)
        return self._client

    def _safety_settings(self):
        if self.cfg.safety_threshold != "block_none":
            return None
        from google.genai import types  # type: ignore
        cats = [
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        ]
        return [types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE) for c in cats]

    # ------------------------------------------------------------------ prompt
    def _system_prompt(self, source_language: Optional[str]) -> str:
        target = _lang_name(self.cfg.target_language)
        parts = [
            f"You are a professional subtitle translator. Translate subtitle lines from {_lang_name(source_language)} into natural, "
            f"fluent {target}.",
            "Be faithful: keep the meaning, tone, register, emotion and any coarse language of the speaker; do not summarise, "
            "omit, censor or add explanations.",
            "Keep pronouns / forms of address natural and consistent across the whole dialogue "
            "(for Vietnamese choose appropriate anh/chị/em/tôi/bạn/mày/tao... from context and keep them stable).",
            "Keep each line short enough for subtitles. Keep proper names consistent. Preserve numbers.",
            "Return ONLY a JSON array of objects {\"id\": <same id>, \"text\": <translation>} with exactly one object per input id, "
            "in the same order. Never merge, split or drop lines.",
        ]
        if self.cfg.context_note.strip():
            parts.append(f"Context from the user: {self.cfg.context_note.strip()}")
        if self.cfg.glossary:
            gl = "; ".join(f"{k} => {v}" for k, v in self.cfg.glossary.items())
            parts.append(f"Glossary (use these translations exactly): {gl}")
        return "\n".join(parts)

    @staticmethod
    def _user_prompt(batch: Sequence[Segment], before: Sequence[Segment], after: Sequence[Segment]) -> str:
        def lines(segs):
            return "\n".join(f"- {s.text}" for s in segs) or "(none)"

        payload = [{"id": i, "text": s.text} for i, s in enumerate(batch)]
        return (
            "Previous lines (context only, do NOT translate):\n" + lines(before) + "\n\n"
            "Following lines (context only, do NOT translate):\n" + lines(after) + "\n\n"
            "Translate these lines:\n" + json.dumps(payload, ensure_ascii=False)
        )

    # ------------------------------------------------------------------- cache
    def _cache_key(self, batch, before, after, source_language) -> str:
        blob = json.dumps(
            {
                "m": self.cfg.model, "t": self.cfg.target_language, "temp": self.cfg.temperature,
                "note": self.cfg.context_note, "gl": self.cfg.glossary, "src": source_language,
                "b": [s.text for s in batch], "pre": [s.text for s in before], "post": [s.text for s in after],
            },
            ensure_ascii=False, sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[list[str]]:
        if not self.cache_dir:
            return None
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list) and all(isinstance(x, str) for x in data):
                    return data
            except Exception:
                return None
        return None

    def _cache_put(self, key: str, texts: list[str]) -> None:
        if self.cache_dir:
            (self.cache_dir / f"{key}.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")

    # ----------------------------------------------------------------- request
    def _call_api(self, batch, before, after, source_language) -> list[str]:
        client = self._get_client()
        from google.genai import types  # type: ignore

        config = types.GenerateContentConfig(
            system_instruction=self._system_prompt(source_language),
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
            temperature=self.cfg.temperature,
            safety_settings=self._safety_settings(),
            http_options=types.HttpOptions(timeout=int(self.cfg.timeout_s * 1000)),
        )
        self.stats["requests"] += 1
        try:
            resp = client.models.generate_content(
                model=self._model, contents=self._user_prompt(batch, before, after), config=config
            )
        except Exception as exc:  # noqa: BLE001 - classify below
            msg = _redact(str(exc), self._key)
            code = getattr(exc, "code", None)
            if code in _RETRYABLE_CODES or isinstance(exc, (TimeoutError, ConnectionError)):
                raise RetryableError(msg[:300]) from None
            if code in (401, 403):
                raise TranslationError(f"API authentication/permission error ({code}): {msg[:200]}") from None
            if code == 400 and "block" in msg.lower():
                raise BlockedError(msg[:200]) from None
            raise TranslationError(f"API error ({code}): {msg[:300]}") from None
        return self._parse_response(resp, len(batch))

    @staticmethod
    def _parse_response(resp, expected: int) -> list[str]:
        feedback = getattr(resp, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise BlockedError(f"prompt blocked: {feedback.block_reason}")
        cands = getattr(resp, "candidates", None) or []
        if cands:
            fr = getattr(cands[0], "finish_reason", None)
            fr_name = getattr(fr, "name", str(fr or "")).upper().split(".")[-1]
            if fr_name in _BLOCK_FINISH:
                raise BlockedError(f"finish_reason={fr_name}")
        text = getattr(resp, "text", None)
        if not text:
            raise BlockedError("empty response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CountMismatchError(f"invalid JSON: {exc}") from None
        if not isinstance(data, list):
            raise CountMismatchError("response is not a JSON array")
        by_id = {}
        for item in data:
            if isinstance(item, dict) and "id" in item and "text" in item:
                by_id[int(item["id"])] = str(item["text"]).strip()
        if len(data) != expected or set(by_id) != set(range(expected)):
            raise CountMismatchError(f"expected {expected} items, got {len(data)} (ids ok: {len(by_id)})")
        return [by_id[i] for i in range(expected)]

    def _request_with_retry(self, batch, before, after, source_language) -> list[str]:
        """Backoff on transient errors; when a model stays unavailable, switch to the next fallback model.

        If every model is unavailable the run stops with a clear error (progress stays in the cache,
        so re-running resumes) instead of silently leaving lines untranslated.
        """
        models = [self.cfg.model] + [m for m in self.cfg.fallback_models if m != self.cfg.model]
        start = models.index(self._model) if self._model in models else 0
        last_exc: Optional[Exception] = None
        for model in models[start:]:
            self._model = model
            attempt = 0
            while True:
                if self.should_cancel and self.should_cancel():
                    raise TranslationError("cancelled")
                try:
                    return self._call_api(batch, before, after, source_language)
                except RetryableError as exc:
                    last_exc = exc
                    if attempt >= self.cfg.max_retries:
                        self.stats["model_switches"] = self.stats.get("model_switches", 0) + 1
                        break
                    self.stats["retries"] += 1
                    self._sleep(min(60.0, 2.0 ** attempt + 0.5))
                    attempt += 1
        raise TranslationError(
            f"All translation models unavailable ({', '.join(models[start:])}); last error: {last_exc}. "
            "Re-run later: finished batches are cached."
        )

    # -------------------------------------------------------------- public API
    def translate(self, batch, context_before=(), context_after=(), source_language=None) -> list[TranslatedSegment]:
        batch = list(batch)
        if not batch:
            return []
        key = self._cache_key(batch, context_before, context_after, source_language)
        cached = self._cache_get(key)
        if cached is not None and len(cached) == len(batch):
            self.stats["cache_hits"] += 1
            return self._build(batch, cached)
        try:
            texts = self._request_with_retry(batch, context_before, context_after, source_language)
        except (BlockedError, CountMismatchError, RetryableError) as exc:
            if isinstance(exc, BlockedError):
                self.stats["blocked_batches"] += 1
            elif isinstance(exc, CountMismatchError):
                self.stats["count_mismatches"] += 1
            if len(batch) > 1:  # retry with smaller batches
                mid = len(batch) // 2
                left = self.translate(batch[:mid], context_before, list(batch[mid:mid + self.cfg.context_segments]) or context_after,
                                      source_language)
                right = self.translate(batch[mid:], list(batch[max(0, mid - self.cfg.context_segments):mid]) or context_before,
                                       context_after, source_language)
                return left + right
            return [self._single_failure(batch[0], str(exc), source_language, context_before, context_after)]
        self._cache_put(key, texts)
        return self._build(batch, texts)

    def _single_failure(self, seg, reason, source_language, before, after) -> TranslatedSegment:
        if self.fallback is not None:
            try:
                out = self.fallback.translate([seg], before, after, source_language)
                if len(out) == 1 and not out[0].failed:
                    self.stats["fallback_segments"] += 1
                    return out[0]
            except Exception:  # noqa: BLE001
                pass
        self.stats["failed_segments"] += 1
        return TranslatedSegment(
            start=seg.start, end=seg.end, source=seg.text, text=seg.text,
            language=seg.language, failed=True, note=f"translation failed/blocked: {reason[:160]}",
        )

    @staticmethod
    def _build(batch, texts) -> list[TranslatedSegment]:
        out = []
        for s, t in zip(batch, texts):
            failed = not t.strip()
            out.append(TranslatedSegment(
                start=s.start, end=s.end, source=s.text, text=(s.text if failed else t),
                language=s.language, failed=failed, note="empty translation" if failed else "",
            ))
        return out
