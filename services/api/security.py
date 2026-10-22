"""API security (task 7.7): scoped API keys, request size limit, log redaction.

Keys are random tokens; only their sha256 is configured (`FG_API__API_KEYS`, entries
`scope:hex`), so a leaked config or environment dump does not leak a usable key. Scopes
nest: admin > write > read. Without any configured key every protected route answers
401 (fail closed) unless auth is disabled explicitly for local development.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import secrets
from dataclasses import dataclass

from starlette.types import ASGIApp, Message, Receive, Scope, Send

SCOPES = ("read", "write", "admin")
KEY_HEADER = "X-API-Key"


class ApiError(Exception):
    """Rendered as `{"error": {"code", "message"}}` with `status`."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_key() -> str:
    return f"fg_{secrets.token_urlsafe(32)}"


@dataclass(frozen=True)
class Principal:
    scope: str
    key_id: str  # first hash characters: safe to log and to rate-limit on


@dataclass(frozen=True)
class ApiKeys:
    entries: tuple[tuple[str, str], ...]  # (scope, sha256 hex)
    enabled: bool = True

    @classmethod
    def parse(cls, text: str, *, enabled: bool = True) -> ApiKeys:
        entries = []
        for raw in text.split(","):
            item = raw.strip()
            if not item:
                continue
            scope, _, digest = item.partition(":")
            if scope not in SCOPES or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("API keys must be `scope:sha256hex` with scope read|write|admin")
            entries.append((scope, digest))
        return cls(tuple(entries), enabled)

    def authenticate(self, key: str | None, needed: str) -> Principal:
        if not self.enabled:
            return Principal("admin", "anonymous")
        if not key:
            raise ApiError(401, "unauthorized", f"missing {KEY_HEADER} header")
        digest = hash_key(key)
        scopes = [s for s, d in self.entries if hmac.compare_digest(d, digest)]
        if not scopes:
            raise ApiError(401, "unauthorized", "invalid API key")
        best = max(scopes, key=SCOPES.index)
        if SCOPES.index(best) < SCOPES.index(needed):
            raise ApiError(403, "forbidden", f"this key has scope '{best}', needs '{needed}'")
        return Principal(best, digest[:12])


class BodyLimit:
    """Reject request bodies over `max_bytes` with 413, before the app parses them.

    The body is buffered up to the limit (it is small by then) so chunked uploads without
    a Content-Length are capped too.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            body += message.get("body", b"")
            if len(body) > self.max_bytes:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    async def _reject(self, send: Send) -> None:
        payload = json.dumps(
            {
                "error": {
                    "code": "payload_too_large",
                    "message": f"request body over {self.max_bytes} bytes",
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})


SECRET_PATTERN = re.compile(r"(?i)(api_key=|x-api-key[\"']?\s*[:=]\s*[\"']?)[^&\s\"']+")


class RedactFilter(logging.Filter):
    """Mask API keys in log lines (the WebSocket takes its key as `?api_key=`).

    Request bodies are never logged, so amounts and balances stay out of the logs; only
    transaction and account IDs appear.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = SECRET_PATTERN.sub(r"\1***", message)
        if redacted != message:
            record.msg, record.args = redacted, None
        return True


def install_redaction() -> None:
    for name in ("uvicorn.access", "uvicorn.error", "fraudgraph.api"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactFilter) for f in logger.filters):
            logger.addFilter(RedactFilter())
