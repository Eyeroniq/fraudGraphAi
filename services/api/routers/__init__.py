"""API routers (A7). Each module builds its router for one app, so rate-limit counters
and state never leak between app instances (tests create many).

No `from __future__ import annotations` here: slowapi wraps the endpoints, and FastAPI
resolves string annotations against the wrapper's module, not this one.
"""
