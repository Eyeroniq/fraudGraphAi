"""Redis-backed feature state for the API (task 7.1, A5).

Same contract as `InMemoryStore`, so the engine and the parity tests do not care which
one they get. Layout, under a configurable prefix:

- `{prefix}h:{key}` - one hash per state key (`o:<acct>`, `d:<acct>`, `f:<acct>`). Its
  fields are the step buckets: `<step>:n`, `<step>:t` (sum), `<step>:q` (sum of squares)
  and `<step>:m` (counterparties, comma-joined). Buckets older than the horizon are
  dropped when a new bucket is opened, and the whole hash expires after `ttl_seconds`
  without writes (idle accounts are evicted, as in the in-memory store).
- `{prefix}latest` - the newest step applied (out-of-order guard).
- `{prefix}seen:{marker}` - `SET NX` marker per applied event (txn_id), so a retried
  request never double-counts.

Every event is applied by one Lua script, so its writes are atomic and the guard, the
marker and the bucket updates cannot interleave with another API worker. Single-node
Redis: the script touches only the keys it is given.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import redis

from fraudgraph.features.state import Bucket, OutOfOrderError, Write

if TYPE_CHECKING:
    from redis.commands.core import Script

DAY_SECONDS = 24 * 3600

# KEYS: latest, seen marker, then one hash per write.
# ARGV: step, horizon, ttl_seconds, seen_ttl_seconds, has_marker, then per write:
#       amount, amount^2, member ('' for none).
# Returns 1 if applied, 0 if the marker existed; errors with "OUT_OF_ORDER <latest>".
APPLY_LUA = """
local step = tonumber(ARGV[1])
local s = ARGV[1]
local horizon = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local latest = tonumber(redis.call('GET', KEYS[1]))
if latest and step < latest then
  return redis.error_reply('OUT_OF_ORDER ' .. latest)
end
if ARGV[5] == '1' then
  if not redis.call('SET', KEYS[2], '1', 'NX', 'EX', tonumber(ARGV[4])) then
    return 0
  end
end
if (not latest) or step > latest then
  redis.call('SET', KEYS[1], s)
end
local oldest = step - horizon
for i = 3, #KEYS do
  local key = KEYS[i]
  local base = 6 + (i - 3) * 3
  local n = redis.call('HINCRBY', key, s .. ':n', 1)
  redis.call('HINCRBYFLOAT', key, s .. ':t', ARGV[base])
  redis.call('HINCRBYFLOAT', key, s .. ':q', ARGV[base + 1])
  local member = ARGV[base + 2]
  if member ~= '' then
    local members = redis.call('HGET', key, s .. ':m')
    if members then
      redis.call('HSET', key, s .. ':m', members .. ',' .. member)
    else
      redis.call('HSET', key, s .. ':m', member)
    end
  end
  if n == 1 then
    for _, field in ipairs(redis.call('HKEYS', key)) do
      local at = tonumber(string.match(field, '^(%d+):'))
      if at and at < oldest then
        redis.call('HDEL', key, field)
      end
    end
  end
  redis.call('EXPIRE', key, ttl)
end
return 1
"""


class RedisStore:
    """Feature state in Redis. Reads filter by step exactly like `InMemoryStore`."""

    def __init__(
        self,
        client: redis.Redis,
        horizon_steps: int,
        *,
        prefix: str = "fg:",
        ttl_seconds: int = 7 * DAY_SECONDS,
        seen_ttl_seconds: int = 2 * DAY_SECONDS,
    ) -> None:
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        self.client = client
        self.horizon_steps = horizon_steps
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self.seen_ttl_seconds = seen_ttl_seconds
        self._apply: Script = client.register_script(APPLY_LUA)

    @classmethod
    def from_url(cls, url: str, horizon_steps: int, **kwargs: Any) -> RedisStore:
        return cls(redis.Redis.from_url(url), horizon_steps, **kwargs)

    @property
    def latest_step(self) -> int | None:
        value = self.client.get(f"{self.prefix}latest")
        return None if value is None else int(cast(bytes, value))

    def ping(self) -> bool:
        return bool(self.client.ping())

    def history(self, key: str, before_step: int) -> list[Bucket]:
        raw = cast(dict[bytes, bytes], self.client.hgetall(self._hash(key)))
        oldest = before_step - self.horizon_steps
        fields: dict[int, dict[str, str]] = {}
        for name, value in raw.items():
            step_text, _, kind = name.decode().partition(":")
            step = int(step_text)
            if oldest <= step < before_step:
                fields.setdefault(step, {})[kind] = value.decode()
        buckets = []
        for step in sorted(fields):
            f = fields[step]
            bucket = Bucket(step, 0.0, None)
            bucket.n = int(f["n"])
            bucket.total = float(f["t"])
            bucket.total_sq = float(f["q"])
            bucket.members = f["m"].split(",") if "m" in f else None
            buckets.append(bucket)
        return buckets

    def add(self, key: str, step: int, amount: float, member: str | None = None) -> None:
        self.apply(None, [(key, step, amount, member)])

    def apply(self, marker: str | None, writes: Sequence[Write]) -> bool:
        if not writes:
            return True
        step = writes[0][1]
        if any(w[1] != step for w in writes):
            raise ValueError("all writes of one unit must share a step")
        keys = [
            f"{self.prefix}latest",
            f"{self.prefix}seen:{marker or ''}",
            *(self._hash(w[0]) for w in writes),
        ]
        args: list[str | int] = [
            step,
            self.horizon_steps,
            self.ttl_seconds,
            self.seen_ttl_seconds,
            "0" if marker is None else "1",
        ]
        for _, _, amount, member in writes:
            args += [repr(float(amount)), repr(float(amount) * float(amount)), member or ""]
        try:
            applied = self._apply(keys=keys, args=args)
        except redis.ResponseError as exc:
            if "OUT_OF_ORDER" in str(exc):
                latest = str(exc).rsplit(" ", 1)[-1]
                raise OutOfOrderError(f"step {step} arrived after step {latest}") from exc
            raise
        return bool(applied)

    def _hash(self, key: str) -> str:
        return f"{self.prefix}h:{key}"
