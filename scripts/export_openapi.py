"""Write the API's OpenAPI schema to web/openapi.json (input of `npm run gen:api`).

    uv run python scripts/export_openapi.py

The console's TypeScript types (`web/lib/schema.d.ts`) are generated from this file, so
`services/api/schemas.py` stays the single source of the contract. CI regenerates both and
fails if they differ from what is committed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.api.main import create_app  # noqa: E402


def main() -> int:
    schema = create_app(None).openapi()
    out = ROOT / "web" / "openapi.json"
    text = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    out.write_text(text, encoding="utf-8", newline="\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
