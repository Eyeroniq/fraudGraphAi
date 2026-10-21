"""Install the repository's pre-commit hook into `.git/hooks/pre-commit`.

`pre-commit install` refuses to run when `core.hooksPath` is set globally, so this
writes a small shim instead. A global hooks setup that chains to `.git/hooks`
will still run it. An existing hook not written by this script is left alone.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MARKER = "# fraudgraph: installed by scripts/install_git_hooks.py"
SHIM = f"""#!/bin/sh
{MARKER}
exec uv run --frozen pre-commit run --hook-stage pre-commit
"""


def main() -> int:
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        print("install_git_hooks: not a git checkout, skipping.")
        return 0

    hook = Path(common).resolve() / "hooks" / "pre-commit"
    if hook.exists() and MARKER not in hook.read_text(encoding="utf-8", errors="replace"):
        print(f"install_git_hooks: {hook} exists and was not written by us; leaving it.")
        return 0

    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(SHIM, encoding="utf-8", newline="\n")
    hook.chmod(0o755)
    print(f"install_git_hooks: wrote {hook}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
