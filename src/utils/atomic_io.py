from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def atomic_json_write(path: Path | str, data: object) -> None:
    """Write JSON to a temp file then atomically replace path.

    Creates parent directories as needed. Avoids partial-read races and
    leaves the destination unchanged if serialisation fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, default=str)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
