from pathlib import Path


def resolve_capsule_path(local_relative: str) -> Path:
    """Resolve `local_relative` under the Code Ocean capsule, else the repo root."""
    capsule_path = Path("/root/capsule") / local_relative
    try:
        is_capsule = capsule_path.exists()
    except OSError:
        is_capsule = False
    if is_capsule:
        return capsule_path
    return Path(__file__).resolve().parents[3] / local_relative


RESULTS = resolve_capsule_path("results")
RESULTS.mkdir(parents=True, exist_ok=True)
