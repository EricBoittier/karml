import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO, check=True, capture_output=True, text=True
    )
    return [path for path in result.stdout.splitlines() if (REPO / path).exists()]


def test_runtime_logs_are_not_tracked() -> None:
    offenders = [path for path in _tracked_files() if Path(path).name == "nohup.out"]
    assert offenders == []


def test_active_workflow_code_has_no_personal_checkpoint_fallback() -> None:
    needle = "/mm" + "home/boitt" + "ier/home/karml_tutorial/acodcm/ckpts"
    offenders: list[str] = []
    for relative in _tracked_files():
        path = Path(relative)
        if not relative.startswith("workflows/") or path.suffix not in {".py", ".sh"}:
            continue
        if needle in (REPO / relative).read_text(encoding="utf-8", errors="replace"):
            offenders.append(relative)
    assert offenders == []


_PERSONAL = ("/mm" + "home/", "boitti" + "er", "metatomic" + "-runs", "/Users/eric", "ericb" + "oittier", "/home/eric" + "b", "icl" + "oud")
# pyproject.toml carries the maintainer's own author e-mail (package metadata); this file holds the needles.
_SKIP = {"pyproject.toml", "tests/unit/test_repository_hygiene.py"}


def test_tracked_text_files_have_no_personal_paths() -> None:
    """No home paths, user/account names or private e-mail in any tracked text file (site names such as sciCORE are public)."""
    offenders: list[str] = []
    for relative in _tracked_files():
        if relative in _SKIP:
            continue
        data = (REPO / relative).read_bytes()
        if b"\0" in data[:8192]:
            continue
        text = data.decode("utf-8", errors="replace")
        if any(p in text for p in _PERSONAL) or any(p in relative for p in _PERSONAL):
            offenders.append(relative)
    assert offenders == []
