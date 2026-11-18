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
    needle = "/mmhome/boittier/home/karml_tutorial/acodcm/ckpts"
    offenders: list[str] = []
    for relative in _tracked_files():
        path = Path(relative)
        if not relative.startswith("workflows/") or path.suffix not in {".py", ".sh"}:
            continue
        if needle in (REPO / relative).read_text(encoding="utf-8", errors="replace"):
            offenders.append(relative)
    assert offenders == []


_PERSONAL = ("/mm" + "home/", "boitti" + "0000", "/Users/eric", "/cluster/home/boit", "/home/eric" + "b/")


def test_docs_and_configs_have_no_personal_paths() -> None:
    """Docs, JSON, YAML and editor configs must not carry personal or cluster-account paths.

    (Cluster submission scripts under scripts/ and workflows/ still hold site-specific defaults; they are tracked separately.)
    """
    offenders: list[str] = []
    for relative in _tracked_files():
        path = Path(relative)
        if relative.startswith("benchmarks/results/") or relative.endswith(".bak"):
            continue
        if not (relative.startswith(("docs/", ".cursor/")) or path.suffix in {".md", ".json", ".yaml", ".yml"}):
            continue
        if relative.startswith(("scripts/", "workflows/")) and path.suffix not in {".md"}:
            continue
        text = (REPO / relative).read_text(encoding="utf-8", errors="replace")
        if any(p in text for p in _PERSONAL):
            offenders.append(relative)
    assert offenders == []
