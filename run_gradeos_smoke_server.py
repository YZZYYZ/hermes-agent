from __future__ import annotations

import os
import shutil
from pathlib import Path

import uvicorn


def _load_backend_env() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / "GradeOS-Platform" / "backend" / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _sync_gradeos_skills(repo_root: Path, hermes_home: Path) -> None:
    source_dir = repo_root / "GradeOS-Platform" / "docs" / "hermes-skills"
    target_dir = hermes_home / "skills"
    if not source_dir.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.iterdir():
        if not source.is_dir():
            continue
        shutil.copytree(source, target_dir / source.name, dirs_exist_ok=True)
    snapshot = hermes_home / ".skills_prompt_snapshot.json"
    snapshot.unlink(missing_ok=True)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    values = _load_backend_env()
    hermes_home = repo_root / "Hermes" / ".hermes-test"
    _sync_gradeos_skills(repo_root, hermes_home)
    os.environ.setdefault("OPENROUTER_API_KEY", values.get("LLM_API_KEY", ""))
    os.environ.setdefault("HERMES_HOME", str(hermes_home))
    os.environ.setdefault("HERMES_INFERENCE_PROVIDER", "openrouter")
    os.environ.setdefault("HERMES_INFERENCE_MODEL", "qwen/qwen3.7-plus")
    teacher_service_token = values.get("HERMES_AGENT_SERVICE_TOKEN", "").strip()
    student_service_token = values.get("HERMES_STUDENT_AGENT_SERVICE_TOKEN", "").strip()
    if teacher_service_token and student_service_token and teacher_service_token != student_service_token:
        raise RuntimeError(
            "Local Hermes uses one internal key; HERMES_AGENT_SERVICE_TOKEN and "
            "HERMES_STUDENT_AGENT_SERVICE_TOKEN must match."
        )
    # CODEX CHANGE: the local smoke service accepts one bearer key for both
    # assistant routes, so inherit GradeOS's configured service token instead
    # of falling back to an unrelated default token.
    internal_key = (
        values.get("HERMES_INTERNAL_KEY", "").strip()
        or teacher_service_token
        or student_service_token
        or "gradeos-local-dev-token"
    )
    os.environ["HERMES_INTERNAL_KEY"] = internal_key
    os.environ.setdefault(
        "GRADEOS_INTERNAL_API_BASE_URL",
        values.get("GRADEOS_INTERNAL_API_BASE_URL", "http://127.0.0.1:8001"),
    )
    os.environ.setdefault(
        "GRADEOS_INTERNAL_SERVICE_TOKEN",
        values.get(
            "GRADEOS_INTERNAL_SERVICE_TOKEN",
            values.get(
                "HERMES_AGENT_SERVICE_TOKEN",
                values.get("HERMES_INTERNAL_KEY", "gradeos-local-dev-token"),
            ),
        ),
    )
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # CODEX CHANGE: importing the smoke app imports run_agent, which loads the
    # Hermes project .env and can contain a stale local integration token.
    # Restore the backend-derived token after that import without changing the
    # user's chosen internal API address (for example, port 8001).
    # TODO: remove this ordering bridge when local GradeOS secrets use one source.
    import gradeos_smoke_server

    os.environ["HERMES_INTERNAL_KEY"] = internal_key
    gradeos_smoke_server.configure_gradeos_internal_token(internal_key)
    os.environ["GRADEOS_INTERNAL_SERVICE_TOKEN"] = values.get(
        "GRADEOS_INTERNAL_SERVICE_TOKEN",
        teacher_service_token or internal_key,
    )
    uvicorn.run(
        gradeos_smoke_server.app,
        host="127.0.0.1",
        port=8765,
        log_level="info",
    )
