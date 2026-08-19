from __future__ import annotations

import asyncio
import json
import os
import yaml
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from run_agent import AIAgent
from tools.gradeos_tools import set_gradeos_session_scope


app = FastAPI(title="GradeOS Hermes Smoke Server")

DEFAULT_DEV_TOKEN = "gradeos-local-dev-token"
_gradeos_internal_token: str | None = None


class TeacherInfo(BaseModel):
    teacher_id: str = "local_teacher"
    org_id: Optional[str] = None


class StudentInfo(BaseModel):
    student_id: str = "local_student"


class TeacherAgentChatRequest(BaseModel):
    request_id: str = "local-smoke"
    session_id: str = "gradeos-smoke-session"
    message: str
    teacher: TeacherInfo = Field(default_factory=TeacherInfo)
    scope: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    tool_policy: Dict[str, Any] = Field(
        default_factory=lambda: {
            "allow_writes": False,
            "allow_external_web": False,
            "disable_memory": True,
            "disable_session_search": True,
        }
    )
    gradeos_tools: Dict[str, Any] = Field(default_factory=dict)


class StudentAgentChatRequest(BaseModel):
    request_id: str = "local-student-smoke"
    request_fingerprint: str = ""
    session_id: str = "gradeos-student-smoke-session"
    message: str
    student: StudentInfo = Field(default_factory=StudentInfo)
    scope: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    tool_policy: Dict[str, Any] = Field(
        default_factory=lambda: {
            "allow_writes": False,
            "allow_external_web": False,
            "system_data": False,
            "native_memory": False,
            "artifact_generation": False,
        }
    )
    gradeos_scope_token: Optional[str] = None


class TeacherAgentChatResponse(BaseModel):
    request_id: str
    session_id: str
    session_key: str
    content: str
    model: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    action_proposals: List[Dict[str, Any]] = Field(default_factory=list)
    usage: Dict[str, Any] = Field(default_factory=dict)


class StudentAgentChatResponse(BaseModel):
    request_id: str
    request_fingerprint: str = ""
    session_id: str
    session_key: str
    content: str
    model: Optional[str] = None
    response_type: str = "explanation"
    next_question: Optional[str] = None
    question_options: List[str] = Field(default_factory=list)
    focus_mode: bool = True
    concept_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    mastery: Dict[str, Any] = Field(default_factory=dict)
    parse_status: str = "hermes_ok"
    parse_error_code: Optional[str] = None
    safety_level: Optional[str] = None
    usage: Dict[str, Any] = Field(default_factory=dict)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "openrouter_key": bool(os.getenv("OPENROUTER_API_KEY")),
        "provider": os.getenv("HERMES_INFERENCE_PROVIDER") or "openrouter",
        "auth_configured": bool(os.getenv("HERMES_INTERNAL_KEY") or os.getenv("API_SERVER_KEY")),
        "gradeos_internal_api": os.getenv("GRADEOS_INTERNAL_API_BASE_URL", "http://127.0.0.1:8001"),
    }


def configure_gradeos_internal_token(token: str) -> None:
    """Pin the local GradeOS service token after Hermes has loaded its .env."""
    global _gradeos_internal_token
    _gradeos_internal_token = token.strip() or None


def _expected_token() -> str:
    # CODEX CHANGE: GradeOS authentication must not change while an active
    # student conversation lazily initializes other Hermes configuration.
    # TODO: replace this local bridge when both projects consume shared config.
    return (
        _gradeos_internal_token
        or os.getenv("HERMES_INTERNAL_KEY")
        or os.getenv("API_SERVER_KEY")
        or DEFAULT_DEV_TOKEN
    )


def _verify_internal_auth(authorization: Optional[str]) -> None:
    expected = _expected_token()
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Hermes internal bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=403, detail="Invalid Hermes internal bearer token")


def _validate_header_scope(value: str, header_name: str) -> str:
    if len(value) > 256 or any(ch in value for ch in ("\r", "\n", "\x00")):
        raise HTTPException(status_code=400, detail=f"Invalid {header_name}")
    return value


def _fallback_session_key(request: TeacherAgentChatRequest) -> str:
    tenant_id = (
        request.scope.get("tenant_id")
        or request.scope.get("org_id")
        or request.teacher.org_id
        or "local"
    )
    user_id = request.scope.get("user_id") or request.teacher.teacher_id
    return f"gradeos:tenant:{tenant_id}:teacher:{user_id}:conversation:{request.session_id}"


def _fallback_student_session_key(request: StudentAgentChatRequest) -> str:
    tenant_id = request.scope.get("tenant_id") or "local"
    user_id = request.student.student_id
    return f"gradeos:tenant:{tenant_id}:student:{user_id}:conversation:{request.session_id}"


def _build_agent(session_id: str, session_key: str) -> AIAgent:
    return AIAgent(
        provider=os.getenv("HERMES_INFERENCE_PROVIDER") or "openrouter",
        model=os.getenv("HERMES_INFERENCE_MODEL") or "qwen/qwen3.7-plus",
        platform="gradeos_teacher",
        session_id=session_id,
        user_id=session_key,
        gateway_session_key=session_key,
        enabled_toolsets=["skills", "gradeos_tools"],
        disabled_toolsets=[
            "memory",
            "session_search",
            "terminal",
            "file",
            "browser",
            "computer",
            "computer_use",
            "code_execution",
            "delegation",
            "cronjob",
            "web",
            "image_gen",
            "tts",
        ],
        skip_context_files=True,
        quiet_mode=True,
        max_iterations=8,
    )


def _build_student_agent(session_id: str, session_key: str) -> AIAgent:
    return AIAgent(
        provider=os.getenv("HERMES_INFERENCE_PROVIDER") or "openrouter",
        model=os.getenv("HERMES_INFERENCE_MODEL") or "qwen/qwen3.7-plus",
        platform="gradeos_student",
        session_id=session_id,
        user_id=session_key,
        gateway_session_key=session_key,
        enabled_toolsets=["skills"],
        disabled_toolsets=[
            "memory",
            "session_search",
            "terminal",
            "file",
            "browser",
            "computer",
            "computer_use",
            "code_execution",
            "delegation",
            "cronjob",
            "web",
            "image_gen",
            "tts",
            "gradeos_tools",
        ],
        skip_context_files=True,
        quiet_mode=True,
        max_iterations=6,
    )


def _compact_attachment(attachment: Dict[str, Any]) -> Dict[str, Any]:
    text = attachment.get("text")
    compact = {
        "attachment_id": attachment.get("attachment_id"),
        "kind": attachment.get("kind"),
        "name": attachment.get("name"),
        "mime_type": attachment.get("mime_type"),
        "size_bytes": attachment.get("size_bytes"),
        "url": attachment.get("url"),
        "has_data_url": bool(attachment.get("data_url")),
        "metadata": (
            attachment.get("metadata") if isinstance(attachment.get("metadata"), dict) else {}
        ),
    }
    if isinstance(text, str) and text:
        compact["text_preview"] = text[:2400]
    return compact


def _image_refs_from_attachments(attachments: List[Dict[str, Any]]) -> List[str]:
    """Return only the image data/URL refs already authorized for this request."""
    refs: List[str] = []
    for item in attachments:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        mime_type = str(item.get("mime_type") or "").strip().lower()
        data_url = str(item.get("data_url") or "").strip()
        url = str(item.get("url") or "").strip()
        if kind != "image" and not mime_type.startswith("image/"):
            continue
        if data_url.startswith("data:image/"):
            refs.append(data_url)
        elif url.startswith(("http://", "https://")):
            refs.append(url)
    return refs


def _multimodal_user_message(text: str, attachments: List[Dict[str, Any]]) -> Any:
    """Return a native-vision user message when image refs are present."""
    image_refs = _image_refs_from_attachments(attachments)
    if not image_refs:
        return text
    hint_lines = [f"[Image {index} attached]" for index in range(1, len(image_refs) + 1)]
    parts: List[Dict[str, Any]] = [
        {"type": "text", "text": f"{text}\n\n" + "\n".join(hint_lines)}
    ]
    parts.extend(
        {"type": "image_url", "image_url": {"url": ref}} for ref in image_refs
    )
    return parts


def _configure_native_vision() -> None:
    """Mark the GradeOS Hermes model as vision-capable in its own config."""
    if not os.environ.get("HERMES_HOME"):
        return
    if os.getenv("GRADEOS_HERMES_NATIVE_VISION", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return
    try:
        from hermes_constants import get_hermes_home

        config_path = get_hermes_home() / "config.yaml"
        data: Dict[str, Any] = {}
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                data = loaded
        model_cfg = data.get("model")
        if not isinstance(model_cfg, dict):
            model_cfg = {}
            data["model"] = model_cfg
        if "supports_vision" not in model_cfg:
            model_cfg["supports_vision"] = True
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
    except Exception:
        pass


_configure_native_vision()


def _json_from_text(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _list_field(data: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_list_field(data: Dict[str, Any], key: str) -> List[str]:
    value = data.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _usage_field(parsed: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    usage = parsed.get("usage")
    return usage if isinstance(usage, dict) else fallback


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _agent_usage(agent: AIAgent, result: Any, model_name: str) -> Dict[str, Any]:
    data = result if isinstance(result, dict) else {}
    usage = {
        "input_tokens": _safe_int(
            data.get("input_tokens", getattr(agent, "session_input_tokens", 0))
        ),
        "output_tokens": _safe_int(
            data.get("output_tokens", getattr(agent, "session_output_tokens", 0))
        ),
        "prompt_tokens": _safe_int(
            data.get("prompt_tokens", getattr(agent, "session_prompt_tokens", 0))
        ),
        "completion_tokens": _safe_int(
            data.get("completion_tokens", getattr(agent, "session_completion_tokens", 0))
        ),
        "total_tokens": _safe_int(
            data.get("total_tokens", getattr(agent, "session_total_tokens", 0))
        ),
        "cache_read_input_tokens": _safe_int(
            data.get("cache_read_tokens", getattr(agent, "session_cache_read_tokens", 0))
        ),
        "cache_creation_input_tokens": _safe_int(
            data.get("cache_write_tokens", getattr(agent, "session_cache_write_tokens", 0))
        ),
        "reasoning_tokens": _safe_int(
            data.get("reasoning_tokens", getattr(agent, "session_reasoning_tokens", 0))
        ),
        "api_calls": _safe_int(data.get("api_calls", getattr(agent, "session_api_calls", 0))),
        "estimated_cost_usd": _safe_float(
            data.get("estimated_cost_usd", getattr(agent, "session_estimated_cost_usd", 0.0))
        ),
        "cost_status": data.get("cost_status", getattr(agent, "session_cost_status", None)),
        "cost_source": data.get("cost_source", getattr(agent, "session_cost_source", None)),
        "provider": data.get("provider") or getattr(agent, "provider", None),
        "model": data.get("model") or getattr(agent, "model", None) or model_name,
    }
    if not usage["total_tokens"]:
        if usage["prompt_tokens"] or usage["completion_tokens"]:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        else:
            usage["total_tokens"] = (
                usage["input_tokens"]
                + usage["output_tokens"]
                + usage["cache_read_input_tokens"]
                + usage["cache_creation_input_tokens"]
            )
    return {key: value for key, value in usage.items() if value not in (None, "", 0, 0.0)}


@app.post("/v1/gradeos/teacher-agent/chat", response_model=TeacherAgentChatResponse)
async def teacher_agent_chat(
    request: TeacherAgentChatRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_hermes_session_key: Optional[str] = Header(default=None, alias="X-Hermes-Session-Key"),
    x_hermes_session_id: Optional[str] = Header(default=None, alias="X-Hermes-Session-Id"),
) -> TeacherAgentChatResponse:
    _verify_internal_auth(authorization)
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not configured")

    session_id = _validate_header_scope(
        x_hermes_session_id or request.session_id,
        "X-Hermes-Session-Id",
    )
    session_key = _validate_header_scope(
        x_hermes_session_key or _fallback_session_key(request),
        "X-Hermes-Session-Key",
    )

    attachment_context = [_compact_attachment(item) for item in request.attachments]
    model_name = os.getenv("HERMES_INFERENCE_MODEL") or "qwen/qwen3.7-plus"
    response_contract = {
        "content": "concise teacher-facing markdown",
        "citations": [],
        "artifacts": [],
        "action_proposals": [],
        "tool_calls": [],
        "model": model_name,
        "usage": {},
    }
    enriched_message = (
        "Use $gradeos-teacher-assistant for this GradeOS request. "
        "If the skill is available, load it with skill_view before answering.\n"
        "Return exactly one JSON object with these frontend-facing fields: "
        "content, citations, artifacts, action_proposals, tool_calls, model, usage. "
        "Do not use answer as a separate output field.\n\n"
        "<gradeos-context>\n"
        f"teacher_id: {request.teacher.teacher_id}\n"
        f"session_id: {session_id}\n"
        f"session_key: {session_key}\n"
        f"scope: {request.scope}\n"
        f"context: {request.context}\n"
        f"history: {request.history[-8:]}\n"
        f"attachments: {attachment_context}\n"
        f"tool_policy: {request.tool_policy}\n"
        f"gradeos_tools: {request.gradeos_tools}\n"
        f"response_contract: {response_contract}\n"
        "security: GradeOS backend is the authority for user identity, tenant scope, "
        "permissions, retrieval, and audit. Do not infer or request data outside the "
        "provided scope. Do not use MEMORY.md, USER.md, or bare session_search as "
        "GradeOS data sources.\n"
        "</gradeos-context>\n\n"
        f"Teacher question:\n{request.message}"
    )
    user_message = _multimodal_user_message(enriched_message, request.attachments)
    set_gradeos_session_scope(
        session_id,
        teacher_id=request.teacher.teacher_id,
        scope=request.scope,
    )
    agent = _build_agent(session_id, session_key)
    result = await asyncio.to_thread(agent.run_conversation, user_message)
    content = ""
    usage: Dict[str, Any] = {}
    if isinstance(result, dict):
        raw_content = result.get("final_response") or result.get("response") or ""
        content = (
            raw_content
            if isinstance(raw_content, str)
            else json.dumps(raw_content, ensure_ascii=False)
        )
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    else:
        content = str(result)
    if not usage:
        usage = _agent_usage(agent, result, model_name)
    parsed = _json_from_text(content) or {}

    return TeacherAgentChatResponse(
        request_id=request.request_id,
        session_id=session_id,
        session_key=session_key,
        content=str(parsed.get("content") or content),
        model=str(parsed.get("model") or model_name),
        citations=_list_field(parsed, "citations"),
        tool_calls=_list_field(parsed, "tool_calls"),
        artifacts=_list_field(parsed, "artifacts"),
        action_proposals=_list_field(parsed, "action_proposals"),
        usage=_usage_field(parsed, usage),
    )


@app.post("/v1/gradeos/student-agent/chat", response_model=StudentAgentChatResponse)
async def student_agent_chat(
    request: StudentAgentChatRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_hermes_session_key: Optional[str] = Header(default=None, alias="X-Hermes-Session-Key"),
    x_hermes_session_id: Optional[str] = Header(default=None, alias="X-Hermes-Session-Id"),
) -> StudentAgentChatResponse:
    _verify_internal_auth(authorization)
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not configured")

    session_id = _validate_header_scope(
        x_hermes_session_id or request.session_id,
        "X-Hermes-Session-Id",
    )
    session_key = _validate_header_scope(
        x_hermes_session_key or _fallback_student_session_key(request),
        "X-Hermes-Session-Key",
    )

    attachment_context = [_compact_attachment(item) for item in request.attachments]
    model_name = os.getenv("HERMES_INFERENCE_MODEL") or "qwen/qwen3.7-plus"
    response_contract = {
        "content": "student-facing tutoring answer",
        "response_type": "chat | question | assessment | explanation",
        "next_question": "optional diagnostic question",
        "question_options": [],
        "focus_mode": True,
        "concept_breakdown": [],
        "mastery": {
            "score": 0,
            "level": "beginner | developing | proficient | mastery",
            "analysis": "",
            "evidence": [],
            "suggestions": [],
        },
        "model": model_name,
        "usage": {},
    }
    # Codex change: student smoke endpoint is intentionally toolless/read-only.
    enriched_message = (
        "You are the GradeOS student learning assistant. Tutor the student with "
        "first-principles explanations and Socratic questions.\n"
        "Return exactly one JSON object with these fields: content, response_type, "
        "next_question, question_options, focus_mode, concept_breakdown, mastery, "
        "model, usage, parse_status. Do not return teacher-facing artifacts, action "
        "proposals, internal URLs, raw tokens, or hidden reasoning.\n\n"
        "<gradeos-student-context>\n"
        f"student_id: {request.student.student_id}\n"
        f"session_id: {session_id}\n"
        f"session_key: {session_key}\n"
        f"scope: {request.scope}\n"
        f"context: {request.context}\n"
        f"history: {request.history[-8:]}\n"
        f"attachments: {attachment_context}\n"
        f"tool_policy: {request.tool_policy}\n"
        f"response_contract: {response_contract}\n"
        "security: GradeOS backend is the authority for identity, class scope, "
        "conversation persistence, progress records, and safety events. Do not "
        "attempt to read or write GradeOS internal data directly.\n"
        "</gradeos-student-context>\n\n"
        f"Student question:\n{request.message}"
    )
    user_message = _multimodal_user_message(enriched_message, request.attachments)
    agent = _build_student_agent(session_id, session_key)
    result = await asyncio.to_thread(agent.run_conversation, user_message)
    content = ""
    usage: Dict[str, Any] = {}
    if isinstance(result, dict):
        raw_content = result.get("final_response") or result.get("response") or ""
        content = (
            raw_content
            if isinstance(raw_content, str)
            else json.dumps(raw_content, ensure_ascii=False)
        )
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    else:
        content = str(result)
    if not usage:
        usage = _agent_usage(agent, result, model_name)
    parsed = _json_from_text(content) or {}
    mastery = parsed.get("mastery") if isinstance(parsed.get("mastery"), dict) else {}

    return StudentAgentChatResponse(
        request_id=request.request_id,
        # CODEX CHANGE: GradeOS validates this correlation value before it
        # persists a student-assistant turn; echo the backend-issued value.
        request_fingerprint=request.request_fingerprint,
        session_id=session_id,
        session_key=session_key,
        content=str(parsed.get("content") or content),
        model=str(parsed.get("model") or model_name),
        response_type=str(parsed.get("response_type") or "explanation"),
        next_question=(
            str(parsed.get("next_question")) if parsed.get("next_question") else None
        ),
        question_options=_string_list_field(parsed, "question_options"),
        focus_mode=bool(parsed.get("focus_mode", True)),
        concept_breakdown=_list_field(parsed, "concept_breakdown"),
        mastery=mastery,
        parse_status=str(parsed.get("parse_status") or "hermes_ok"),
        parse_error_code=(
            str(parsed.get("parse_error_code")) if parsed.get("parse_error_code") else None
        ),
        safety_level=str(parsed.get("safety_level")) if parsed.get("safety_level") else None,
        usage=_usage_field(parsed, usage),
    )
