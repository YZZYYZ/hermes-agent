from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from run_agent import AIAgent
from tools.gradeos_tools import set_gradeos_session_scope


app = FastAPI(title="GradeOS Hermes Smoke Server")

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
        "auth_configured": bool(os.getenv("HERMES_AGENT_SERVICE_TOKEN")),
        "gradeos_internal_api": os.getenv("GRADEOS_INTERNAL_API_BASE_URL", "http://127.0.0.1:8001"),
    }


def _expected_token() -> Optional[str]:
    # Codex change: mirror GradeOS teacher Hermes auth and fail closed without a service token.
    token = os.getenv("HERMES_AGENT_SERVICE_TOKEN")
    return token.strip() if token and token.strip() else None


def _verify_internal_auth(authorization: Optional[str]) -> None:
    expected = _expected_token()
    if not expected:
        raise HTTPException(status_code=503, detail="HERMES_AGENT_SERVICE_TOKEN is not configured")
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


# Codex change: verify student-scoped GradeOS tokens before Hermes handles student chats.
def _student_scope_signing_secret() -> Optional[str]:
    secret = os.getenv("GRADEOS_HERMES_SCOPE_SIGNING_SECRET")
    return secret.strip() if secret and secret.strip() else None


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}".encode("ascii"))


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _student_scope_signature(encoded_payload: str, secret: str) -> str:
    digest = hmac.new(
        secret.encode("utf-8"),
        encoded_payload.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(digest)


def _clean_student_scope(scope: Dict[str, Any]) -> Dict[str, Any]:
    allowed_keys = {
        "batch_id",
        "class_id",
        "homework_id",
        "conversation_id",
        "org_id",
        "tenant_id",
    }
    cleaned: Dict[str, Any] = {}
    for key in allowed_keys:
        value = scope.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = str(value) if not isinstance(value, bool) else value
    return cleaned


def _verify_student_scope_token(
    token: Optional[str],
    *,
    student_id: str,
    session_id: str,
    scope: Dict[str, Any],
) -> None:
    secret = _student_scope_signing_secret()
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="GRADEOS_HERMES_SCOPE_SIGNING_SECRET is not configured",
        )
    if not token or "." not in token:
        raise HTTPException(status_code=403, detail="Missing GradeOS student scope token")

    encoded_payload, signature = token.split(".", 1)
    expected_signature = _student_scope_signature(encoded_payload, secret)
    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(status_code=403, detail="Invalid GradeOS student scope token")

    try:
        payload = json.loads(_b64url_decode(encoded_payload))
    except Exception as exc:
        raise HTTPException(status_code=403, detail="Invalid GradeOS student scope payload") from exc

    if payload.get("v") != 1 or payload.get("principal_type") != "student":
        raise HTTPException(status_code=403, detail="Unsupported GradeOS student scope token")
    if int(payload.get("exp") or 0) <= int(time.time()):
        raise HTTPException(status_code=403, detail="Expired GradeOS student scope token")
    if str(payload.get("student_id") or "").strip() != student_id:
        raise HTTPException(status_code=403, detail="Student is outside signed GradeOS scope")
    if str(payload.get("session_id") or "").strip() != session_id:
        raise HTTPException(status_code=403, detail="Session is outside signed GradeOS scope")

    signed_scope = _clean_student_scope(
        payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    )
    request_scope = _clean_student_scope(scope)
    if signed_scope != request_scope:
        raise HTTPException(status_code=403, detail="Request scope is outside signed GradeOS scope")


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


def _prompt_safe_dict(value: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        lowered = key_text.lower()
        if any(secret in lowered for secret in ("token", "secret", "key", "auth", "url")):
            continue
        if lowered in {"teacher_id", "user_id", "student_id", "session_id", "session_key"}:
            safe[key_text] = "[scoped]"
        else:
            safe[key_text] = item
    return safe


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
    set_gradeos_session_scope(
        session_id,
        teacher_id=request.teacher.teacher_id,
        scope=request.scope,
    )
    agent = _build_agent(session_id, session_key)
    result = await asyncio.to_thread(agent.run_conversation, enriched_message)
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
    request_started = time.perf_counter()
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
    student_id = str(request.student.student_id or "").strip()
    if not student_id:
        raise HTTPException(status_code=400, detail="student_id is required")
    _verify_student_scope_token(
        request.gradeos_scope_token,
        student_id=student_id,
        session_id=session_id,
        scope=request.scope,
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
        "authenticated_student: true\n"
        f"scope: {_prompt_safe_dict(request.scope)}\n"
        f"context: {_prompt_safe_dict(request.context)}\n"
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
    build_started = time.perf_counter()
    agent = _build_student_agent(session_id, session_key)
    agent_build_ms = (time.perf_counter() - build_started) * 1000
    run_started = time.perf_counter()
    result = await asyncio.to_thread(agent.run_conversation, enriched_message)
    run_conversation_ms = (time.perf_counter() - run_started) * 1000
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
    response_usage = dict(_usage_field(parsed, usage))
    response_usage["hermes_timings_ms"] = {
        "agent_build_ms": round(agent_build_ms, 2),
        "run_conversation_ms": round(run_conversation_ms, 2),
        "total_ms": round((time.perf_counter() - request_started) * 1000, 2),
    }

    return StudentAgentChatResponse(
        request_id=request.request_id,
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
        usage=response_usage,
    )
