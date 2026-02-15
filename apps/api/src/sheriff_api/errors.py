from collections.abc import Mapping

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def _default_error_code(status_code: int) -> str:
    if status_code == 400:
        return "bad_request"
    if status_code == 401:
        return "unauthorized"
    if status_code == 403:
        return "forbidden"
    if status_code == 404:
        return "not_found"
    if status_code == 409:
        return "conflict"
    if status_code == 422:
        return "validation_error"
    if status_code >= 500:
        return "internal_error"
    return "request_error"


def _normalize_details(detail: object) -> dict | None:
    if detail is None:
        return None
    if isinstance(detail, Mapping):
        return dict(detail)
    if isinstance(detail, str):
        return {"reason": detail}
    return {"reason": str(detail)}


def _with_request_context(details: dict | None, request: Request) -> dict:
    context = {
        "request_path": request.url.path,
        "request_method": request.method,
    }
    if details is None:
        return context
    return {**details, **context}


def build_error_response(
    *,
    code: str,
    message: str,
    request: Request,
    details: dict | None = None,
    status_code: int,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": _with_request_context(details, request),
            }
        },
    )


def api_error(status_code: int, *, code: str, message: str, details: dict | None = None) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
            "details": details,
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    details = None
    code = _default_error_code(exc.status_code)
    message = "Request failed"

    if isinstance(detail, Mapping):
        raw_code = detail.get("code")
        raw_message = detail.get("message")
        details = _normalize_details(detail.get("details"))
        if isinstance(raw_code, str) and raw_code:
            code = raw_code
        if isinstance(raw_message, str) and raw_message:
            message = raw_message
    elif isinstance(detail, str):
        message = detail
    elif detail is not None:
        message = str(detail)

    return build_error_response(
        code=code,
        message=message,
        request=request,
        details=details,
        status_code=exc.status_code,
    )


async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return build_error_response(
        code="validation_error",
        message="Request validation failed",
        request=request,
        details={"issues": exc.errors()},
        status_code=422,
    )
