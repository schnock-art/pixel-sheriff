from fastapi import APIRouter

from sheriff_api.errors import api_error

router = APIRouter(tags=["exports"])

LEGACY_MESSAGE = "Legacy project export endpoints are retired; use dataset-version export endpoints under /datasets/versions/{id}/export."


@router.post("/projects/{project_id}/exports")
async def create_export_legacy(project_id: str) -> dict:
    raise api_error(
        status_code=410,
        code="exports_legacy_gone",
        message=LEGACY_MESSAGE,
        details={"project_id": project_id},
    )


@router.get("/projects/{project_id}/exports")
async def list_exports_legacy(project_id: str) -> dict:
    raise api_error(
        status_code=410,
        code="exports_legacy_gone",
        message=LEGACY_MESSAGE,
        details={"project_id": project_id},
    )


@router.get("/projects/{project_id}/exports/{content_hash}/download")
async def download_export_legacy(project_id: str, content_hash: str) -> dict:
    raise api_error(
        status_code=410,
        code="exports_legacy_gone",
        message=LEGACY_MESSAGE,
        details={"project_id": project_id, "content_hash": content_hash},
    )
