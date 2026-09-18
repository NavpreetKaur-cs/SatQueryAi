"""
SatQuery AI backend.

Run from inside backend/:
    uvicorn main:app --reload --port 8000

See ../backend/README.md for full setup, endpoint docs, and known gaps.
"""

import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import agent_client
import image_store
from schemas import QueryRequest, QueryResponse, UploadResponse, VisualOutput

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]

app = FastAPI(title="SatQuery AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves uploaded originals + generated previews at /uploads/originals/... and
# /uploads/previews/... so the frontend can load them directly as <img> src.
app.mount("/uploads", StaticFiles(directory=REPO_ROOT / "uploads"), name="uploads")


@app.get("/health")
def health():
    return {"status": "ok", "using_real_agent": agent_client.USING_REAL_AGENT}


@app.post("/images", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Dedicated image-upload endpoint (separate from /query, per the team
    doc's 'Create APIs for image upload and analysis requests'). Returns an
    imageId the frontend can then reference in /query instead of resending
    the whole image every time.
    """
    try:
        data = await file.read()
        record = image_store.save_upload_bytes(data, file.filename or "upload")
        return UploadResponse(
            success=True,
            imageId=record["imageId"],
            width=record["width"],
            height=record["height"],
            format=record["format"],
            previewUrl=record["previewUrl"],
        )
    except image_store.ImageStoreError as e:
        return UploadResponse(success=False, error=str(e))
    except Exception as e:  # noqa: BLE001 — surface unexpected errors in the contract shape too
        traceback.print_exc()
        return UploadResponse(success=False, error=f"Unexpected error: {e}")


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Main analysis endpoint. Accepts either an imageId (from a prior
    /images upload) or an inline dataUrl for each image — see
    schemas.ImageInput. Calls into agent_client, which transparently uses
    the real agent/controller once Person 5's module exists, or a
    keyword-based mock until then.
    """
    try:
        before_record = image_store.resolve_image(req.images.before.imageId, req.images.before.dataUrl)
        after_record = None
        if req.images.after:
            after_record = image_store.resolve_image(req.images.after.imageId, req.images.after.dataUrl)

        agent_result = agent_client.run_query(
            query=req.query,
            images={
                "before": {
                    "path": before_record["path"],
                    "width": before_record["width"],
                    "height": before_record["height"],
                },
                "after": (
                    {
                        "path": after_record["path"],
                        "width": after_record["width"],
                        "height": after_record["height"],
                    }
                    if after_record
                    else None
                ),
            },
            metadata=req.metadata,
        )

        if not agent_result.get("success", True):
            return QueryResponse(
                success=False,
                error=agent_result.get("error") or "Agent reported failure with no error message.",
                metadata={"queryType": agent_result.get("queryType")},
            )

        return QueryResponse(
            success=True,
            result=agent_result.get("result"),
            confidence=agent_result.get("confidence"),
            visual_output=VisualOutput(regions=agent_result.get("regions", [])),
            metadata={
                "queryType": agent_result.get("queryType"),
                "modelsUsed": agent_result.get("modelsUsed", []),
                "executionSummary": agent_result.get("executionSummary"),
            },
        )
    except image_store.ImageStoreError as e:
        return QueryResponse(success=False, error=str(e))
    except Exception as e:  # noqa: BLE001 — always answer in the contract shape, never a bare 500
        traceback.print_exc()
        return QueryResponse(success=False, error=f"Unexpected error: {e}")
