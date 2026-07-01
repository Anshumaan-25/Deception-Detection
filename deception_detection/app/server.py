import os
import json
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Defense-Grade Path Setup
BASE_DIR = Path("pipeline_system_outputs").resolve()
FRONTEND_DIST = Path("frontend/dist").resolve()
LEDGER_PATH = BASE_DIR / "batch_ledger.json"

app = FastAPI(title="SPOVNOB Control Center - Production")

# 1. Performance: Compress heavy JSON payloads
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 2. Security: Strict CORS bounds
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["Range", "Accept-Ranges", "Content-Type"],
)

def secure_resolve(target_path: Path) -> Path:
    """Hardlocked Path Traversal Protection"""
    resolved = target_path.resolve()
    if not resolved.is_relative_to(BASE_DIR):
        raise HTTPException(status_code=403, detail="FATAL: Path Traversal Violation. Access Denied.")
    return resolved

@app.get("/api/sessions")
def get_sessions():
    sessions = []
    if not BASE_DIR.exists():
        return {"sessions": sessions}
    
    for session_path in BASE_DIR.iterdir():
        if session_path.is_dir():
            metadata_file = secure_resolve(session_path / "metadata.json")
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        meta = json.load(f)
                    sessions.append(meta)
                except Exception:
                    pass
    return {"sessions": sessions}

@app.get("/api/data/{session_id}")
def get_session_data(session_id: str):
    session_dir = secure_resolve(BASE_DIR / session_id)
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
        
    calibrated_path = session_dir / f"{session_id}_calibrated_features.csv"
    windowed_path = session_dir / f"{session_id}_windowed_features.csv"
    
    target_path = calibrated_path if calibrated_path.exists() else windowed_path
    target_path = secure_resolve(target_path)
    
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="No feature matrix found")
        
    try:
        df = pd.read_csv(target_path)
        # Clean nulls for JSON
        df = df.replace({np.nan: None})
        # Performance: split serialization drops massive key duplication bloat
        payload = df.to_dict(orient="split")
        return JSONResponse(content={"data": payload})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/video/{session_id}")
def stream_video(session_id: str, request: Request):
    """Native HTTP 206 Byte-Range MP4 Streamer"""
    session_dir = secure_resolve(BASE_DIR / session_id)
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
        
    video_files = list(session_dir.glob("*.mp4"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Canonical video missing")
        
    video_path = secure_resolve(video_files[0])
    file_size = video_path.stat().st_size
    range_header = request.headers.get("Range")
    
    if not range_header:
        headers = {
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Content-Type": "video/mp4",
        }
        def file_iterator():
            with open(video_path, "rb") as f:
                while chunk := f.read(1024 * 1024 * 4): # 4MB chunks for fast workstation bus
                    yield chunk
        return StreamingResponse(file_iterator(), headers=headers)
        
    try:
        range_str = range_header.replace("bytes=", "").split("-")
        start = int(range_str[0])
        end = int(range_str[1]) if len(range_str) > 1 and range_str[1] else file_size - 1
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Range Header")
        
    if start >= file_size or end >= file_size:
        raise HTTPException(status_code=416, detail="Range Not Satisfiable")
        
    chunk_size = end - start + 1
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
        "Content-Type": "video/mp4",
    }
    
    def ranged_file_iterator():
        with open(video_path, "rb") as f:
            f.seek(start)
            bytes_left = chunk_size
            while bytes_left > 0:
                chunk = f.read(min(1024 * 1024 * 4, bytes_left))
                if not chunk:
                    break
                bytes_left -= len(chunk)
                yield chunk
                
    return StreamingResponse(ranged_file_iterator(), status_code=206, headers=headers)

# ═══════════════════════════════════════════════════════════════════
# FACTORY STATUS & SERVER-SENT EVENTS (SSE) ENDPOINT
# ═══════════════════════════════════════════════════════════════════

def _read_ledger() -> dict:
    """Safely reads the batch_ledger.json from disk."""
    if not LEDGER_PATH.exists():
        return {}
    try:
        with open(LEDGER_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

@app.get("/api/factory/status")
def factory_status():
    """Returns the current state of the batch processing ledger as a JSON snapshot."""
    ledger = _read_ledger()
    summary = {
        "total": len(ledger),
        "queued": 0,
        "active": 0,
        "completed": 0,
        "failed": 0,
        "interrupted": 0,
    }
    for sid, entry in ledger.items():
        state = entry.get("state", "UNKNOWN")
        if state == "QUEUED":
            summary["queued"] += 1
        elif state in ("AUDIO_PROCESSING", "TENSORRT_ACTIVE", "MATH_NORMALIZATION"):
            summary["active"] += 1
        elif state == "COMPLETED":
            summary["completed"] += 1
        elif state == "FAILED":
            summary["failed"] += 1
        elif state == "INTERRUPTED":
            summary["interrupted"] += 1

    return {"summary": summary, "sessions": ledger}

@app.get("/api/factory/stream")
async def factory_stream():
    """
    Long-lived Server-Sent Events (SSE) broadcast endpoint.
    Reads the batch ledger every 2 seconds and pushes state updates
    to connected dashboard clients.
    """
    async def event_generator():
        last_snapshot = ""
        while True:
            ledger = _read_ledger()
            current_snapshot = json.dumps(ledger, sort_keys=True)

            # Only push an event if the ledger state actually changed
            if current_snapshot != last_snapshot:
                last_snapshot = current_snapshot
                payload = json.dumps({
                    "type": "ledger_update",
                    "data": ledger,
                })
                yield f"data: {payload}\n\n"

            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ═══════════════════════════════════════════════════════════════════
# PRODUCTION SPA CATCH-ALL ROUTER
# ═══════════════════════════════════════════════════════════════════

@app.get("/assets/{file_path:path}")
def serve_assets(file_path: str):
    asset_path = (FRONTEND_DIST / "assets" / file_path).resolve()
    if not asset_path.is_relative_to(FRONTEND_DIST):
        raise HTTPException(status_code=403, detail="Invalid asset path")
    if not asset_path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(asset_path)

@app.get("/{full_path:path}")
def catch_all(full_path: str):
    """Fallback handler to prevent SPA Sub-Path Refresh Trap"""
    index_file = (FRONTEND_DIST / "index.html").resolve()
    if not index_file.exists():
        return JSONResponse({"error": "Production build not found. Run npm run build in frontend/"}, status_code=404)
    return FileResponse(index_file)

if __name__ == "__main__":
    import uvicorn
    print("🚀 SPOVNOB Production Visualizer Sidecar running on port 8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

