import os
import shutil
import uuid
import time
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Set Matplotlib and Ultralytics cache directories
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
os.environ["YOLO_CONFIG_DIR"] = "/tmp/yolo"
os.makedirs("/tmp/mpl", exist_ok=True)
os.makedirs("/tmp/yolo", exist_ok=True)

from engine import ANPREngine

app = FastAPI(
    title="ANPR Indian Vehicle Number Plate Recognition",
    description="Automatic Number Plate Recognition system for Indian HSRP and standard vehicle registration plates.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Workspace directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, "media")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(MEDIA_DIR, exist_ok=True)

# Initialize ANPR Engine
engine = ANPREngine(
    coco_path=os.path.join(BASE_DIR, "yolov8n.pt"),
    plate_path=os.path.join(BASE_DIR, "models", "license_plate_detector.pt")
)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Template not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/detect-image")
async def detect_image(file: UploadFile = File(...)):
    """Upload and process a single vehicle image."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file format")

        result = engine.detect_image(frame)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect-sample")
async def detect_sample():
    """Runs detection on sample car image for immediate 1-click test."""
    sample_path = os.path.join(BASE_DIR, "sample_test.jpg")
    if not os.path.exists(sample_path):
        # Generate sample frame from video if available
        sample_vid = os.path.join(BASE_DIR, "Files", "License Plate Detection Test.f399.mp4")
        if os.path.exists(sample_vid):
            cap = cv2.VideoCapture(sample_vid)
            ret, frame = cap.read()
            cap.release()
            if ret:
                cv2.imwrite(sample_path, frame)

    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample image not found")

    frame = cv2.imread(sample_path)
    result = engine.detect_image(frame)
    return result


@app.post("/api/detect-video")
async def detect_video(file: UploadFile = File(...)):
    """Upload and process traffic video with SORT vehicle tracking."""
    unique_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(MEDIA_DIR, f"input_{unique_id}_{file.filename}")
    output_filename = f"tracked_{unique_id}.mp4"
    output_path = os.path.join(MEDIA_DIR, output_filename)

    try:
        t_start = time.time()
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process up to 150 frames to provide fast real-time web responsiveness
        video_result = engine.process_video(input_path, output_path, max_frames=150)
        elapsed = round(time.time() - t_start, 2)
        detections = video_result["detections"]
        plates_count = sum(1 for d in detections if d.get("has_plate"))

        return {
            "success": True,
            "inference_time_seconds": elapsed,
            "video_url": f"/media/{output_filename}",
            "frames_processed": video_result["frames_processed"],
            "vehicles_found": len(detections),
            "plates_found": plates_count,
            "detections": detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded raw input to save disk space
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass


@app.get("/media/{filename}")
async def get_media(filename: str):
    file_path = os.path.join(MEDIA_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Media file not found")
    return FileResponse(file_path, media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
