# deepfake_api.py
import cv2
from PIL import Image
from transformers import pipeline
from fastapi import FastAPI, UploadFile, File
import uvicorn
import tempfile
import os

app = FastAPI()

pipe = pipeline("image-classification",
                model="prithivMLmods/Deep-Fake-Detector-v2-Model")

def extract_frames(video_path, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames


@app.post("/analyze_video/")
async def analyze_video(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Extract frames
    frames = extract_frames(tmp_path, frame_rate=30)

    results = []
    for frame in frames[:10]:  # limit to 10 frames for speed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        result = pipe(pil_img)
        results.append(result)

    # Remove temp file
    os.remove(tmp_path)

    # Aggregate results
    counts = {}
    for r in results:
        for item in r:
            label = item["label"]
            counts[label] = counts.get(label, 0) + item["score"]

    # Normalize scores
    total = sum(counts.values())
    aggregated = {k: v / total for k, v in counts.items()}

    return {
        "frame_count": len(frames),
        "analyzed_frames": len(results),
        "aggregated_scores": aggregated
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
