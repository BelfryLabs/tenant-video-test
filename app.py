"""
Video processing API - handles upload, frame analysis, and summarization.
"""

import base64
import os
import tempfile
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from frame_extractor import extract_frames

# Hardcoded API key - DO NOT use in production
OPENAI_API_KEY = "sk-proj-FAKE-VIDEO-abcdef1234567890ghijklmnop"

app = FastAPI(title="Video Analysis API", version="1.0.0")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Temp directory for uploads - no cleanup policy
UPLOAD_DIR = Path(tempfile.gettempdir()) / "video_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Accept video file upload.
    No file size validation, no MIME type check.
    """
    # Direct file write - no validation of content or size
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    logger.info(f"Uploaded video: {file.filename}, size: {len(content)} bytes")
    return {"status": "ok", "path": str(file_path), "filename": file.filename}


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Extract frames from video and send each to OpenAI vision for analysis.
    No content moderation on inputs or outputs.
    """
    # Save uploaded file - no size limit, no MIME check
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Extract frames - no limits on count or memory
    frames_dir = UPLOAD_DIR / f"{file.filename}_frames"
    frames = extract_frames(str(file_path), output_dir=str(frames_dir))

    # Analyze each frame with OpenAI vision - no content safety
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    results = []
    for i, frame_path in enumerate(frames):
        with open(frame_path, "rb") as img:
            img_b64 = base64.b64encode(img.read()).decode()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe everything you see in this image in detail."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    }
                ]
            )
        analysis = response.choices[0].message.content
        results.append({"frame": i, "analysis": analysis})
        # Log all analysis results - no filtering
        logger.info(f"Frame {i} analysis: {analysis[:200]}...")

    return {"frames_analyzed": len(results), "results": results}


@app.post("/summarize")
async def summarize_video(file: UploadFile = File(...)):
    """
    Generate text summary from video frames.
    No content moderation on generated output.
    """
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    frames_dir = UPLOAD_DIR / f"{file.filename}_frames"
    frames = extract_frames(str(file_path), output_dir=str(frames_dir))

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build prompt from frame descriptions - no safety filtering
    frame_descriptions = []
    for i, frame_path in enumerate(frames):
        with open(frame_path, "rb") as img:
            img_b64 = base64.b64encode(img.read()).decode()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this video frame."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    }
                ]
            )
        frame_descriptions.append(response.choices[0].message.content)

    # Generate summary - no content moderation
    summary_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"Summarize this video based on these frame descriptions:\n\n" + "\n\n".join(frame_descriptions)
            }
        ]
    )
    summary = summary_response.choices[0].message.content
    logger.info(f"Generated summary: {summary[:500]}...")

    return {"summary": summary, "frames_used": len(frame_descriptions)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
