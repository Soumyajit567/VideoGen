from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from sqlalchemy.orm import Session
from models import ChatMessage, Video
from database import get_db
from fastapi_jwt_auth import AuthJWT
from pydantic import BaseModel
import logging
from datetime import datetime
from typing import Optional
from fastapi.responses import FileResponse, StreamingResponse
import os
import requests
import re  # Ensure 're' is imported

router = APIRouter()
logger = logging.getLogger(__name__)

class ChatMessageCreate(BaseModel):
    content: str

def jwt_auth_dependency(Authorize: AuthJWT = Depends()):
    try:
        Authorize.jwt_required()
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    return Authorize

@router.post("/send_message", summary="Send a chat message")
async def send_message(
    message: ChatMessageCreate,
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends(jwt_auth_dependency)
):
    user_id = Authorize.get_jwt_subject()

    # Save the user's message to the database
    chat_message = ChatMessage(
        user_id=user_id,
        content=message.content,
        is_user=True
    )
    db.add(chat_message)
    db.commit()
    db.refresh(chat_message)

    # Create a system message for video processing
    processing_message = ChatMessage(
        user_id=user_id,
        content="Your video is being processed and will appear here when ready.",
        is_user=False
    )
    db.add(processing_message)
    db.commit()
    db.refresh(processing_message)

    # Prepare the data for the /generate endpoint
    video_data = {
        "input_text": message.content,
        "video_length": 1,  # Set to 1 minute for example
        "movable_elements": []
    }

    # Get the Authorization header
    authorization_header = request.headers.get('Authorization')
    if authorization_header:
        token = authorization_header.split(" ")[1]
    else:
        logger.error("Authorization header missing")
        raise HTTPException(status_code=401, detail="Authorization header missing")

    # Define a background task to generate the video
    def generate_video_task():
        headers = {"Authorization": f"Bearer {token}"}
        # Pass the chat_message_id as a query parameter
        response = requests.post(
            f"http://127.0.0.1:8001/api/video/generate?chat_message_id={processing_message.id}",
            json=video_data,
            headers=headers
        )
        if response.status_code != 200:
            logger.error(f"Failed to generate video: {response.content}")
            # Optionally, add a system message indicating an error
            error_message = ChatMessage(
                user_id=user_id,
                content="An error occurred while generating your video.",
                is_user=False
            )
            db.add(error_message)
            db.commit()
        else:
            # The /generate endpoint handles video generation and updates the conversation
            pass

    # Add the background task
    background_tasks.add_task(generate_video_task)

    return {"message": "Message received and video generation started."}

@router.get("/get_conversation", summary="Get conversation history")
def get_conversation(
    since: Optional[str] = None,
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends(jwt_auth_dependency)
):
    user_id = Authorize.get_jwt_subject()

    query = db.query(ChatMessage).filter_by(user_id=user_id)
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
            query = query.filter(ChatMessage.timestamp > since_dt)
        except ValueError:
            logger.error("Invalid 'since' datetime format.")
            raise HTTPException(status_code=400, detail="Invalid 'since' datetime format.")

    messages = query.order_by(ChatMessage.timestamp).all()
    conversation = []
    for msg in messages:
        message_data = {
            "id": msg.id,
            "content": msg.content,
            "is_user": msg.is_user,
            "timestamp": msg.timestamp.isoformat()
        }
        if msg.video:
            # Construct the full URL for video download
            message_data["video_url"] = f"http://127.0.0.1:8001/api/chat/download_video/{msg.video.id}"
            message_data["video_id"] = msg.video.id
        conversation.append(message_data)
    return {"conversation": conversation}

@router.get("/download_video/{video_id}", summary="Download video")
async def download_video(
    video_id: int,
    request: Request,
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends(jwt_auth_dependency)
):
    user_id = Authorize.get_jwt_subject()

    # Directly filter Video by id and user_id
    video = db.query(Video).filter(
        Video.id == video_id,
        Video.user_id == user_id
    ).first()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    file_path = video.file_path

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get('range')
    if range_header:
        # Example of Range header: "bytes=0-1023"
        byte1, byte2 = 0, None
        match = re.search(r'bytes=(\d+)-(\d*)', range_header)
        if match:
            groups = match.groups()
            byte1 = int(groups[0])
            byte2 = int(groups[1]) if groups[1] else file_size - 1

        chunk_size = (byte2 - byte1) + 1
        async with aiofiles.open(file_path, 'rb') as f:
            await f.seek(byte1)
            data = await f.read(chunk_size)

        headers = {
            'Content-Range': f'bytes {byte1}-{byte2}/{file_size}',
            'Accept-Ranges': 'bytes',
            'Content-Length': str(chunk_size),
            'Content-Type': 'video/mp4',
        }

        return StreamingResponse(iter([data]), status_code=206, headers=headers)
    else:
        return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type='video/mp4')
