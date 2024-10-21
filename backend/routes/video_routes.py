# # routes/video_routes.py

# import os
# import logging
# import numpy as np
# from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
# from sqlalchemy.orm import Session
# from fastapi_jwt_auth import AuthJWT
# from fastapi_jwt_auth.exceptions import InvalidHeaderError
# from fastapi.responses import FileResponse, StreamingResponse
# from database import get_db, SessionLocal
# from models import Video, ChatMessage  # Import ChatMessage
# from pydantic import BaseModel, validator
# import traceback
# from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip, CompositeVideoClip, ImageClip
# from openai import OpenAI
# from dotenv import load_dotenv
# from PIL import Image
# import io
# import requests
# import time
# import re
# from typing import List, Optional, Callable
# from datetime import datetime

# # Import the sanitization function
# from .utils import sanitize_filename  # Ensure utils.py is in the same directory

# # Load environment variables from .env file
# load_dotenv()

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize OpenAI client with API key from environment variable
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     logger.error("OPENAI_API_KEY environment variable not set.")
#     raise ValueError("OPENAI_API_KEY environment variable not set.")

# client = OpenAI(api_key=OPENAI_API_KEY)

# router = APIRouter()

# # Helper function to get unique file path
# def get_unique_file_path(file_path: str) -> str:
#     if not os.path.exists(file_path):
#         return file_path

#     base, extension = os.path.splitext(file_path)
#     counter = 1
#     new_file_path = f"{base}_{counter}{extension}"

#     while os.path.exists(new_file_path):
#         counter += 1
#         new_file_path = f"{base}_{counter}{extension}"
#         if counter > 1000:
#             raise Exception("Too many files with the same name exist.")

#     return new_file_path

# # Pydantic model for movable elements
# class MovableElement(BaseModel):
#     element_path: str
#     element_type: str
#     movement: str = 'linear'
#     position: Optional[str] = 'center'
#     size: Optional[tuple] = None
#     radius: Optional[int] = None
#     center_x: Optional[int] = None
#     center_y: Optional[int] = None
#     custom_path: Optional[Callable[[float], tuple]] = None
#     y_position: Optional[int] = 50

# # Pydantic model for video creation
# class VideoCreate(BaseModel):
#     input_text: str
#     video_length: int  # in minutes
#     movable_elements: Optional[List[MovableElement]] = []

#     @validator('video_length')
#     def video_length_must_be_positive(cls, v):
#         if v <= 0:
#             raise ValueError('Video length must be positive')
#         return v

#     @validator('input_text')
#     def validate_input_text(cls, v):
#         if re.search(r'[<>:"/\\|?*]', v):
#             raise ValueError("Input text contains invalid characters for filenames.")
#         return v

# # Function to sanitize the prompt
# def sanitize_prompt(prompt: str) -> str:
#     sanitized = re.sub(r'[^\w\s.,!?]', '', prompt)
#     sanitized = re.sub(r'\s+', ' ', sanitized)
#     return sanitized.strip()

# # Function to generate keyframe descriptions with retry logic
# def generate_keyframe_description(input_text: str, num_visuals: int = 5, max_retries: int = 3) -> list:
#     attempt = 0
#     backoff = 2  # Seconds
#     while attempt < max_retries:
#         try:
#             prompt = (
#                 f"Provide {num_visuals} detailed and distinct visual descriptions for the following video theme. "
#                 f"Each description should start with 'Visual X:', where X is the visual number (e.g., 'Visual 1:').\n"
#                 f"Theme: {input_text}"
#             )

#             completion = client.chat.completions.create(
#                 model="gpt-4",
#                 messages=[
#                     {"role": "system", "content": "You are a creative assistant specializing in visual storytelling."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=800
#             )
#             description = completion.choices[0].message.content.strip()
#             logger.info(f"Generated keyframe description: {description}")

#             # Use regex to extract all Visual X: descriptions
#             pattern = rf"Visual\s+\d+:\s*(.*?)\n(?=Visual\s+\d+:|$)"
#             visuals = re.findall(pattern, description, re.DOTALL)

#             # Prepend "Visual X:" to each visual
#             visuals = [f"Visual {i + 1}: {visual.strip()}" for i, visual in enumerate(visuals)]

#             if len(visuals) < num_visuals:
#                 logger.warning(f"Expected {num_visuals} visuals, but got {len(visuals)}.")

#             logger.info(f"Successfully generated {len(visuals)} visuals.")
#             return visuals
#         except Exception as e:
#             attempt += 1
#             logger.error(f"Attempt {attempt} - Error in generating keyframe description: {e}")
#             if attempt < max_retries:
#                 logger.info(f"Retrying in {backoff} seconds...")
#                 time.sleep(backoff)
#                 backoff *= 2  # Exponential backoff
#             else:
#                 logger.error("Max retries reached. Unable to generate keyframe descriptions.")
#                 raise HTTPException(status_code=500, detail=f"Failed to generate keyframe description: {str(e)}")

# # Function to generate images based on descriptions using OpenAI's DALL·E with error handling
# def generate_images(descriptions: list, num_images: int = 5) -> list:
#     try:
#         images = []
#         for i, description in enumerate(descriptions):
#             if i >= num_images:
#                 break  # Limit to the specified number of images

#             logger.info(f"Generating image {i + 1} of {num_images} with description: {description}")

#             # Sanitize the description
#             sanitized_description = sanitize_prompt(description)
#             logger.debug(f"Sanitized description for image {i + 1}: {sanitized_description}")

#             try:
#                 response = client.images.generate(
#                     model="dall-e-3",
#                     prompt=sanitized_description,
#                     n=1,
#                     size="1024x1024"
#                 )
#                 image_url = response.data[0].url
#                 logger.debug(f"Image URL for image {i + 1}: {image_url}")

#                 img_response = requests.get(image_url)
#                 if img_response.status_code != 200:
#                     logger.error(f"Failed to fetch image {i + 1} from URL: {image_url}")
#                     continue
#                 img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
#                 images.append(img)
#                 logger.info(f"Generated and fetched image {i + 1}.")
#                 time.sleep(1)  # Sleep to respect rate limits

#             except Exception as e:
#                 logger.error(f"Error during image generation for image {i + 1}: {e}")
#                 logger.error(f"Prompt causing error: {sanitized_description}")
#                 continue

#         if not images:
#             raise HTTPException(status_code=500, detail="No images were generated.")

#         logger.info(f"Successfully generated {len(images)} images.")
#         return images
#     except Exception as e:
#         logger.error(f"Error in generating images: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to generate images: {str(e)}")

# # Function to save PIL Image to disk
# def save_images(images: list, base_path: str):
#     try:
#         os.makedirs(base_path, exist_ok=True)
#         for idx, img in enumerate(images):
#             img_path = os.path.join(base_path, f"image_{idx + 1}.png")
#             img.save(img_path)
#             logger.info(f"Saved image {idx + 1} at {img_path}")
#     except Exception as e:
#         logger.error(f"Error in saving images: {e}")
#         raise HTTPException(status_code=500, detail="Failed to save images")

# # Function to generate video frames based on descriptions using images
# def generate_video_frames_from_images(images: list, duration: int = 300, fps: int = 24) -> list:
#     try:
#         processed_images = []
#         for img in images:
#             img = img.resize((640, 480))  # Resize to desired video resolution
#             processed_images.append(np.array(img))

#         total_frames = duration * fps
#         frames = []
#         num_images = len(processed_images)
#         frames_per_image = total_frames // num_images if num_images else 1

#         for img in processed_images:
#             for _ in range(frames_per_image):
#                 frames.append(img)

#         # If there are remaining frames, add them to the last image
#         remaining_frames = duration * fps - frames_per_image * num_images
#         for _ in range(remaining_frames):
#             frames.append(processed_images[-1] if processed_images else np.zeros((480, 640, 3), dtype=np.uint8))

#         logger.info("Generated video frames from images.")
#         return frames
#     except Exception as e:
#         logger.error(f"Error in generating video frames from images: {e}")
#         raise HTTPException(status_code=500, detail="Failed to generate video frames from images")

# # Function to save video frames as a video file using moviepy
# def save_video(frames: list, path: str, fps: int = 24):
#     try:
#         logger.info(f"Saving video frames to {path} with {fps} FPS.")
#         clip = ImageSequenceClip(frames, fps=fps)
#         clip.write_videofile(path, codec='libx264', audio=False)
#         logger.info(f"Video saved at path: {path}")
#     except Exception as e:
#         logger.error(f"Error in saving video: {e}")
#         raise HTTPException(status_code=500, detail="Failed to save video")

# # Function to generate speech using OpenAI's audio.speech.create
# def generate_audio(text: str, audio_path: str):
#     try:
#         logger.info(f"Generating audio narration and saving to {audio_path}.")
#         response = client.audio.speech.create(
#             model="tts-1",
#             voice="alloy",
#             input=text
#         )
#         response.stream_to_file(audio_path)
#         logger.info(f"Audio saved at path: {audio_path}")
#     except Exception as e:
#         logger.error(f"Error in generating audio: {e}")
#         raise HTTPException(status_code=500, detail="Failed to generate audio")

# # Function to assemble video with audio
# def assemble_video_with_audio(video_path: str, audio_path: str, output_video_path: str):
#     try:
#         logger.info(f"Assembling video from {video_path} and audio from {audio_path} into {output_video_path}.")
#         video = VideoFileClip(video_path)
#         audio = AudioFileClip(audio_path)

#         # Log original durations
#         logger.info(f"Original video duration: {video.duration} seconds")
#         logger.info(f"Original audio duration: {audio.duration} seconds")

#         # Ensure the audio duration matches the video duration
#         audio = audio.set_duration(video.duration)
#         logger.info(f"Trimmed audio duration: {audio.duration} seconds")

#         video = video.set_audio(audio)
#         video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
#         logger.info(f"Final video assembled successfully at path: {output_video_path}")
#     except Exception as e:
#         logger.error(f"Error in assembling video with audio: {e}")
#         raise HTTPException(status_code=500, detail="Failed to assemble video with audio")

# # Function to add moving elements overlays
# def add_moving_elements(
#     base_video_path: str,
#     movable_elements: list,  # List of MovableElement
#     output_video_path: str
# ):
#     try:
#         # Load the base video
#         video = VideoFileClip(base_video_path)
#         duration = video.duration

#         # List to hold all overlay clips
#         overlays = []

#         for element in movable_elements:
#             if element.element_type == 'image':
#                 overlay = ImageClip(element.element_path, transparent=True).set_duration(duration)
#             elif element.element_type == 'video':
#                 overlay = VideoFileClip(element.element_path, has_mask=True).set_duration(duration)
#             else:
#                 logger.warning(f"Invalid element_type '{element.element_type}' for element '{element.element_path}'. Skipping.")
#                 continue

#             # Resize if size is specified
#             if element.size:
#                 overlay = overlay.resize(newsize=element.size)

#             # Define the movement path
#             if element.movement == 'linear':
#                 # Move from left to right
#                 overlay = overlay.set_position(lambda t: (int(640 * (t / duration)) - overlay.w / 2, element.y_position))
#             elif element.movement == 'circular':
#                 # Circular movement around a point
#                 radius = element.radius if element.radius else 100
#                 center_x = element.center_x if element.center_x else 320
#                 center_y = element.center_y if element.center_y else 240

#                 overlay = overlay.set_position(lambda t: (
#                     int(center_x + radius * np.cos(2 * np.pi * t / duration)) - overlay.w / 2,
#                     int(center_y + radius * np.sin(2 * np.pi * t / duration)) - overlay.h / 2
#                 ))
#             elif element.movement == 'path' and element.custom_path:
#                 # Custom path defined by a function
#                 overlay = overlay.set_position(element.custom_path)
#             else:
#                 # Static position or unrecognized movement type
#                 overlay = overlay.set_position(element.position)

#             overlays.append(overlay)

#         if overlays:
#             # Composite all overlays onto the video
#             final = CompositeVideoClip([video] + overlays)

#             # Write the final video
#             final.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
#             logger.info(f"Final video with moving elements saved at {output_video_path}")
#         else:
#             # If no valid overlays, simply copy the base video to output
#             os.rename(base_video_path, output_video_path)
#             logger.info(f"No movable elements provided. Base video saved as {output_video_path}")

#     except Exception as e:
#         logger.error(f"Error in adding moving elements: {e}")
#         raise HTTPException(status_code=500, detail="Failed to add moving elements to video")

# # Authentication dependency using JWT
# def jwt_auth_dependency(Authorize: AuthJWT = Depends()):
#     try:
#         Authorize.jwt_required()
#     except InvalidHeaderError as e:
#         logger.error(f"InvalidHeaderError: {e}")
#         raise HTTPException(status_code=401, detail=str(e))
#     except Exception as e:
#         logger.error(f"Authentication error: {e}")
#         raise HTTPException(status_code=401, detail="Invalid token")
#     return Authorize

# # API endpoint to generate a new video
# @router.post("/generate", tags=["Video"], summary="Generate a new video", response_model=dict)
# async def generate_video(
#     request: Request,
#     video: VideoCreate,
#     background_tasks: BackgroundTasks,
#     db: Session = Depends(get_db),
#     Authorize: AuthJWT = Depends(jwt_auth_dependency)
# ):
#     try:
#         logger.info(f"Incoming headers: {request.headers}")

#         current_user_id = Authorize.get_jwt_subject()
#         logger.info(f"User ID: {current_user_id}")

#         if video.video_length > 15:
#             logger.error("Video length exceeds maximum limit")
#             raise HTTPException(status_code=400, detail="Video length cannot exceed 15 minutes")

#         # Generate keyframe descriptions with reduced number of visuals
#         keyframe_descriptions = generate_keyframe_description(video.input_text, num_visuals=5)  # Reduced from 20 to 5
#         logger.info(f"Keyframe descriptions: {keyframe_descriptions}")

#         # Define directories with sanitized input
#         sanitized_input = sanitize_filename(video.input_text.strip())
#         logger.info(f"Sanitized input for directory: {sanitized_input}")

#         generated_images_path = os.path.join("generated_images", sanitized_input)
#         assembled_video_path = os.path.join("generated_videos", f"{sanitized_input}_assembled.mp4")
#         final_video_path = os.path.join("generated_videos", f"{sanitized_input}_final.mp4")

#         # Ensure directories exist
#         os.makedirs(generated_images_path, exist_ok=True)
#         os.makedirs("generated_videos", exist_ok=True)

#         # Get the chat_message_id from query parameters if provided
#         chat_message_id = request.query_params.get('chat_message_id')
#         if chat_message_id:
#             try:
#                 chat_message_id = int(chat_message_id)
#             except ValueError:
#                 logger.error("Invalid chat_message_id provided.")
#                 raise HTTPException(status_code=400, detail="Invalid chat_message_id.")
#         else:
#             logger.warning("No chat_message_id provided. The video will be generated without linking to a specific message.")

#         # Add background task for video processing
#         background_tasks.add_task(
#             process_video_creation,
#             keyframe_descriptions,
#             video.video_length,
#             generated_images_path,
#             assembled_video_path,
#             final_video_path,
#             current_user_id,
#             video.movable_elements,
#             chat_message_id=chat_message_id  # Pass the chat_message_id
#         )
#         logger.info("Video processing task added to background.")

#         return {"message": "Video processing started, check back later"}

#     except HTTPException as http_err:
#         logger.error(f"HTTP error occurred: {http_err.detail}")
#         raise http_err
#     except Exception as e:
#         logger.error(f"Unexpected error occurred: {e}")
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the video")

# # Background task to process video creation
# def process_video_creation(
#     keyframe_descriptions: list,
#     video_length: int,
#     generated_images_path: str,
#     assembled_video_path: str,
#     final_video_path: str,
#     current_user_id: str,
#     movable_elements: list,  # List of MovableElement
#     chat_message_id: int = None  # New parameter
# ):
#     try:
#         logger.info("Starting video creation process.")

#         # Create a new database session
#         db = SessionLocal()

#         # Step 1: Determine number of images based on actual visuals
#         num_images = len(keyframe_descriptions)
#         logger.info(f"Number of images to generate: {num_images}")

#         # Step 2: Generate images
#         images = generate_images(keyframe_descriptions, num_images)
#         save_images(images, generated_images_path)

#         # Step 3: Generate audio narration
#         audio_path = os.path.join(generated_images_path, "audio.mp3")
#         # Combine all descriptions into a single text for narration
#         narration_text = " ".join([desc.split(":", 1)[1].strip() for desc in keyframe_descriptions])
#         generate_audio(narration_text, audio_path)

#         # Step 4: Calculate audio duration
#         audio = AudioFileClip(audio_path)
#         audio_duration = audio.duration
#         logger.info(f"Narration audio duration: {audio_duration} seconds")

#         # Step 5: Generate video frames based on audio duration
#         total_duration = int(audio_duration)  # Round down to nearest second
#         fps = 24  # Increased FPS for smoother animation
#         frames = generate_video_frames_from_images(images, duration=total_duration, fps=fps)
#         temp_video_path = os.path.join(generated_images_path, "temp_video.mp4")
#         save_video(frames, temp_video_path, fps=fps)

#         # Step 6: Assemble video with audio
#         assemble_video_with_audio(temp_video_path, audio_path, assembled_video_path)

#         # Step 7: Add moving elements overlays
#         if movable_elements:
#             # Determine a unique final video path
#             unique_final_video_path = get_unique_file_path(final_video_path)
#             add_moving_elements(
#                 base_video_path=assembled_video_path,
#                 movable_elements=movable_elements,
#                 output_video_path=unique_final_video_path
#             )
#             final_video_path = unique_final_video_path  # Update to the unique path
#         else:
#             # If no movable elements, determine a unique final video path
#             unique_final_video_path = get_unique_file_path(final_video_path)
#             # Rename the assembled video to the unique final video path
#             os.rename(assembled_video_path, unique_final_video_path)
#             logger.info(f"No movable elements to add. Final video saved at {unique_final_video_path}")
#             final_video_path = unique_final_video_path  # Update to the unique path

#         # Step 8: Clean up temporary files
#         if os.path.exists(temp_video_path):
#             os.remove(temp_video_path)
#             logger.info(f"Temporary video file removed: {temp_video_path}")
#         if os.path.exists(assembled_video_path) and assembled_video_path != final_video_path:
#             os.remove(assembled_video_path)
#             logger.info(f"Assembled video file removed: {assembled_video_path}")

#         # Step 9: Save video record to the database
#         new_video = Video(
#             user_id=current_user_id,
#             input_text=" ".join(keyframe_descriptions),
#             video_length=total_duration / 60,  # Store duration in minutes
#             file_path=final_video_path,
#             chat_message_id=chat_message_id  # Link to chat message
#         )
#         db.add(new_video)
#         db.commit()
#         db.refresh(new_video)  # To get the new video id

#         logger.info(f"Video record created in DB with file path: {final_video_path}")

#         # Capture the video ID before closing the session
#         video_id = new_video.id

#         logger.info("Video creation process completed successfully.")

#         # If chat_message_id is provided, update the processing message with video_url
#         if chat_message_id:
#             # Fetch the system message
#             system_message = db.query(ChatMessage).filter_by(id=chat_message_id, user_id=current_user_id).first()
#             if system_message:
#                 # Update the system message with the video_url and content
#                 system_message.content = "Here is your generated video."
#                 system_message.video_id = video_id  # Assuming ChatMessage has video_id FK
#                 db.commit()
#                 logger.info(f"Updated system message {chat_message_id} with video_url.")

#         # Close the session **after** accessing video_id
#         db.close()

#         # Since background tasks don't utilize return values, no need to return anything
#         # If you need to perform further actions with video_id, handle them here
#         return  # Exit the function without returning

#     except HTTPException as http_err:
#         logger.error(f"HTTP error occurred during video processing: {http_err.detail}")
#         raise http_err
#     except Exception as e:
#         logger.error(f"Error in video processing: {e}")
#         logger.error(traceback.format_exc())
#         # Optionally, add a system message indicating an error occurred
#         if chat_message_id:
#             try:
#                 error_message = ChatMessage(
#                     user_id=current_user_id,
#                     content="An error occurred while generating your video.",
#                     is_user=False,
#                     timestamp=datetime.utcnow()
#                 )
#                 db.add(error_message)
#                 db.commit()
#                 logger.info(f"Added error message to chat message {chat_message_id}.")
#             except Exception as inner_e:
#                 logger.error(f"Failed to add error message to chat: {inner_e}")
#         raise HTTPException(status_code=500, detail="Failed to generate video.")

# # API endpoint to download the generated video
# @router.get("/download/{video_id}", tags=["Video"], summary="Download video")
# def download_video(video_id: int, db: Session = Depends(get_db), Authorize: AuthJWT = Depends(jwt_auth_dependency)):
#     try:
#         current_user_id = Authorize.get_jwt_subject()
#         video = db.query(Video).filter_by(id=video_id, user_id=current_user_id).first()

#         if not video:
#             raise HTTPException(status_code=404, detail="Video not found")

#         file_path = video.file_path

#         if not os.path.exists(file_path):
#             raise HTTPException(status_code=404, detail="File not found")

#         return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type='video/mp4')

#     except Exception as e:
#         logger.error(f"Error in downloading video: {e}")
#         raise HTTPException(status_code=500, detail="Failed to download video")

# # API endpoint to retrieve a specific video frame
# @router.get("/get_frame/{video_id}/{frame_number}", tags=["Video"], summary="Get a specific video frame")
# def get_video_frame(
#     video_id: int,
#     frame_number: int,
#     db: Session = Depends(get_db),
#     Authorize: AuthJWT = Depends(jwt_auth_dependency)
# ):
#     try:
#         current_user_id = Authorize.get_jwt_subject()
#         video = db.query(Video).filter_by(id=video_id, user_id=current_user_id).first()

#         if not video:
#             raise HTTPException(status_code=404, detail="Video not found")

#         video_path = video.file_path

#         if not os.path.exists(video_path):
#             logger.error(f"Video file not found: {video_path}")
#             raise HTTPException(status_code=404, detail="Video file not found")

#         # Use MoviePy to extract the frame
#         clip = VideoFileClip(video_path)
#         duration = clip.duration
#         fps = clip.fps
#         total_frames = int(duration * fps)

#         if frame_number < 0 or frame_number >= total_frames:
#             raise HTTPException(status_code=400, detail="Frame number out of range")

#         # Get the time corresponding to the frame number
#         frame_time = frame_number / fps

#         # Get the frame as an image
#         frame = clip.get_frame(frame_time)
#         clip.reader.close()
#         if clip.audio:
#             clip.audio.reader.close_proc()

#         # Convert the frame (numpy array) to an image
#         img = Image.fromarray(frame)

#         # Save the image to a BytesIO stream
#         img_io = io.BytesIO()
#         img.save(img_io, 'JPEG')
#         img_io.seek(0)

#         return StreamingResponse(img_io, media_type="image/jpeg")

#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         logger.error(f"Error retrieving frame {frame_number} from video {video_id}: {e}")
#         raise HTTPException(status_code=500, detail="Failed to retrieve video frame")

# routes/video_routes.py

import os
import logging
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from sqlalchemy.orm import Session
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import InvalidHeaderError
from fastapi.responses import FileResponse, StreamingResponse
from database import get_db, SessionLocal
from models import Video, ChatMessage  # Import ChatMessage
from pydantic import BaseModel, validator
import traceback
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip, CompositeVideoClip, ImageClip
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import io
import requests
import time
import re
from typing import List, Optional, Callable
from datetime import datetime

# Import the sanitization function
from .utils import sanitize_filename  # Ensure utils.py is in the same directory

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set.")
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=OPENAI_API_KEY)

router = APIRouter()

# Helper function to get unique file path
def get_unique_file_path(file_path: str) -> str:
    if not os.path.exists(file_path):
        return file_path

    base, extension = os.path.splitext(file_path)
    counter = 1
    new_file_path = f"{base}_{counter}{extension}"

    while os.path.exists(new_file_path):
        counter += 1
        new_file_path = f"{base}_{counter}{extension}"
        if counter > 1000:
            raise Exception("Too many files with the same name exist.")

    return new_file_path

# Pydantic model for movable elements
class MovableElement(BaseModel):
    element_path: str
    element_type: str
    movement: str = 'linear'
    position: Optional[str] = 'center'
    size: Optional[tuple] = None
    radius: Optional[int] = None
    center_x: Optional[int] = None
    center_y: Optional[int] = None
    custom_path: Optional[Callable[[float], tuple]] = None
    y_position: Optional[int] = 50

# Pydantic model for video creation
class VideoCreate(BaseModel):
    input_text: str
    video_length: int  # in minutes
    movable_elements: Optional[List[MovableElement]] = []

    @validator('video_length')
    def video_length_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Video length must be positive')
        return v

    @validator('input_text')
    def validate_input_text(cls, v):
        if re.search(r'[<>:"/\\|?*]', v):
            raise ValueError("Input text contains invalid characters for filenames.")
        return v

# Function to sanitize the prompt
def sanitize_prompt(prompt: str) -> str:
    sanitized = re.sub(r'[^\w\s.,!?]', '', prompt)
    sanitized = re.sub(r'\s+', ' ', sanitized)
    return sanitized.strip()

# Function to generate keyframe descriptions with retry logic
def generate_keyframe_description(input_text: str, num_visuals: int = 5, max_retries: int = 3) -> list:
    attempt = 0
    backoff = 2  # Seconds
    while attempt < max_retries:
        try:
            prompt = (
                f"Provide {num_visuals} detailed and distinct visual descriptions for the following video theme. "
                f"Each description should start with 'Visual X:', where X is the visual number (e.g., 'Visual 1:').\n"
                f"Theme: {input_text}"
            )

            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a creative assistant specializing in visual storytelling."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800
            )
            description = completion.choices[0].message.content.strip()
            logger.info(f"Generated keyframe description: {description}")

            # Use regex to extract all Visual X: descriptions
            pattern = rf"Visual\s+\d+:\s*(.*?)\n(?=Visual\s+\d+:|$)"
            visuals = re.findall(pattern, description, re.DOTALL)

            # Prepend "Visual X:" to each visual
            visuals = [f"Visual {i + 1}: {visual.strip()}" for i, visual in enumerate(visuals)]

            if len(visuals) < num_visuals:
                logger.warning(f"Expected {num_visuals} visuals, but got {len(visuals)}.")

            logger.info(f"Successfully generated {len(visuals)} visuals.")
            return visuals
        except Exception as e:
            attempt += 1
            logger.error(f"Attempt {attempt} - Error in generating keyframe description: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
            else:
                logger.error("Max retries reached. Unable to generate keyframe descriptions.")
                raise HTTPException(status_code=500, detail=f"Failed to generate keyframe description: {str(e)}")

# Function to generate images based on descriptions using OpenAI's DALL·E with error handling
def generate_images(descriptions: list, num_images: int = 5) -> list:
    try:
        images = []
        for i, description in enumerate(descriptions):
            if i >= num_images:
                break  # Limit to the specified number of images

            logger.info(f"Generating image {i + 1} of {num_images} with description: {description}")

            # Sanitize the description
            sanitized_description = sanitize_prompt(description)
            logger.debug(f"Sanitized description for image {i + 1}: {sanitized_description}")

            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=sanitized_description,
                    n=1,
                    size="1024x1024"
                )
                image_url = response.data[0].url
                logger.debug(f"Image URL for image {i + 1}: {image_url}")

                img_response = requests.get(image_url)
                if img_response.status_code != 200:
                    logger.error(f"Failed to fetch image {i + 1} from URL: {image_url}")
                    continue
                img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
                images.append(img)
                logger.info(f"Generated and fetched image {i + 1}.")
                time.sleep(1)  # Sleep to respect rate limits

            except Exception as e:
                logger.error(f"Error during image generation for image {i + 1}: {e}")
                logger.error(f"Prompt causing error: {sanitized_description}")
                continue

        if not images:
            raise HTTPException(status_code=500, detail="No images were generated.")

        logger.info(f"Successfully generated {len(images)} images.")
        return images
    except Exception as e:
        logger.error(f"Error in generating images: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate images: {str(e)}")

# Function to save PIL Image to disk
def save_images(images: list, base_path: str):
    try:
        os.makedirs(base_path, exist_ok=True)
        for idx, img in enumerate(images):
            img_path = os.path.join(base_path, f"image_{idx + 1}.png")
            img.save(img_path)
            logger.info(f"Saved image {idx + 1} at {img_path}")
    except Exception as e:
        logger.error(f"Error in saving images: {e}")
        raise HTTPException(status_code=500, detail="Failed to save images")

# Function to generate video frames based on descriptions using images
def generate_video_frames_from_images(images: list, duration: int = 300, fps: int = 24) -> list:
    try:
        processed_images = []
        for img in images:
            img = img.resize((640, 480))  # Resize to desired video resolution
            processed_images.append(np.array(img))

        total_frames = duration * fps
        frames = []
        num_images = len(processed_images)
        frames_per_image = total_frames // num_images if num_images else 1

        for img in processed_images:
            for _ in range(frames_per_image):
                frames.append(img)

        # If there are remaining frames, add them to the last image
        remaining_frames = duration * fps - frames_per_image * num_images
        for _ in range(remaining_frames):
            frames.append(processed_images[-1] if processed_images else np.zeros((480, 640, 3), dtype=np.uint8))

        logger.info("Generated video frames from images.")
        return frames
    except Exception as e:
        logger.error(f"Error in generating video frames from images: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate video frames from images")

# Function to save video frames as a video file using moviepy
def save_video(frames: list, path: str, fps: int = 24):
    try:
        logger.info(f"Saving video frames to {path} with {fps} FPS.")
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(path, codec='libx264', audio=False)
        logger.info(f"Video saved at path: {path}")
    except Exception as e:
        logger.error(f"Error in saving video: {e}")
        raise HTTPException(status_code=500, detail="Failed to save video")

# Function to generate speech using OpenAI's audio.speech.create
def generate_audio(text: str, audio_path: str):
    try:
        logger.info(f"Generating audio narration and saving to {audio_path}.")
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        response.stream_to_file(audio_path)
        logger.info(f"Audio saved at path: {audio_path}")
    except Exception as e:
        logger.error(f"Error in generating audio: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate audio")

# Function to assemble video with audio
def assemble_video_with_audio(video_path: str, audio_path: str, output_video_path: str):
    try:
        logger.info(f"Assembling video from {video_path} and audio from {audio_path} into {output_video_path}.")
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)

        # Log original durations
        logger.info(f"Original video duration: {video.duration} seconds")
        logger.info(f"Original audio duration: {audio.duration} seconds")

        # Ensure the audio duration matches the video duration
        audio = audio.set_duration(video.duration)
        logger.info(f"Trimmed audio duration: {audio.duration} seconds")

        video = video.set_audio(audio)
        video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
        logger.info(f"Final video assembled successfully at path: {output_video_path}")
    except Exception as e:
        logger.error(f"Error in assembling video with audio: {e}")
        raise HTTPException(status_code=500, detail="Failed to assemble video with audio")

# Function to add moving elements overlays
def add_moving_elements(
    base_video_path: str,
    movable_elements: list,  # List of MovableElement
    output_video_path: str
):
    try:
        # Load the base video
        video = VideoFileClip(base_video_path)
        duration = video.duration

        # List to hold all overlay clips
        overlays = []

        for element in movable_elements:
            if element.element_type == 'image':
                overlay = ImageClip(element.element_path, transparent=True).set_duration(duration)
            elif element.element_type == 'video':
                overlay = VideoFileClip(element.element_path, has_mask=True).set_duration(duration)
            else:
                logger.warning(f"Invalid element_type '{element.element_type}' for element '{element.element_path}'. Skipping.")
                continue

            # Resize if size is specified
            if element.size:
                overlay = overlay.resize(newsize=element.size)

            # Define the movement path
            if element.movement == 'linear':
                # Move from left to right
                overlay = overlay.set_position(lambda t: (int(640 * (t / duration)) - overlay.w / 2, element.y_position))
            elif element.movement == 'circular':
                # Circular movement around a point
                radius = element.radius if element.radius else 100
                center_x = element.center_x if element.center_x else 320
                center_y = element.center_y if element.center_y else 240

                overlay = overlay.set_position(lambda t: (
                    int(center_x + radius * np.cos(2 * np.pi * t / duration)) - overlay.w / 2,
                    int(center_y + radius * np.sin(2 * np.pi * t / duration)) - overlay.h / 2
                ))
            elif element.movement == 'path' and element.custom_path:
                # Custom path defined by a function
                overlay = overlay.set_position(element.custom_path)
            else:
                # Static position or unrecognized movement type
                overlay = overlay.set_position(element.position)

            overlays.append(overlay)

        if overlays:
            # Composite all overlays onto the video
            final = CompositeVideoClip([video] + overlays)

            # Write the final video
            final.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
            logger.info(f"Final video with moving elements saved at {output_video_path}")
        else:
            # If no valid overlays, simply copy the base video to output
            os.rename(base_video_path, output_video_path)
            logger.info(f"No movable elements provided. Base video saved as {output_video_path}")

    except Exception as e:
        logger.error(f"Error in adding moving elements: {e}")
        raise HTTPException(status_code=500, detail="Failed to add moving elements to video")

# Authentication dependency using JWT
def jwt_auth_dependency(Authorize: AuthJWT = Depends()):
    try:
        Authorize.jwt_required()
    except InvalidHeaderError as e:
        logger.error(f"InvalidHeaderError: {e}")
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    return Authorize

# API endpoint to generate a new video
@router.post("/generate", tags=["Video"], summary="Generate a new video", response_model=dict)
async def generate_video(
    request: Request,
    video: VideoCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends(jwt_auth_dependency)
):
    try:
        logger.info(f"Incoming headers: {request.headers}")

        current_user_id = Authorize.get_jwt_subject()
        logger.info(f"User ID: {current_user_id}")

        if video.video_length > 15:
            logger.error("Video length exceeds maximum limit")
            raise HTTPException(status_code=400, detail="Video length cannot exceed 15 minutes")

        # Generate keyframe descriptions with reduced number of visuals
        keyframe_descriptions = generate_keyframe_description(video.input_text, num_visuals=5)  # Reduced from 20 to 5
        logger.info(f"Keyframe descriptions: {keyframe_descriptions}")

        # Define directories with sanitized input
        sanitized_input = sanitize_filename(video.input_text.strip())
        logger.info(f"Sanitized input for directory: {sanitized_input}")

        generated_images_path = os.path.join("generated_images", sanitized_input)
        assembled_video_path = os.path.join("generated_videos", f"{sanitized_input}_assembled.mp4")
        final_video_path = os.path.join("generated_videos", f"{sanitized_input}_final.mp4")

        # Ensure directories exist
        os.makedirs(generated_images_path, exist_ok=True)
        os.makedirs("generated_videos", exist_ok=True)

        # Get the chat_message_id from query parameters if provided
        chat_message_id = request.query_params.get('chat_message_id')
        if chat_message_id:
            try:
                chat_message_id = int(chat_message_id)
            except ValueError:
                logger.error("Invalid chat_message_id provided.")
                raise HTTPException(status_code=400, detail="Invalid chat_message_id.")
        else:
            logger.warning("No chat_message_id provided. The video will be generated without linking to a specific message.")

        # Add background task for video processing
        background_tasks.add_task(
            process_video_creation,
            keyframe_descriptions,
            video.video_length,
            generated_images_path,
            assembled_video_path,
            final_video_path,
            current_user_id,
            video.movable_elements,
            chat_message_id=chat_message_id  # Pass the chat_message_id
        )
        logger.info("Video processing task added to background.")

        return {"message": "Video processing started, check back later"}

    except HTTPException as http_err:
        logger.error(f"HTTP error occurred: {http_err.detail}")
        raise http_err
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the video")


def process_video_creation(
    keyframe_descriptions: list,
    video_length: int,
    generated_images_path: str,
    assembled_video_path: str,
    final_video_path: str,
    current_user_id: str,
    movable_elements: list,  # List of MovableElement
    chat_message_id: int = None  # New parameter
):
    try:
        logger.info("Starting video creation process.")

        # Create a new database session
        db = SessionLocal()

        # Step 1: Determine number of images based on actual visuals
        num_images = len(keyframe_descriptions)
        logger.info(f"Number of images to generate: {num_images}")

        # Step 2: Generate images
        images = generate_images(keyframe_descriptions, num_images)
        save_images(images, generated_images_path)

        # Step 3: Generate audio narration
        audio_path = os.path.join(generated_images_path, "audio.mp3")
        # Combine all descriptions into a single text for narration
        narration_text = " ".join([desc.split(":", 1)[1].strip() for desc in keyframe_descriptions])
        generate_audio(narration_text, audio_path)

        # Step 4: Calculate audio duration
        audio = AudioFileClip(audio_path)
        audio_duration = audio.duration
        logger.info(f"Narration audio duration: {audio_duration} seconds")

        # Step 5: Generate video frames based on audio duration
        total_duration = int(audio_duration)  # Round down to nearest second
        fps = 24  # Increased FPS for smoother animation
        frames = generate_video_frames_from_images(images, duration=total_duration, fps=fps)
        temp_video_path = os.path.join(generated_images_path, "temp_video.mp4")
        save_video(frames, temp_video_path, fps=fps)

        # Step 6: Assemble video with audio
        assemble_video_with_audio(temp_video_path, audio_path, assembled_video_path)

        # Step 7: Add moving elements overlays
        if movable_elements:
            # Determine a unique final video path
            unique_final_video_path = get_unique_file_path(final_video_path)
            add_moving_elements(
                base_video_path=assembled_video_path,
                movable_elements=movable_elements,
                output_video_path=unique_final_video_path
            )
            final_video_path = unique_final_video_path  # Update to the unique path
        else:
            # If no movable elements, determine a unique final video path
            unique_final_video_path = get_unique_file_path(final_video_path)
            # Rename the assembled video to the unique final video path
            os.rename(assembled_video_path, unique_final_video_path)
            logger.info(f"No movable elements to add. Final video saved at {unique_final_video_path}")
            final_video_path = unique_final_video_path  # Update to the unique path

        # Step 8: Clean up temporary files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            logger.info(f"Temporary video file removed: {temp_video_path}")
        if os.path.exists(assembled_video_path) and assembled_video_path != final_video_path:
            os.remove(assembled_video_path)
            logger.info(f"Assembled video file removed: {assembled_video_path}")

        # Step 9: Save video record to the database
        new_video = Video(
            user_id=current_user_id,
            input_text=" ".join(keyframe_descriptions),
            video_length=total_duration / 60,  # Store duration in minutes
            file_path=final_video_path,
            chat_message_id=chat_message_id  # Link to chat message
        )
        db.add(new_video)
        db.commit()
        db.refresh(new_video)  # To get the new video id

        logger.info(f"Video record created in DB with file path: {final_video_path}")

        # Capture the video ID before closing the session
        video_id = new_video.id

        logger.info("Video creation process completed successfully.")

        # If chat_message_id is provided, update the processing message with video_url
        if chat_message_id:
            # Fetch the system message
            system_message = db.query(ChatMessage).filter_by(id=chat_message_id, user_id=current_user_id).first()
            if system_message:
                # Update the system message with the video_url and content
                system_message.content = "Here is your generated video."
                system_message.video_id = video_id  # Assuming ChatMessage has video_id FK
                db.commit()
                logger.info(f"Updated system message {chat_message_id} with video_url.")

        # Close the session **after** accessing video_id
        db.close()

        # Since background tasks don't utilize return values, no need to return anything
        # If you need to perform further actions with video_id, handle them here
        return  # Exit the function without returning

    except HTTPException as http_err:
        logger.error(f"HTTP error occurred during video processing: {http_err.detail}")
        raise http_err
    except Exception as e:
        logger.error(f"Error in video processing: {e}")
        logger.error(traceback.format_exc())
        # Optionally, add a system message indicating an error occurred
        if chat_message_id:
            try:
                error_message = ChatMessage(
                    user_id=current_user_id,
                    content="An error occurred while generating your video.",
                    is_user=False,
                    timestamp=datetime.utcnow()
                )
                db.add(error_message)
                db.commit()
                logger.info(f"Added error message to chat message {chat_message_id}.")
            except Exception as inner_e:
                logger.error(f"Failed to add error message to chat: {inner_e}")
        raise HTTPException(status_code=500, detail="Failed to generate video.")
    

# API endpoint to download the generated video
@router.get("/download/{video_id}", tags=["Video"], summary="Download video")
def download_video(video_id: int, db: Session = Depends(get_db), Authorize: AuthJWT = Depends(jwt_auth_dependency)):
    try:
        current_user_id = Authorize.get_jwt_subject()
        video = db.query(Video).filter_by(id=video_id, user_id=current_user_id).first()

        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        file_path = video.file_path

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type='video/mp4')

    except Exception as e:
        logger.error(f"Error in downloading video: {e}")
        raise HTTPException(status_code=500, detail="Failed to download video")

# API endpoint to retrieve a specific video frame
@router.get("/get_frame/{video_id}/{frame_number}", tags=["Video"], summary="Get a specific video frame")
def get_video_frame(
    video_id: int,
    frame_number: int,
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends(jwt_auth_dependency)
):
    try:
        current_user_id = Authorize.get_jwt_subject()
        video = db.query(Video).filter_by(id=video_id, user_id=current_user_id).first()

        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        video_path = video.file_path

        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            raise HTTPException(status_code=404, detail="Video file not found")

        # Use MoviePy to extract the frame
        clip = VideoFileClip(video_path)
        duration = clip.duration
        fps = clip.fps
        total_frames = int(duration * fps)

        if frame_number < 0 or frame_number >= total_frames:
            raise HTTPException(status_code=400, detail="Frame number out of range")

        # Get the time corresponding to the frame number
        frame_time = frame_number / fps

        # Get the frame as an image
        frame = clip.get_frame(frame_time)
        clip.reader.close()
        if clip.audio:
            clip.audio.reader.close_proc()

        # Convert the frame (numpy array) to an image
        img = Image.fromarray(frame)

        # Save the image to a BytesIO stream
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG')
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/jpeg")

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving frame {frame_number} from video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve video frame")
