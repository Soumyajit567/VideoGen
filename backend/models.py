from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, Boolean
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    videos = relationship("Video", back_populates="owner")
    chats = relationship("ChatMessage", back_populates="user")

class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    input_text = Column(Text, nullable=False)
    video_length = Column(Integer, nullable=False)
    file_path = Column(String, nullable=False)
    owner = relationship("User", back_populates="videos")
    chat_message_id = Column(Integer, ForeignKey('chat_messages.id', ondelete='CASCADE'), nullable=True)
    chat_message = relationship("ChatMessage", back_populates="video")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    content = Column(Text, nullable=False)
    is_user = Column(Boolean, default=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    video = relationship("Video", uselist=False, back_populates="chat_message")
    user = relationship("User", back_populates="chats")
