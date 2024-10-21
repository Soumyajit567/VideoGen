import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Box,
  Button,
  Container,
  TextField,
  Typography,
  Paper,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  Alert,
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';

const Chat = ({ token }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [videoBlobs, setVideoBlobs] = useState({});
  const [error, setError] = useState('');

  // Fetch conversation history
  const fetchMessages = useCallback(async () => {
    try {
      const response = await axios.get('http://127.0.0.1:8001/api/chat/get_conversation', {
        headers: { Authorization: `Bearer ${token}` },
      });
      setMessages(response.data.conversation);
    } catch (err) {
      console.error('Error fetching conversation:', err);
      setError('Failed to fetch conversation. Please try again.');
    }
  }, [token]);

  useEffect(() => {
    fetchMessages(); // Initial fetch
    const interval = setInterval(fetchMessages, 5000); // Poll every 5 seconds for updates
    return () => clearInterval(interval); // Cleanup on unmount
  }, [fetchMessages]);

  // Handle sending a new message
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (input.trim() === '') return; // Prevent sending empty messages
    try {
      setIsGenerating(true);
      await axios.post(
        'http://127.0.0.1:8001/api/chat/send_message',
        { content: input },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setInput('');
      fetchMessages(); // Fetch updated messages
    } catch (err) {
      console.error('Error sending message:', err);
      setError('Failed to send message. Please try again.');
      setIsGenerating(false);
    }
  };

  // Check if any video generation is still pending
  const checkIfVideoIsBeingGenerated = useCallback(() => {
    return messages.some(
      (msg) => !msg.video_url && !msg.is_user // If no video_url and not a user message, it's being processed
    );
  }, [messages]);

  useEffect(() => {
    const generating = checkIfVideoIsBeingGenerated();
    setIsGenerating(generating);
  }, [messages, checkIfVideoIsBeingGenerated]);

  // Function to handle authenticated video download
  const handleDownload = async (video_url, video_id) => {
    try {
      setDownloading(true);
      const response = await axios.get(video_url, {
        headers: { Authorization: `Bearer ${token}` },
        responseType: 'blob',
      });

      if (response.status === 200 && response.data.type === 'video/mp4') {
        const blobUrl = window.URL.createObjectURL(response.data);
        const link = document.createElement('a');
        link.href = blobUrl;

        const filename = `video_${video_id}.mp4`;
        link.setAttribute('download', filename);

        document.body.appendChild(link);
        link.click();

        document.body.removeChild(link);
        window.URL.revokeObjectURL(blobUrl);
      } else {
        alert('Failed to download video. Please try again.');
      }

      setDownloading(false);
    } catch (err) {
      console.error('Error downloading video:', err);
      setDownloading(false);
      if (err.response && err.response.status === 401) {
        alert('Unauthorized access. Please log in again.');
      } else {
        alert('Failed to download video. Please try again.');
      }
    }
  };

  // Function to fetch video Blob URLs
  const fetchVideoBlobs = useCallback(async () => {
    const newBlobs = {};
    for (const msg of messages) {
      if (msg.video_url && !videoBlobs[msg.video_id]) {
        try {
          const response = await axios.get(msg.video_url, {
            headers: { Authorization: `Bearer ${token}` },
            responseType: 'blob',
          });

          if (response.status === 200 && response.data.type === 'video/mp4') {
            const blobUrl = window.URL.createObjectURL(response.data);
            newBlobs[msg.video_id] = blobUrl;
          } else {
            console.error(`Failed to fetch video Blob for video_id: ${msg.video_id}`);
          }
        } catch (err) {
          console.error(`Error fetching video Blob for video_id: ${msg.video_id}`, err);
        }
      }
    }

    if (Object.keys(newBlobs).length > 0) {
      setVideoBlobs((prevBlobs) => ({ ...prevBlobs, ...newBlobs }));
    }
  }, [messages, token, videoBlobs]);

  useEffect(() => {
    fetchVideoBlobs();
  }, [fetchVideoBlobs]);

  useEffect(() => {
    // Cleanup Blob URLs on component unmount
    return () => {
      for (const blobUrl of Object.values(videoBlobs)) {
        window.URL.revokeObjectURL(blobUrl);
      }
    };
  }, [videoBlobs]);

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Chat Window */}
      <Paper elevation={3} sx={{ p: 2, maxHeight: '60vh', overflow: 'auto' }}>
        <List>
          {messages.map((msg, index) => (
            <ListItem
              key={index}
              alignItems="flex-start"
              sx={{
                backgroundColor: msg.is_user ? '#e3f2fd' : '#f1f1f1',
                borderRadius: 2,
                mb: 1,
              }}
            >
              <ListItemText
                primary={
                  <Typography
                    variant="body1"
                    sx={{ fontFamily: "'Roboto', sans-serif", whiteSpace: 'pre-wrap' }}
                  >
                    {msg.content}
                  </Typography>
                }
                secondary={
                  <>
                    {msg.video_url ? (
                      <Box sx={{ mt: 1 }}>
                        {videoBlobs[msg.video_id] ? (
                          <video width="320" height="240" controls>
                            <source src={videoBlobs[msg.video_id]} type="video/mp4" />
                            Your browser does not support the video tag.
                          </video>
                        ) : (
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <CircularProgress size={20} sx={{ mr: 1 }} />
                            <Typography variant="body2">Loading video...</Typography>
                          </Box>
                        )}
                        <Button
                          variant="contained"
                          color="primary"
                          startIcon={<DownloadIcon />}
                          sx={{ mt: 1 }}
                          onClick={() => handleDownload(msg.video_url, msg.video_id)}
                          disabled={downloading}
                        >
                          {downloading ? 'Downloading...' : 'Download Video'}
                        </Button>
                      </Box>
                    ) : msg.is_user ? null : (
                      <Typography variant="body2" color="text.secondary">
                        Video is being processed...
                      </Typography>
                    )}
                  </>
                }
              />
            </ListItem>
          ))}
        </List>
      </Paper>

      {/* Message Input */}
      <Box component="form" onSubmit={handleSendMessage} sx={{ mt: 2, display: 'flex', gap: 2 }}>
        <TextField
          fullWidth
          variant="outlined"
          label="Enter your prompt"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <Button
          type="submit"
          variant="contained"
          color="secondary"
          disabled={isGenerating}
          sx={{ px: 4 }}
        >
          {isGenerating ? <CircularProgress size={24} color="inherit" /> : 'Send'}
        </Button>
      </Box>

      {/* Generating Video Indicator */}
      {isGenerating && (
        <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <CircularProgress size={24} />
          <Typography variant="body1">Generating video... Please wait.</Typography>
        </Box>
      )}

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
    </Box>
  );
};

export default Chat;
