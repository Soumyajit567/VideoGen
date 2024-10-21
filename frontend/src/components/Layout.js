import React from 'react';
import { AppBar, Toolbar, Typography, Container } from '@mui/material';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';

const Layout = ({ children }) => {
  return (
    <>
      {/* AppBar with VideoGen Title */}
      <AppBar position="static">
        <Toolbar>
          <VideoLibraryIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontFamily: "'Roboto', sans-serif" }}>
            VideoGen
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
        {children}
      </Container>
    </>
  );
};

export default Layout;
