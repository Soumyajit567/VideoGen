import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom'; // Updated import
import Login from './components/Login';
import Register from './components/Register';
import Chat from './components/Chat';
import Layout from './components/Layout'; // Import Layout

function App() {
  const [token, setToken] = useState(localStorage.getItem('token') || '');

  // Optional: Function to handle token updates
  const handleSetToken = (newToken) => {
    setToken(newToken);
    localStorage.setItem('token', newToken); // Persist token
  };

  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/register" element={<Register />} />
          <Route 
            path="/chat" 
            element={token ? <Chat token={token} /> : <Navigate to="/" replace />} 
          />
          <Route path="/" element={<Login setToken={handleSetToken} />} />
          {/* Redirect any unknown routes to Login */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;

