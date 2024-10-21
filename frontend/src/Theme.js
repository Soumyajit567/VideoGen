import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2', 
    },
    secondary: {
      main: '#ff4081', 
    },
  },
  typography: {
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      fontFamily: "'Roboto', sans-serif",
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      fontFamily: "'Roboto', sans-serif",
    },
    button: {
      textTransform: 'none', 
    },
  },
});

export default theme;
