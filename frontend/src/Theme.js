// // src/Theme.js

// import { createTheme } from '@mui/material/styles';

// const theme = createTheme({
//   palette: {
//     primary: {
//       main: '#1976d2', // Blue
//     },
//     secondary: {
//       main: '#ff4081', // Pink
//     },
//   },
//   typography: {
//     h1: {
//       fontSize: '2.5rem',
//       fontWeight: 700,
//       fontFamily: "'Roboto', sans-serif",
//     },
//     h2: {
//       fontSize: '2rem',
//       fontWeight: 600,
//       fontFamily: "'Roboto', sans-serif",
//     },
//     button: {
//       textTransform: 'none', // Prevent uppercase transformation
//     },
//   },
// });

// export default theme;

// src/Theme.js

import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2', // Blue
    },
    secondary: {
      main: '#ff4081', // Pink
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
      textTransform: 'none', // Prevent uppercase transformation
    },
  },
});

export default theme;
