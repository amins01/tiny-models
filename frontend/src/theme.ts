import { createTheme } from "@mui/material/styles";

export const getTheme = (mode: "light" | "dark") =>
  createTheme({
    typography: {
      fontFamily:
        "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      button: { textTransform: "none", fontWeight: 500 },
    },
    palette: {
      mode,
      primary: { main: mode === "dark" ? "#CFCFCF" : "#111111" },
      background: {
        default: mode === "dark" ? "#0b0b0c" : "#ffffff",
        paper: mode === "dark" ? "#121213" : "#ffffff",
      },
      text: {
        primary: mode === "dark" ? "#E6E6E6" : "#111111",
        secondary: mode === "dark" ? "rgba(230,230,230,0.7)" : "#666666",
      },
      divider: mode === "dark" ? "rgba(230,230,230,0.06)" : "rgba(0,0,0,0.08)",
    },
    shape: { borderRadius: 8 },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            backgroundColor: mode === "dark" ? "#0b0b0c" : "#ffffff",
            color: mode === "dark" ? "#E6E6E6" : "#111111",
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            boxShadow: "none",
            borderRadius: "6px",
            "&:hover": { boxShadow: "none" },
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundColor: mode === "dark" ? "#121213" : "#ffffff",
            border: `1px solid ${mode === "dark" ? "rgba(230,230,230,0.04)" : "#EAEAEA"}`,
            boxShadow: "none",
          },
        },
      },
    },
  });