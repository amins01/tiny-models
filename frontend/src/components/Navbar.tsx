import { Link, useLocation } from "react-router-dom";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Container from "@mui/material/Container";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import { useTheme, alpha } from "@mui/material/styles";
import LightModeIcon from "@mui/icons-material/LightMode";
import DarkModeIcon from "@mui/icons-material/DarkMode";

import { PAGES } from "../constants";

type Props = {
  themeMode?: "light" | "dark";
  toggleTheme?: () => void;
};

export default function Navbar({ themeMode = "dark", toggleTheme }: Props) {
  const theme = useTheme();
  const location = useLocation();
  const bg = alpha(theme.palette.background.paper, theme.palette.mode === "dark" ? 0.28 : 0.97);

  return (
    <AppBar
      position="sticky"
      elevation={0}
      sx={{
        backgroundColor: bg,
        backdropFilter: "blur(10px)",
        borderBottom: `1px solid ${theme.palette.divider}`,
      }}
    >
      <Container maxWidth="md">
        <Toolbar disableGutters sx={{ gap: 2, minHeight: 64 }}>
          <Typography
            variant="h6"
            component={Link}
            to="/"
            sx={{
              fontFamily: "'Inter', sans-serif",
              fontWeight: 800,
              color: "text.primary",
              textDecoration: "none",
              fontSize: "1.25rem",
              mr: 2,
            }}
          >
            TinyModels
          </Typography>

          <Stack direction="row" spacing={3} sx={{ flexGrow: 1 }}>
            {PAGES.map((page) => {
              const isActive = location.pathname === page.path;
              return (
                <Typography
                  key={page.name}
                  component={Link}
                  to={page.path}
                  sx={{
                    fontSize: "0.95rem",
                    fontWeight: isActive ? 700 : 500,
                    color: isActive ? "text.primary" : "text.secondary",
                    textDecoration: "none",
                    px: 0.5,
                    py: 0.25,
                    borderRadius: 1,
                    "&:hover": { color: "text.primary" },
                  }}
                >
                  {page.name}
                </Typography>
              );
            })}
          </Stack>

          <Stack direction="row" spacing={1} alignItems="center">
            <Tooltip title={themeMode === "dark" ? "Switch to light" : "Switch to dark"}>
              <IconButton
                size="small"
                onClick={() => toggleTheme && toggleTheme()}
                sx={{
                  color: "text.primary",
                  border: `1px solid ${alpha(theme.palette.text.primary, 0.06)}`,
                  bgcolor: alpha(theme.palette.background.default, 0.02),
                  "&:hover": {
                    bgcolor: alpha(theme.palette.background.default, 0.04),
                  },
                }}
                aria-label="toggle theme"
              >
                {themeMode === "dark" ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
              </IconButton>
            </Tooltip>
          </Stack>
        </Toolbar>
      </Container>
    </AppBar>
  );
}