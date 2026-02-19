import { useEffect, useRef } from "react";

import { AutoAwesome } from "@mui/icons-material";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { useTheme } from "@mui/material/styles";

import FillerCard from "./FillerCard";

interface OutputCanvasProps {
  canvas: any;
  displayText: any;
}

export default function OutputCanvas({ canvas, displayText }: OutputCanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();

  useEffect(() => {
    if (!canvas || !canvasRef.current) return;

    canvas.style.width = "100%";
    canvasRef.current.innerHTML = "";
    canvasRef.current.appendChild(canvas);
  }, [canvas]);

  const placeholderIconBg = theme.palette.mode === "dark"
    ? theme.palette.background.paper
    : theme.palette.background.default;

  const placeholderBorder = theme.palette.divider;
  const placeholderText = theme.palette.text.secondary;

  return (
    <div>
      {!canvas && (
        <FillerCard
          content={
            <>
              <Box
                sx={{
                  p: 2,
                  borderRadius: "50%",
                  bgcolor: placeholderIconBg,
                  border: `1px solid ${placeholderBorder}`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: placeholderText,
                }}
              >
                <AutoAwesome fontSize="medium" />
              </Box>
              <Box sx={{ textAlign: "center", mt: 1, ml: 1, mr: 1}}>
                <Typography variant="caption" sx={{ color: placeholderText }}>
                  {displayText}
                </Typography>
              </Box>
            </>
          }
        />
      ) || (
        <div
          ref={canvasRef}
          style={{
            width: 256,
            height: 256,
            imageRendering: "pixelated",
            borderRadius: 8,
            overflow: "hidden",
            boxShadow: theme.shadows[4],
            border: `1px solid ${theme.palette.divider}`,
            background: theme.palette.background.paper,
          }}
        />
      )}
    </div>
  );
}