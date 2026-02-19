import { useEffect, useRef, useState } from "react";

import LinearProgress from "@mui/material/LinearProgress";
import Slider from "@mui/material/Slider";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import Tooltip from "@mui/material/Tooltip";
import { Paper, Chip } from "@mui/material";
import { CheckCircle, Downloading } from "@mui/icons-material";
import { useTheme, alpha } from "@mui/material/styles";

import { tensorToCanvas } from "../utils/imageUtils";
import type { ModelType } from "../constants";
import "./InferenceDashboard.css"

interface DashboardProps {
  currentStep: number;
  totalSteps: number;
  stepTime: number;
  totalTime: number;
  imageData: Float32Array | null;
  // velocityData: Float32Array | null;
  isProcessing: boolean;
  isModelLoaded: boolean;
  modelType: ModelType;
  onStepChange: (newStep: number) => void;
  onNoiseReset: () => Promise<void>;
}

export default function InferenceDashboard({ 
  currentStep, totalSteps, stepTime, totalTime, imageData, isProcessing,
  isModelLoaded, modelType, onStepChange, onNoiseReset // velocityData
}: DashboardProps) {
  const theme = useTheme();
  const imgCanvasRef = useRef<HTMLDivElement>(null);
  // const velCanvasRef = useRef<HTMLDivElement>(null);
  const [prevTotalSteps, setPrevTotalSteps] = useState<number>(10);

  useEffect(() => {
    if (imageData && imgCanvasRef.current) {
      const canvas = tensorToCanvas(imageData, modelType, true);
      canvas.style.width = "100%";
      canvas.style.imageRendering = "pixelated";
      imgCanvasRef.current.innerHTML = "";
      imgCanvasRef.current.appendChild(canvas);
    }
    // if (velocityData && velCanvasRef.current) {
    //     const canvas = tensorToCanvas(velocityData, modelType, true);
    //     canvas.style.width = "100%";
    //     canvas.style.imageRendering = "pixelated";
    //     velCanvasRef.current.innerHTML = "";
    //     velCanvasRef.current.appendChild(canvas);
    // }
    setPrevTotalSteps(totalSteps);
  }, [imageData]);

  return (
    <Paper elevation={0} sx={{ 
      p: 2, 
      borderRadius: 3, 
      border: `1px solid ${theme.palette.divider}`,
      background: theme.palette.background.paper,
      minWidth: 280,
      position: "relative",
      overflow: "hidden"
    }}>
      
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 2, alignItems: "center" }}>
        <Typography variant="subtitle2" fontWeight="bold" sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          Model Status
        </Typography>
        
        <Chip 
          label={isModelLoaded ? "Loaded" : "Loading..."} 
          color={isModelLoaded ? "success" : "warning"}
          size="small"
          variant={isModelLoaded ? "filled" : "outlined"}
          icon={isModelLoaded ? <CheckCircle /> : <Downloading className="animate-bounce" />}
          sx={{
            fontWeight: "bold",
            fontFamily: "monospace",
            height: 24,
            fontSize: "0.7rem",
            backgroundColor: isModelLoaded ? undefined : alpha(theme.palette.warning.main, 0.12)
          }}
        />
      </Box>

      <Box sx={{ 
        opacity: isModelLoaded ? 1 : 0.5, 
        pointerEvents: isModelLoaded ? "auto" : "none",
        transition: "opacity 0.3s ease"
      }}>

        <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, mb: 2 }}>
          <Box sx={{
            bgcolor: theme.palette.mode === "dark" ? alpha(theme.palette.background.paper, 0.02) : theme.palette.background.default,
            p: 1,
            borderRadius: 2,
            border: `1px solid ${theme.palette.divider}`
          }}>
            <Typography variant="caption" color="text.secondary">Total Latency</Typography>
            <Typography variant="body2" fontWeight="bold" fontFamily="monospace">
              {totalTime.toFixed(0)}ms
            </Typography>
          </Box>
          <Box sx={{
            bgcolor: theme.palette.mode === "dark" ? alpha(theme.palette.background.paper, 0.02) : theme.palette.background.default,
            p: 1,
            borderRadius: 2,
            border: `1px solid ${theme.palette.divider}`
          }}>
            <Typography variant="caption" color="text.secondary">Step Time</Typography>
            <Typography variant="body2" fontWeight="bold" fontFamily="monospace">
              {stepTime.toFixed(1)}ms
            </Typography>
          </Box>
        </Box>

        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
            <Typography variant="caption">Sampling Step</Typography>
            <Typography variant="caption" fontWeight="bold">{currentStep}/{prevTotalSteps}</Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={(currentStep / prevTotalSteps) * 100}
            sx={{
              height: 6,
              borderRadius: 3,
              bgcolor: alpha(theme.palette.text.primary, 0.06),
              '& .MuiLinearProgress-bar': {
                bgcolor: theme.palette.primary.main,
              },
            }}
          />
        </Box>

        <Box sx={{ display: "flex", gap: 2 }}>
          <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", flex: 1 }}>
            <Typography variant="caption" sx={{ fontSize: "0.65rem" }}>Denoised X</Typography>
            <div
              ref={imgCanvasRef}
              className="canvas-container"
              style={{
                width: "128px",
                aspectRatio: "1/1",
                background: imageData && imgCanvasRef.current ? theme.palette.background.paper : alpha(theme.palette.text.primary, 0.02),
                border: `1px solid ${theme.palette.divider}`,
                borderRadius: 8,
              }}
            />
          </Box>
          {/* <Box sx={{ textAlign: "center", flex: 1 }}>
            <Typography variant="caption" sx={{ fontSize: "0.65rem" }}>Velocity</Typography>
            <div 
              ref={velCanvasRef} 
              className="canvas-container" 
              style={{ width: "128px", aspectRatio: "1/1", background: velocityData && velCanvasRef.current ? "#fafafa" : "#333" }}
            />
          </Box> */}
        </Box>

        <Box sx={{ mt: 2, pt: 2, borderTop: `1px solid ${theme.palette.divider}` }}>
          <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <Typography variant="caption" color="text.secondary">Steps ({totalSteps})</Typography>
            <Tooltip title="Reset Noise">
              <IconButton size="small" onClick={onNoiseReset} disabled={isProcessing}>
                <RestartAltIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
          <Slider
            size="small"
            defaultValue={10}
            min={1} max={50}
            aria-label="Steps"
            valueLabelDisplay="auto"
            onChange={(_, val) => onStepChange(val as number)}
            disabled={isProcessing}
          />
        </Box>

      </Box>
    </Paper>
  );
}