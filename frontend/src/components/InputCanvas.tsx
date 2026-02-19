import { useRef, useState } from "react";

import { ReactSketchCanvas, type ReactSketchCanvasRef } from "react-sketch-canvas";
import { Slider, Typography, Paper, Button, Box, Divider } from "@mui/material";
import { Delete, Image } from "@mui/icons-material";
import { useTheme } from "@mui/material/styles";

import { PALETTE, processDrawingForModel, type Material } from "../utils/imageUtils";
import { type ModelType } from "../constants";

interface InputCanvasProps {
  onDraw: (tensorData: Float32Array) => void;
  modelType: ModelType;
}

export default function InputCanvas({ onDraw, modelType }: InputCanvasProps) {
  const canvasRef = useRef<ReactSketchCanvasRef>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const theme = useTheme();

  const [selectedMaterial, setSelectedMaterial] = useState<Material>(PALETTE[1]);
  const [tool, setTool] = useState<string>("pen");
  const [backgroundImage, setBackgroundImage] = useState<string>("");
  const [brushSize, setBrushSize] = useState<number>(20);

  const handleStroke = async () => {
    if (!canvasRef.current) return;
    const imageBase64 = await canvasRef.current.exportImage("png");
    const tensorData = await processDrawingForModel(imageBase64, modelType);
    onDraw(tensorData);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (event) => {
      const base64 = event.target?.result as string;
      setBackgroundImage(base64);
      canvasRef.current?.clearCanvas();
      const tensorData = await processDrawingForModel(base64, modelType);
      onDraw(tensorData);
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  const paperBg = theme.palette.background.paper;
  const borderColor = theme.palette.divider;
  const textColor = theme.palette.text.primary;
  const secondaryText = theme.palette.text.secondary;
  const btnBorder = borderColor;

  return (
    <Box sx={{ display: "flex", gap: 3, alignItems: "start" }}>
      <Paper elevation={0} sx={{
        p: 2,
        width: 200,
        borderRadius: 3,
        border: `1px solid ${borderColor}`,
        display: "flex",
        flexDirection: "column",
        gap: 2,
        bgcolor: paperBg,
        color: textColor
      }}>
        <Box>
          <Typography variant="caption" fontWeight="bold" sx={{ color: textColor }}>
            Materials
          </Typography>
          <Box sx={{
            display: "grid",
            gridTemplateColumns: "repeat(2, 1fr)",
            gap: 1,
            mt: 1
          }}>
            {PALETTE.map(item => {
              const isSelected = selectedMaterial.id === item.id && tool !== "eraser";
              return (
                <Button
                  key={item.id}
                  variant={isSelected ? "contained" : "outlined"}
                  onClick={() => { setSelectedMaterial(item); setTool("pen"); }}
                  sx={{
                    justifyContent: "flex-start",
                    textTransform: "none",
                    borderColor: isSelected ? "transparent" : borderColor,
                    bgcolor: isSelected ? item.color : "transparent",
                    color: isSelected ? theme.palette.getContrastText(item.color) : secondaryText,
                    textShadow: isSelected ? "0 1px 2px rgba(0,0,0,0.3)" : "none",
                    "&:hover": {
                      bgcolor: item.color,
                      color: theme.palette.getContrastText(item.color),
                      opacity: 0.9
                    }
                  }}
                  size="small"
                >
                  <Box
                    component="span"
                    sx={{
                      width: 12, height: 12, borderRadius: "50%",
                      bgcolor: item.color, mr: 1,
                      border: `1px solid rgba(0,0,0,0.1)`
                    }}
                  />
                  {item.name}
                </Button>
              );
            })}
          </Box>
        </Box>

        <Divider sx={{ borderColor }} />

        <Box>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <Typography variant="caption" fontWeight="bold" sx={{ color: textColor }}>Size</Typography>
            <Typography variant="caption" sx={{ color: secondaryText }}>{brushSize}px</Typography>
          </Box>
          <Slider
            size="small"
            value={brushSize}
            min={5} max={60}
            onChange={(_, val) => setBrushSize(val as number)}
            sx={{ color: theme.palette.primary.main }}
          />
        </Box>

        <Divider sx={{ borderColor }} />

        <Box sx={{ display: "flex", gap: 1 }}>
          <input type="file" ref={fileInputRef} onChange={handleImageUpload} hidden accept="image/*" />
          <Button
            fullWidth variant="outlined" size="small"
            onClick={() => fileInputRef.current?.click()}
            startIcon={<Image />}
            sx={{ borderColor: btnBorder, color: textColor }}
          >
            Upload
          </Button>
          <Button
            fullWidth variant="outlined" size="small"
            onClick={() => { canvasRef.current?.clearCanvas(); setBackgroundImage(""); }}
            startIcon={<Delete />}
            sx={{ borderColor: btnBorder, color: textColor }}
          >
            Clear
          </Button>
        </Box>
      </Paper>

      <div className="canvas-container" style={{
        position: "relative", width: 256, height: 256,
        border: `1px solid ${borderColor}`,
        borderRadius: 8,
        background: paperBg,
        boxShadow: theme.shadows[4]
      }}>
        <ReactSketchCanvas
          ref={canvasRef}
          width="256px"
          height="256px"
          strokeWidth={brushSize}
          strokeColor={tool === "eraser" ? PALETTE[0].color : selectedMaterial.color}
          canvasColor={PALETTE[0].color}
          backgroundImage={backgroundImage}
          preserveBackgroundImageAspectRatio="meet"
          onStroke={handleStroke}
          style={{ borderRadius: 8 }}
        />
      </div>
    </Box>
  );
}