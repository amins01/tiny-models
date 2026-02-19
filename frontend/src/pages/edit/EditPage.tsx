import { useCallback, useEffect, useRef, useState } from "react";

import "./EditPage.css"
import OutputCanvas from "../../components/OutputCanvas";
import { useInference } from "../../hooks/useInference";
import InferenceDashboard from "../../components/InferenceDashboard";
import type { InferenceTelemetry } from "../../types";
import { generateGaussianNoise } from "../../utils/tensorUtils";
import { MODEL_CONFIGS } from "../../constants";
import { centerCropImage, imageToTensor, tensorToCanvas } from "../../utils/imageUtils";
import FillerCard from "../../components/FillerCard";

import Typography from "@mui/material/Typography";
import Paper from "@mui/material/Paper";
import InputBase from "@mui/material/InputBase";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import { Image as ImageIcon } from "@mui/icons-material";
import SendIcon from "@mui/icons-material/Send";
import Button from "@mui/material/Button";
import Box from "@mui/material/Box";
import EditIcon from "@mui/icons-material/Edit";
import Alert from "@mui/material/Alert";
import Collapse from "@mui/material/Collapse";
import CloseIcon from "@mui/icons-material/Close";

export default function EditPage() {
  const MODEL_TYPE = "edit"
  const CONFIGS  = MODEL_CONFIGS[MODEL_TYPE]
  const [outputCanvas, setOutputCanvas] = useState<any>(null);
  const {ready, generate} = useInference(CONFIGS.modelPath, MODEL_TYPE);
  const [noiseData, setNoiseData] = useState<Float32Array>(generateGaussianNoise(CONFIGS.imgDims.reduce((a, b) => a * b, 1)));
  const isProcessing = useRef<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [imageBase64, setImageBase64] = useState<string>("");
  const [prompt, setPrompt] = useState<string>("");
  const [isCurrentOutputUsed, setIsCurrentOutputUsed] = useState<boolean>(false);
  const [openAlert, setOpenAlert] = useState(true);

  // this.noise = generateGaussianNoise(this.img_dims.reduce((a, b) => a * b, 1));

  const [dashboardState, setDashboardState] = useState<InferenceTelemetry>({
    step: 0,
    totalSteps: 10,
    stepTime: 0,
    totalTime: 0,
    image: new Float32Array(0),
    // velocity: new Float32Array(0),
    sketch: new Float32Array(0),
  });

  const handleWorkerUpdate = useCallback((data: InferenceTelemetry) => {
    setDashboardState(data);
  }, []);

  const startGeneration = async () => {
    if (!ready || !imageBase64) return;

    isProcessing.current = true;
    
    setDashboardState(prev => ({
        ...prev,
        step: 0,
        totalTime: 0,
        image: new Float32Array(0)
    }));

    const dataTensor = await imageToTensor(imageBase64, MODEL_TYPE);
    const result = await generate(dataTensor!, dashboardState.totalSteps, noiseData, handleWorkerUpdate, prompt);
    const canvas = tensorToCanvas(new Float32Array(result as Float32Array), MODEL_TYPE, true);
    setOutputCanvas(canvas);
    isProcessing.current = false;
    setIsCurrentOutputUsed(false); // TODO: change
  };

  useEffect(() => {
    if (dashboardState.sketch?.length > 0) startGeneration();
  }, [noiseData])

  const onNoiseReset = async () => {
    const newNoise = generateGaussianNoise(CONFIGS.imgDims.reduce((a, b) => a * b, 1));
    setNoiseData(newNoise);
  };

  const onStepChange = (newStep: number) => {
    setDashboardState(prev => ({
        ...prev, totalSteps: newStep
    }));
  };

  const onGenerate = (event: any) => {
    event.preventDefault();
    startGeneration();
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (event) => {
      const base64 = event.target?.result as string;
      const img = await centerCropImage(base64, MODEL_TYPE);
      setImageBase64(img);
      setIsCurrentOutputUsed(false);
    };
    reader.readAsDataURL(file);
    e.target.value = ""; 
  };

  const onEditGeneratedImageClick = () => {
    if (!outputCanvas) return;

    const base64 = outputCanvas.toDataURL("image/png");
    setImageBase64(base64);
    setIsCurrentOutputUsed(true);
  }

  return (
    <>
      <div className="column container">
        <Typography variant="h3" fontWeight="400" sx={{ mt: 1, mb: 2, letterSpacing: "-0.02em" }}>
          Image Editing Model
        </Typography>
        <Typography variant="body1" component="div"
          sx={{
            fontSize: "1.125rem",
            lineHeight: 1.75,
            textAlign: "left",
            paddingLeft: "1em",
            paddingRight: "1em",
            maxWidth: "1000px"
          }}
        >
          Image editing model trained from scratch running fully locally (CPU).
        </Typography>
        <Collapse in={openAlert}>
          <Alert 
            severity="warning"
            sx={{ maxWidth: "900px" }}
            action={
              <IconButton
                color="inherit"
                size="small"
                onClick={() => setOpenAlert(false)}
              >
                <CloseIcon fontSize="inherit" />
              </IconButton>
            }
          >
            Inference is slower on the deployed site because it's hosted on GitHub Pages (which I chose for a free "set and forget" setup) 
            and it doesn't allow the browser headers required for WASM multithreading.
            Running it locally should lead to much faster inference.
          </Alert>
        </Collapse>
      </div>
      <div className="row container">
        <div className="column">
          <Paper
            component="form"
            onSubmit={onGenerate}
            sx={{ p: "2px 4px", display: "flex", alignItems: "center", width: "100%" }}
          >
            <InputBase
              sx={{ ml: 1, flex: 1 }}
              placeholder="Write an editing prompt"
              onChange={(event: React.ChangeEvent<HTMLInputElement>) => {
                setPrompt(event.target.value);
              }}
            />
            <Divider sx={{ height: 28, m: 0.5 }} orientation="vertical" />
            <IconButton color="primary" sx={{ p: "10px" }} type="submit" disabled={!imageBase64 || !prompt || isProcessing.current}>
              <SendIcon />
            </IconButton>
          </Paper>
        
          <div className="row container">
            {imageBase64 ? 
              <div className="column">
                <img src={imageBase64} style={{ width: "256px", height: "256px", borderRadius: 3 }} />
                <input type="file" ref={fileInputRef} onChange={handleImageUpload} hidden accept="image/*" />
                <Button
                  variant="contained"
                  startIcon={<ImageIcon />}
                  onClick={() => fileInputRef.current?.click()}
                  fullWidth
                >
                  Upload New Image
                </Button>
              </div>
              :
              <FillerCard
                content={
                <>
                  <input type="file" ref={fileInputRef} onChange={handleImageUpload} hidden accept="image/*" />
                  <Button variant="outlined" startIcon={<ImageIcon />} onClick={() => fileInputRef.current?.click()}>
                    Upload Image
                  </Button>
                  <Box sx={{ textAlign: "center" }}>
                      <Typography variant="caption" sx={{ color: "#94A3B8" }}>
                        <div>
                          1. Upload an image
                          <br /> 
                          2. Write an editing prompt
                          <br />
                          3. Click the send button
                        </div>
                      </Typography>
                    </Box>
                </>
                }
              />
            }
            <div className="column">
              <OutputCanvas canvas={outputCanvas} displayText="Model's output will be displayed here" />
              {outputCanvas &&
              <>
                <Button
                  variant="contained"
                  color="secondary"
                  startIcon={<EditIcon />}
                  onClick={() => onEditGeneratedImageClick()}
                  disabled={isCurrentOutputUsed}
                  fullWidth
                >
                  Edit Generated Image
                </Button>
              </>}
            </div>
          </div>
        </div>

        <div className="column">
          <InferenceDashboard
            currentStep={dashboardState.step}
            totalSteps={dashboardState.totalSteps}
            stepTime={dashboardState.stepTime}
            totalTime={dashboardState.totalTime}
            imageData={dashboardState.image.length > 0 ? dashboardState.image : null}
            // velocityData={dashboardState.velocity.length > 0 ? dashboardState.velocity : null}
            onStepChange={onStepChange}
            onNoiseReset={onNoiseReset}
            isProcessing={isProcessing.current}
            isModelLoaded={ready}
            modelType={MODEL_TYPE}
          />
        </div>
      </div>
    </>
  );
}