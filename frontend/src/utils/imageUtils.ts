import { MODEL_CONFIGS, type ModelType } from "../constants";

export function tensorToCanvas(tensor: Float32Array, modelType: ModelType, normalize = false) {
  const configs = MODEL_CONFIGS[modelType]
  
  const canvas = document.createElement("canvas");
  canvas.width = configs.modelInputSize;
  canvas.height = configs.modelInputSize;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;

  const imgData = ctx.createImageData(configs.modelInputSize, configs.modelInputSize);
  const totalPixels = configs.modelInputSize * configs.modelInputSize;

  for (let i = 0; i < totalPixels; i++) {
    let r = tensor[i];
    let g = tensor[i + totalPixels];
    let b = tensor[i + totalPixels * 2];

    if (normalize) {
        // [-1, 1] -> [0, 255]
        r = ((r + 1) / 2) * 255;
        g = ((g + 1) / 2) * 255;
        b = ((b + 1) / 2) * 255;
    } else {
        // [-1, 1] -> [0, 127]
        r = 127 + (r * 127);
        g = 127 + (g * 127);
        b = 127 + (b * 127);
    }

    imgData.data[i * 4] = r;
    imgData.data[i * 4 + 1] = g;
    imgData.data[i * 4 + 2] = b;
    imgData.data[i * 4 + 3] = 255;
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas;
}

export async function processDrawingForModel(
  base64Image: string, modelType: ModelType
): Promise<Float32Array> {
  return new Promise((resolve) => {
    const configs = MODEL_CONFIGS[modelType]
    const img = new Image();
    img.src = base64Image;
    
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = configs.modelInputSize;
      canvas.height = configs.modelInputSize;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      
      if (!ctx) return;

      ctx.imageSmoothingEnabled = false; 
      ctx.drawImage(img, 0, 0, configs.modelInputSize, configs.modelInputSize);

      const imageData = ctx.getImageData(0, 0, configs.modelInputSize, configs.modelInputSize).data;
      const totalPixels = configs.modelInputSize * configs.modelInputSize;

      const tensorData = new Float32Array(configs.numClasses * totalPixels).fill(0.0);
      for (let i = 0; i < totalPixels; i++) {
        const r = imageData[i * 4];
        const g = imageData[i * 4 + 1];
        const b = imageData[i * 4 + 2];
        const a = imageData[i * 4 + 3];

        let classIdx = 0;

        if (a < 128) {
            classIdx = 0; 
        } else {
            classIdx = getNearestClassIndex(r, g, b);
        }

        // one hot encoding
        const tensorIndex = (classIdx * totalPixels) + i;
        tensorData[tensorIndex] = 1.0;
      }

      resolve(tensorData);
    };
  });
}

export async function centerCropImage(base64Input: string, modelType: ModelType): Promise<string> {
  return new Promise((resolve, reject) => {
    const configs = MODEL_CONFIGS[modelType]
    const img = new Image();
    img.src = base64Input;

    img.onload = () => {
      const minDim = Math.min(img.width, img.height);
      const sx = (img.width - minDim) / 2;
      const sy = (img.height - minDim) / 2;

      const canvas = document.createElement("canvas");
      canvas.width = configs.modelInputSize;
      canvas.height = configs.modelInputSize;
      
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Could not get canvas context"));
        return;
      }

      ctx.drawImage(
        img, 
        sx, sy, minDim, minDim, //center crop
        0, 0, configs.modelInputSize, configs.modelInputSize // resize to model size
      );

      resolve(canvas.toDataURL("image/png"));
    };

    img.onerror = (err) => reject(err);
  });
}

export async function imageToTensor(base64Input: string, modelType: ModelType): Promise<Float32Array> {
  return new Promise((resolve, reject) => {
    const configs = MODEL_CONFIGS[modelType]
    const img = new Image();
    img.src = base64Input;
    img.crossOrigin = "anonymous";

    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = configs.modelInputSize;
      canvas.height = configs.modelInputSize;
      
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) {
        reject(new Error("Could not get canvas context"));
        return;
      }

      ctx.drawImage(img, 0, 0, configs.modelInputSize, configs.modelInputSize);

      const imageData = ctx.getImageData(0, 0, configs.modelInputSize, configs.modelInputSize).data;
      const totalPixels = configs.modelInputSize * configs.modelInputSize;
      
      const tensor = new Float32Array(3 * totalPixels);
      for (let i = 0; i < totalPixels; i++) {
        // [0, 255] -> [-1.0, 1.0]
        const r = (imageData[i * 4] / 127.5) - 1.0;
        const g = (imageData[i * 4 + 1] / 127.5) - 1.0;
        const b = (imageData[i * 4 + 2] / 127.5) - 1.0;

        tensor[i] = r;
        tensor[i + totalPixels] = g;
        tensor[i + totalPixels * 2] = b;
      }

      resolve(tensor);
    };

    img.onerror = (err) => reject(err);
  });
}

export interface Material {
  id: number;
  name: string;
  color: string;
  rgb: [number, number, number];
}

export const PALETTE: Material[] = [
  { id: 0, name: "Sky",    color: "#87CEEB", rgb: [135, 206, 235] },
  { id: 1, name: "Nature", color: "#32CD32", rgb: [50, 205, 50] },
  { id: 2, name: "Mount",  color: "#474747ff", rgb: [85, 107, 47] }, 
  { id: 3, name: "Water",  color: "#4169E1", rgb: [65, 105, 225] },
  { id: 5, name: "Sand",   color: "#F4A460", rgb: [244, 164, 96] },
  { id: 4, name: "Road",   color: "#9e9e9eff", rgb: [128, 128, 128] },
];

export function getNearestClassIndex(r: number, g: number, b: number): number {
  let minDist = Infinity;
  let nearestId = 0;

  for (const mat of PALETTE) {
    const dist = Math.sqrt(
      Math.pow(mat.rgb[0] - r, 2) +
      Math.pow(mat.rgb[1] - g, 2) +
      Math.pow(mat.rgb[2] - b, 2)
    );
    if (dist < minDist) {
      minDist = dist;
      nearestId = mat.id;
    }
  }
  return nearestId;
}
