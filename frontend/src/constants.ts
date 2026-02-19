export const PAGES = [
  { name: "Semantic Gen", path: "/semantic" },
  { name: "Edit Gen", path: "/edit" }
];

export type ModelType = "semantic" | "edit";

export interface ModelConfig {
  modelPath: string | string[];
  modelInputSize: number;
  numClasses: number;
  imgDims: number[];
  condDims: number[];
  txtCondDims?: number[];
  txtKeyPaddingMaskDims?: number[];
  modelPred: "x" | "v";
  textEncoderPath?: string;
}

export const MODEL_CONFIGS: Record<ModelType, ModelConfig> = {
  semantic: {
    modelPath: [
      "/tiny-models/models/landscapes.onnx.part.1",
      "/tiny-models/models/landscapes.onnx.part.2"
    ],
    // "/landscapes_256_x-v_iters_00125000_model.onnx"
    // "/landscapes_128_v-v_iters_00400000_model.onnx"
    modelInputSize: 256, // 128
    numClasses: 8,
    imgDims: [1, 3, 256, 256], // [1, 3, 128, 128]
    condDims: [1, 8, 256, 256], // [1, 8, 128, 128]
    modelPred: "x", // v
  },

  edit: {
    modelPath: [
      "/tiny-models/models/gpt_edit_model.onnx.part.1",
      "/tiny-models/models/gpt_edit_model.onnx.part.2"
    ],
    // "/tiny-models/models/gpt_edit_model.onnx"
    // "/gpt_edit_model.onnx"
    modelInputSize: 128,
    numClasses: 0,
    imgDims: [1, 3, 128, 128],
    condDims: [1, 3, 128, 128],
    txtCondDims: [1, 100, 384],
    txtKeyPaddingMaskDims: [1, 100],
    modelPred: "v",
    textEncoderPath: "/tiny-models/all-MiniLM-L6-v2"
  },
};