import * as ort from "onnxruntime-web";
import { env, AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer } from "@huggingface/transformers";
import * as Comlink from "comlink";

import type { InferenceTelemetry } from "../types";
import { MODEL_CONFIGS, type ModelConfig, type ModelType } from "../constants";

ort.env.wasm.wasmPaths = "/tiny-models/";

if (typeof navigator !== "undefined" && (self as any).crossOriginIsolated) {
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 2;
} else {
  ort.env.wasm.numThreads = 1;
}

ort.env.wasm.proxy = false;
env.allowLocalModels = true;
env.useBrowserCache = true;

class InferenceEngine {
  txtTokenizer: PreTrainedTokenizer | null = null;
  txtEncoder: PreTrainedModel | null = null;
  session: ort.InferenceSession | null = null;
  noise: any = null;
  modelType: string = "";
  configs: ModelConfig | null = null;

  async loadModel(path: string[], modelType: ModelType) {
    console.log("Loading model...")
    try {
      this.modelType = modelType;
      this.configs = MODEL_CONFIGS[modelType];

      let modelDataUint8: Uint8Array;

      // load and merge model files
      const buffers = await Promise.all(
        path.map(p => fetch(p, { mode: "cors", credentials: "omit" }).then(res => {
          if (!res.ok) throw new Error(`Failed to fetch part ${p}: ${res.status}`);
          return res.arrayBuffer();
        }))
      );

      const total = buffers.reduce((s, b) => s + b.byteLength, 0);
      modelDataUint8 = new Uint8Array(total);
      let offset = 0;
      for (const buf of buffers) {
        modelDataUint8.set(new Uint8Array(buf), offset);
        offset += buf.byteLength;
      }

      console.log("Creating inference session...");

      this.session = await ort.InferenceSession.create(modelDataUint8, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
      });

      if (this.configs.textEncoderPath) { // modelType == "edit"
        console.log("Loading text encoder...");
        
        this.txtTokenizer = await AutoTokenizer.from_pretrained(this.configs.textEncoderPath, {
          local_files_only: true 
        });
        
        this.txtEncoder = await AutoModel.from_pretrained(this.configs.textEncoderPath, {
          local_files_only: true
        });
      }
    } catch (e) {
      console.log("Error while loading the model")
      console.error(e);
      throw e;
    }
    console.log("Model loaded")
    return "Model Loaded";
  }

  async sample(
    condTensor: Float32Array,
    T: number = 10,
    noiseTensor: Float32Array,
    onUpdate: React.Dispatch<React.SetStateAction<InferenceTelemetry>>,
    prompt: string | null = null
  ) {
    if (!this.session) throw new Error("Model not loaded");
    if (prompt && (!this.txtTokenizer || !this.txtEncoder)) throw new Error("Text tokenizer or encoder not loaded");

    const cond: ort.Tensor = new ort.Tensor("float32", condTensor, this.configs?.condDims);
    const dt: number = 1 / T;
    const noise = noiseTensor.slice();
    
    let txtCond: ort.Tensor | null = null;
    let txtKeyPaddingMask: ort.Tensor | null = null;
    
    if (prompt) {
      const maxLen = this.configs!.txtCondDims![1];
      const tokenizerOut = await this.txtTokenizer!(prompt, {
          padding: "max_length",
          truncation: true,
          max_length: maxLen,
          return_tensor: "pt"
      });
      
      const { last_hidden_state } = await this.txtEncoder!({
          input_ids: tokenizerOut.input_ids,
          attention_mask: tokenizerOut.attention_mask
      });

      txtCond = new ort.Tensor(
          "float32", 
          last_hidden_state.data, 
          this.configs!.txtCondDims
      );

      // invert mask for pytorch
      const maskData = tokenizerOut.attention_mask.data;
      const boolMask = new Uint8Array(maskData.length);
      for (let i = 0; i < maskData.length; i++) {
          boolMask[i] = maskData[i] === 0n ? 1 : 0;
      }

      txtKeyPaddingMask = new ort.Tensor(
          "bool", 
          boolMask, 
          this.configs!.txtKeyPaddingMaskDims
      );
    }

    let x: ort.Tensor = new ort.Tensor("float32", noise, this.configs?.imgDims);
    let velocity: ort.Tensor;
    
    const startTotal = performance.now();

    for (let i = 0; i < T; i++) {
      const stepStart = performance.now();

      const t = i * dt;

      let result = null

      // TODO: improve/clean up
      if (txtCond && txtKeyPaddingMask) {
        result = await this.session.run({
          x: x,
          t: new ort.Tensor("float32", [t], [1]),
          cond: cond,
          txt_cond: txtCond,
          txt_key_padding_mask: txtKeyPaddingMask,
        });
      }
      else {
        result = await this.session.run({
          x: x,
          t: new ort.Tensor("float32", [t], [1]),
          cond: cond
        });
      }

      const modelOut: ort.Tensor = result!.velocity; // TODO: improve naming

      if (this.configs!.modelPred === "x" && i === T - 1) {
        x = modelOut;
      }
      else {
        const denom = 1 - Math.max(0.05, t);
        velocity = modelOut;
        for (let j = 0; j < x.data.length; j++) {
          if (this.configs!.modelPred === "x") {
            velocity.data[j] = (Number(modelOut.data[j]) - Number(x.data[j])) / denom;
          }
          x.data[j] = Number(x.data[j]) + (Number(velocity.data[j]) * dt);
        }
      }
      
      const stepEnd = performance.now();
      
      if (i % 2 === 0 || i === T - 1) {
        onUpdate({ 
          step: i + 1,
          totalSteps: T,
          image: new Float32Array(x.data as Float32Array), 
          // velocity: new Float32Array(velocity!.data as Float32Array),
          stepTime: stepEnd - stepStart,
          totalTime: stepEnd - startTotal,
          sketch: condTensor
        });
      }
    }
    
    return x.data;
  }
}

Comlink.expose(new InferenceEngine());