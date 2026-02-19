import { useState, useEffect, useRef } from "react";
import * as Comlink from "comlink";
import type { InferenceTelemetry } from "../types";

export function useInference(modelPath: string | string[], modelType: string) {
  const [ready, setReady] = useState(false);
  const engineRef = useRef<any>(null);

  useEffect(() => {
    setReady(false);

    const worker = new Worker(
      new URL("../engine/inference.worker.ts", import.meta.url), 
      { type: "module" }
    );
    
    const engine: any = Comlink.wrap(worker);
    engineRef.current = engine;

    engine.loadModel(modelPath, modelType).then(() => setReady(true));

    return () => worker.terminate();
  }, [modelPath]);

  const generate = async (
    data: Float32Array, numTimesteps: number,
    noiseTensor: Float32Array, onUpdate: (data: InferenceTelemetry) => void,
    prompt: string | null = null
  ) => {
    if (!engineRef.current) return;

    return await engineRef.current.sample(
      data,
      numTimesteps,
      noiseTensor,
      Comlink.proxy(onUpdate),
      prompt
    );
    // return await engineRef.current.sample(data, numTimesteps);
  };

  return {ready, generate};
}