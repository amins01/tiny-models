export interface InferenceTelemetry {
  step: number;
  totalSteps: number;
  image: Float32Array;
  // velocity: Float32Array;
  sketch: Float32Array;
  stepTime: number;
  totalTime: number;
}