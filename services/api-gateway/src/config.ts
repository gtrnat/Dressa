import path from "node:path";

function bool(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined || value === "") return fallback;
  return ["1", "true", "yes"].includes(value.toLowerCase());
}

const isProduction = process.env.NODE_ENV === "production";

export const config = {
  port: Number(process.env.PORT ?? process.env.API_GATEWAY_PORT ?? 3000),
  host: process.env.HOST ?? "0.0.0.0",
  mlEngineUrl: (process.env.ML_ENGINE_URL ?? "http://localhost:8000").replace(/\/$/, ""),
  // Local file storage until S3/R2/Supabase Storage is wired in.
  uploadDir: path.resolve(process.env.UPLOAD_DIR ?? "uploads"),
  publicBaseUrl: (process.env.PUBLIC_BASE_URL ?? `http://localhost:${process.env.API_GATEWAY_PORT ?? 3000}`).replace(/\/$/, ""),
  maxUploadBytes: 10 * 1024 * 1024,
  // DEV ONLY: trust the X-User-Id header instead of real auth. Never enabled in production.
  devAuth: !isProduction && bool(process.env.DEV_AUTH, true),
};
