import path from "node:path";

function bool(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined || value === "") return fallback;
  return ["1", "true", "yes"].includes(value.toLowerCase());
}

function num(value: string | undefined, fallback: number): number {
  const n = Number(value);
  return value === undefined || value === "" || Number.isNaN(n) ? fallback : n;
}

const isProduction = process.env.NODE_ENV === "production";
const port = Number(process.env.PORT ?? process.env.API_GATEWAY_PORT ?? 3000);
const publicBaseUrl = (process.env.PUBLIC_BASE_URL ?? `http://localhost:${port}`).replace(/\/$/, "");

export const config = {
  port,
  host: process.env.HOST ?? "0.0.0.0",
  mlEngineUrl: (process.env.ML_ENGINE_URL ?? "http://localhost:8000").replace(/\/$/, ""),
  qdrantUrl: (process.env.QDRANT_URL ?? "http://localhost:6333").replace(/\/$/, ""),
  qdrantApiKey: process.env.QDRANT_API_KEY || undefined,
  qdrantCollection: process.env.QDRANT_COLLECTION ?? "wardrobe_items",
  redisUrl: process.env.REDIS_URL ?? "redis://localhost:6379",
  // Local file storage until S3/R2/Supabase Storage is wired in.
  uploadDir: path.resolve(process.env.UPLOAD_DIR ?? "uploads"),
  publicBaseUrl,
  maxUploadBytes: 10 * 1024 * 1024,
  // DEV ONLY: trust the X-User-Id header instead of real auth. Never enabled in production.
  devAuth: !isProduction && bool(process.env.DEV_AUTH, true),

  openWeather: {
    apiKey: process.env.OPENWEATHER_API_KEY || undefined,
    baseUrl: (process.env.OPENWEATHER_BASE_URL ?? "https://api.openweathermap.org").replace(/\/$/, ""),
    cacheTtlSec: 30 * 60,
  },

  google: {
    clientId: process.env.GOOGLE_CLIENT_ID || undefined,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET || undefined,
    redirectUri: process.env.GOOGLE_REDIRECT_URI ?? `${publicBaseUrl}/integrations/google/callback`,
    // Where the browser is sent after connecting. Default: a small confirmation page on the gateway.
    successRedirect: process.env.GOOGLE_SUCCESS_REDIRECT || undefined,
  },
  // 32-byte key (base64) for encrypting OAuth refresh tokens at rest.
  tokenEncryptionKey: process.env.TOKEN_ENCRYPTION_KEY || undefined,
  // Secret for signing the OAuth "state" parameter (CSRF protection).
  oauthStateSecret: process.env.OAUTH_STATE_SECRET || undefined,

  // Default weights of Score = w1*StyleMatch + w2*WeatherFit + w3*CalendarFit + w4*RotationWeight.
  // Missing components (no weather / no calendar) are dropped and the rest renormalized.
  scoreWeights: {
    styleMatch: num(process.env.W_STYLE_MATCH, 0.25),
    weatherFit: num(process.env.W_WEATHER_FIT, 0.25),
    calendarFit: num(process.env.W_CALENDAR_FIT, 0.25),
    rotationWeight: num(process.env.W_ROTATION, 0.25),
  },
};
