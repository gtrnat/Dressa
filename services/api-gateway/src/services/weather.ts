import { config } from "../config.js";
import { HttpError } from "../lib/errors.js";
import { cache } from "../lib/redis.js";
import { localParts } from "../lib/time.js";

/** Daytime weather summary for one local date, used by WeatherFit. */
export interface WeatherSummary {
  date: string;
  locationName: string | null;
  /** Mean "feels like" temperature over the daytime window, °C */
  feelsLikeC: number;
  tempMinC: number;
  tempMaxC: number;
  /** Highest probability of precipitation in the window, 0..1 */
  precipProbability: number;
  /** Total rain in the window, mm */
  rainMm: number;
  condition: string;
  rainy: boolean;
  /** Forecast points (3-hourly) the summary is based on */
  samples: number;
  source: "openweathermap";
}

interface OwmForecast {
  list: Array<{
    dt: number;
    main: { temp: number; feels_like: number; temp_min: number; temp_max: number };
    weather: Array<{ main: string; description: string }>;
    pop?: number;
    rain?: { "3h"?: number };
  }>;
  city?: { name?: string };
}

const DAY_START_HOUR = 6;
const DAY_END_HOUR = 22;
const RAINY_PROBABILITY = 0.5;
const RAINY_MM = 1;

async function fetchForecast(lat: number, lon: number): Promise<OwmForecast> {
  const key = `weather:owm:v1:${lat.toFixed(2)}:${lon.toFixed(2)}`;
  const cached = await cache.get<OwmForecast>(key);
  if (cached) return cached;

  if (!config.openWeather.apiKey) throw new HttpError(501, "OPENWEATHER_API_KEY is not configured");
  const url = new URL(`${config.openWeather.baseUrl}/data/2.5/forecast`);
  url.search = new URLSearchParams({
    lat: String(lat),
    lon: String(lon),
    units: "metric",
    appid: config.openWeather.apiKey,
  }).toString();

  let res: Response;
  try {
    res = await fetch(url, { signal: AbortSignal.timeout(8000) });
  } catch (err) {
    throw new HttpError(503, `Weather service unreachable: ${(err as Error).message}`);
  }
  if (res.status === 401) throw new HttpError(502, "OpenWeatherMap rejected the API key (new keys can take ~1 hour to activate)");
  if (!res.ok) throw new HttpError(502, `OpenWeatherMap error (${res.status})`);
  const data = (await res.json()) as OwmForecast;
  await cache.set(key, data, config.openWeather.cacheTtlSec);
  return data;
}

function mostCommon(values: string[]): string {
  const counts = new Map<string, number>();
  for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? "unknown";
}

/** Summarize the forecast for `date` (local, in `timeZone`) between 06:00 and 22:00. */
export function summarizeForecast(data: OwmForecast, date: string, timeZone: string): Omit<WeatherSummary, "date"> | null {
  const now = Date.now() / 1000;
  let points = data.list.filter((p) => {
    const local = localParts(new Date(p.dt * 1000), timeZone);
    return local.date === date && local.hour >= DAY_START_HOUR && local.hour <= DAY_END_HOUR;
  });
  // Late evening "today": the daytime window is over, so describe the next few hours instead.
  if (points.length === 0) {
    const first = data.list.find((p) => p.dt >= now - 3 * 3600);
    if (first && localParts(new Date(first.dt * 1000), timeZone).date === date) {
      points = data.list.filter((p) => p.dt >= first.dt).slice(0, 4);
    }
  }
  if (points.length === 0) return null;

  const avg = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const precipProbability = Math.max(...points.map((p) => p.pop ?? 0));
  const rainMm = points.reduce((s, p) => s + (p.rain?.["3h"] ?? 0), 0);
  return {
    locationName: data.city?.name ?? null,
    feelsLikeC: Math.round(avg(points.map((p) => p.main.feels_like)) * 10) / 10,
    tempMinC: Math.round(Math.min(...points.map((p) => p.main.temp_min)) * 10) / 10,
    tempMaxC: Math.round(Math.max(...points.map((p) => p.main.temp_max)) * 10) / 10,
    precipProbability: Math.round(precipProbability * 100) / 100,
    rainMm: Math.round(rainMm * 10) / 10,
    condition: mostCommon(points.map((p) => p.weather[0]?.description ?? "unknown")),
    rainy: precipProbability >= RAINY_PROBABILITY || rainMm >= RAINY_MM,
    samples: points.length,
    source: "openweathermap",
  };
}

export async function getDailyWeather(lat: number, lon: number, date: string, timeZone: string): Promise<WeatherSummary> {
  const summary = summarizeForecast(await fetchForecast(lat, lon), date, timeZone);
  if (!summary) throw new HttpError(422, `No forecast available for ${date} (OpenWeatherMap covers the next 5 days)`);
  return { date, ...summary };
}
