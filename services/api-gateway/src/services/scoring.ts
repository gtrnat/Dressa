/**
 * Outfit scoring: Score = w1*StyleMatch + w2*WeatherFit + w3*CalendarFit + w4*RotationWeight
 *
 * Every component is in [0, 1]. A component that can't be computed (no weather, no calendar)
 * is null and is left out; the remaining weights are renormalized so they still sum to 1.
 * All functions are pure so they can be unit-tested and explained in the report.
 */
import type { EventType } from "./event-classifier.js";
import type { WeatherSummary } from "./weather.js";

export type Category = "TOP" | "BOTTOM" | "OUTERWEAR" | "DRESS" | "FOOTWEAR" | "ACCESSORY";

export interface ScoreItem {
  id: string;
  category: Category;
  itemType: string;
  colors: string[];
  styles: string[];
  warmthLevel: number;
  waterResistant: boolean;
  wearCount: number;
  lastWornAt: Date | null;
  vector?: number[];
}

export interface Components {
  styleMatch: number;
  weatherFit: number | null;
  calendarFit: number | null;
  rotationWeight: number;
}
export type Weights = Record<keyof Components, number>;

const clamp = (x: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x));
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const gaussian = (diff: number, sigma: number) => Math.exp(-(diff * diff) / (2 * sigma * sigma));

function pairs<T>(xs: T[]): Array<[T, T]> {
  const out: Array<[T, T]> = [];
  for (let i = 0; i < xs.length; i++) for (let j = i + 1; j < xs.length; j++) out.push([xs[i]!, xs[j]!]);
  return out;
}

// ---------------------------------------------------------------------------
// StyleMatch = 0.4*color harmony + 0.3*visual cohesion (CLIP) + 0.3*style consistency
// ---------------------------------------------------------------------------

interface Hsl { h: number; s: number; l: number }

export function hexToHsl(hex: string): Hsl | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return null;
  const n = parseInt(m[1]!, 16);
  const [r, g, b] = [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return { h: 0, s: 0, l };
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  const h = max === r ? (g - b) / d + (g < b ? 6 : 0) : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return { h: h * 60, s, l };
}

/** Neutrals (black/white/gray, navy, denim, beige/khaki/brown) go with almost anything. */
export function isNeutral(c: Hsl): boolean {
  if (c.s < 0.2 || c.l < 0.15 || c.l > 0.9) return true;
  if (c.h >= 200 && c.h <= 240 && c.l < 0.5 && c.s < 0.6) return true; // navy, denim
  if (c.h >= 15 && c.h <= 50 && c.s < 0.5) return true; // beige, khaki, tan, brown
  return false;
}

function hueHarmony(a: number, b: number): number {
  const d = Math.min(Math.abs(a - b), 360 - Math.abs(a - b));
  if (d <= 30) return 0.9; // analogous
  if (d <= 60) return 0.55;
  if (d <= 100) return 0.4; // tends to clash
  if (d <= 140) return 0.6; // triadic
  return 0.8; // complementary
}

export function colorHarmony(items: ScoreItem[]): number {
  const main = items.map((i) => hexToHsl(i.colors[0] ?? "")).filter((c): c is Hsl => c !== null);
  if (main.length < 2) return 0.8;
  const pairScores = pairs(main).map(([a, b]) => {
    const na = isNeutral(a), nb = isNeutral(b);
    if (na && nb) return 0.9;
    if (na || nb) return 1.0;
    return hueHarmony(a.h, b.h);
  });
  const bright = main.filter((c) => !isNeutral(c) && c.s > 0.6).length;
  return clamp(mean(pairScores) - 0.15 * Math.max(0, bright - 2));
}

export function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
}

/** Mean pairwise CLIP similarity, rescaled from the typical garment range [0.6, 0.9] to [0, 1]. */
export function visualCohesion(items: ScoreItem[], sim: (a: ScoreItem, b: ScoreItem) => number): number | null {
  const withVec = items.filter((i) => i.vector);
  if (withVec.length < 2) return null;
  return clamp((mean(pairs(withVec).map(([a, b]) => sim(a, b))) - 0.6) / 0.3);
}

const STYLE_GROUP: Record<string, "dressy" | "everyday" | "sport"> = {
  formal: "dressy", business: "dressy", ethnic: "dressy",
  casual: "everyday", streetwear: "everyday", minimalist: "everyday", vintage: "everyday", bohemian: "everyday",
  sporty: "sport",
};
const GROUP_COMPAT: Record<string, number> = {
  "dressy|everyday": 0.45, "everyday|sport": 0.6, "dressy|sport": 0.1,
};

function styleCompat(a: string, b: string): number {
  if (a === b) return 1;
  const ga = STYLE_GROUP[a], gb = STYLE_GROUP[b];
  if (!ga || !gb) return 0.5;
  if (ga === gb) return 0.85;
  return GROUP_COMPAT[[ga, gb].sort().join("|")] ?? 0.5;
}

export function styleConsistency(items: ScoreItem[]): number {
  const tagged = items.filter((i) => i.styles.length > 0);
  if (tagged.length < 2) return 0.6;
  return mean(
    pairs(tagged).map(([x, y]) => Math.max(...x.styles.flatMap((a) => y.styles.map((b) => styleCompat(a, b))))),
  );
}

export function styleMatch(items: ScoreItem[], sim: (a: ScoreItem, b: ScoreItem) => number) {
  const color = colorHarmony(items);
  const visual = visualCohesion(items, sim);
  const style = styleConsistency(items);
  const value = visual === null ? (0.4 * color + 0.3 * style) / 0.7 : 0.4 * color + 0.3 * visual + 0.3 * style;
  return { value: clamp(value), color, visual, style };
}

// ---------------------------------------------------------------------------
// WeatherFit: layer warmth vs. temperature, plus rain handling
// ---------------------------------------------------------------------------

/** Target warmth for the upper body (top + outer) and the legs, from "feels like" °C. */
export function targetWarmth(feelsLikeC: number) {
  return {
    upper: clamp(1 + (30 - feelsLikeC) / 3, 1, 9), // 30°C -> 1 (t-shirt), 18°C -> 5, 12°C -> 7 (needs a layer)
    lower: clamp(1 + (30 - feelsLikeC) / 4, 1, 5),
  };
}

const OPEN_SHOES = new Set(["sandals", "flats", "heels"]);

export function weatherFit(items: ScoreItem[], w: WeatherSummary): number {
  const top = items.find((i) => i.category === "TOP");
  const dress = items.find((i) => i.category === "DRESS");
  const bottom = items.find((i) => i.category === "BOTTOM");
  const outer = items.find((i) => i.category === "OUTERWEAR");
  const shoes = items.find((i) => i.category === "FOOTWEAR");

  const target = targetWarmth(w.feelsLikeC);
  const upperBase = top?.warmthLevel ?? dress?.warmthLevel ?? 1;
  const upper = upperBase + (outer ? outer.warmthLevel * 0.8 : 0);
  const lower = bottom?.warmthLevel ?? dress?.warmthLevel ?? 1;
  let fit = 0.6 * gaussian(upper - target.upper, 1.5) + 0.4 * gaussian(lower - target.lower, 1.2);

  if (shoes && OPEN_SHOES.has(shoes.itemType) && (w.rainy || w.feelsLikeC < 18)) fit *= 0.85;
  if (w.rainy) {
    const protectedFromRain = items.some((i) => i.waterResistant && (i.category === "OUTERWEAR" || i.category === "FOOTWEAR"));
    fit = protectedFromRain ? fit + 0.1 : fit * 0.9;
  }
  return clamp(fit);
}

// ---------------------------------------------------------------------------
// CalendarFit: how well each item's style and type suit the day's event type
// ---------------------------------------------------------------------------

const STYLE_FOR_EVENT: Record<EventType, Record<string, number>> = {
  formal: { formal: 1, business: 1, ethnic: 0.8, minimalist: 0.6, vintage: 0.4, casual: 0.3, bohemian: 0.3, streetwear: 0.15, sporty: 0 },
  casual: { casual: 1, streetwear: 0.9, minimalist: 0.9, vintage: 0.9, bohemian: 0.85, sporty: 0.6, ethnic: 0.6, formal: 0.5, business: 0.5 },
  sport: { sporty: 1, casual: 0.4, streetwear: 0.4, minimalist: 0.3, vintage: 0.1, bohemian: 0.1, ethnic: 0.1, formal: 0, business: 0 },
};

const TYPE_FOR_EVENT: Record<EventType, Record<string, number>> = {
  formal: {
    shirt: 1, blazer: 1, trousers: 1, "formal shoes": 1, "formal leather shoes": 1, dress: 0.9, heels: 0.9,
    kurta: 0.8, skirt: 0.8, flats: 0.8, jacket: 0.7, top: 0.7, blouse: 0.8, "polo shirt": 0.6, sweater: 0.6,
    cardigan: 0.6, coat: 0.8, jeans: 0.4, sneakers: 0.3, "t-shirt": 0.2, sweatshirt: 0.2, leggings: 0.2,
    "tank top": 0.1, hoodie: 0.1, sandals: 0.1, shorts: 0, "track pants": 0, sweatpants: 0,
  },
  sport: {
    "t-shirt": 1, "tank top": 1, "track pants": 1, sweatpants: 1, shorts: 1, leggings: 1, sneakers: 1,
    sweatshirt: 0.7, hoodie: 0.7, jacket: 0.5, top: 0.4, trousers: 0.2, skirt: 0.2, flats: 0.2, sandals: 0.2,
    jeans: 0.1, shirt: 0.1, dress: 0.1, blazer: 0, "formal shoes": 0, "formal leather shoes": 0, heels: 0,
  },
  casual: {
    "t-shirt": 1, jeans: 1, sneakers: 1, shirt: 0.9, "polo shirt": 0.9, shorts: 0.9, top: 0.9, dress: 0.9,
    skirt: 0.9, sandals: 0.9, sweatshirt: 0.9, hoodie: 0.9, sweater: 0.9, jacket: 0.9, flats: 0.9,
    trousers: 0.8, leggings: 0.7, "track pants": 0.6, heels: 0.6, blazer: 0.6, "formal shoes": 0.5,
  },
};

export function itemCalendarFit(item: ScoreItem, event: EventType): number {
  const styleScore = item.styles.length ? Math.max(...item.styles.map((s) => STYLE_FOR_EVENT[event][s] ?? 0.5)) : 0.5;
  const typeScore = TYPE_FOR_EVENT[event][item.itemType] ?? 0.6;
  return 0.5 * styleScore + 0.5 * typeScore;
}

export function calendarFit(items: ScoreItem[], event: EventType): number {
  return mean(items.filter((i) => i.category !== "ACCESSORY").map((i) => itemCalendarFit(i, event)));
}

// ---------------------------------------------------------------------------
// RotationWeight: prefer items that haven't been worn recently / often
// ---------------------------------------------------------------------------

const NEVER_WORN_DAYS = 30;

export function itemRotation(item: ScoreItem, now: Date, maxWearCount: number): number {
  const days = item.lastWornAt ? Math.max(0, (now.getTime() - item.lastWornAt.getTime()) / 86_400_000) : NEVER_WORN_DAYS;
  const recency = 1 - Math.exp(-days / 7); // worn yesterday ~0.13, a week ago ~0.63, a month ago ~0.99
  const frequency = 1 - item.wearCount / Math.max(1, maxWearCount);
  return 0.7 * recency + 0.3 * frequency;
}

export function rotationWeight(items: ScoreItem[], now: Date, maxWearCount: number): number {
  return mean(items.map((i) => itemRotation(i, now, maxWearCount)));
}

// ---------------------------------------------------------------------------
// Total
// ---------------------------------------------------------------------------

/** Weights actually used: drops unavailable components and renormalizes to sum 1. */
export function effectiveWeights(weights: Weights, available: Record<keyof Components, boolean>): Partial<Weights> {
  const keys = (Object.keys(weights) as Array<keyof Components>).filter((k) => available[k] && weights[k] > 0);
  const sum = keys.reduce((s, k) => s + weights[k], 0);
  return Object.fromEntries(keys.map((k) => [k, sum ? weights[k] / sum : 1 / keys.length]));
}

export function totalScore(c: Components, w: Partial<Weights>): number {
  let total = 0;
  for (const [k, weight] of Object.entries(w) as Array<[keyof Components, number]>) total += weight * (c[k] ?? 0);
  return total;
}
