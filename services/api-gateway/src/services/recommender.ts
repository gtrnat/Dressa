import type { WardrobeItem } from "@prisma/client";
import { config } from "../config.js";
import { HttpError } from "../lib/errors.js";
import { prisma } from "../lib/prisma.js";
import { daysBetween } from "../lib/time.js";
import { type ClassifiedEvent, type EventType, dominantEventType } from "./event-classifier.js";
import { getDayEvents } from "./google-calendar.js";
import {
  type Components,
  type ScoreItem,
  type Weights,
  calendarFit,
  cosine,
  effectiveWeights,
  itemCalendarFit,
  itemRotation,
  rotationWeight,
  styleMatch,
  targetWarmth,
  totalScore,
  weatherFit,
} from "./scoring.js";
import { fetchVectors } from "./vectors.js";
import { type WeatherSummary, getDailyWeather } from "./weather.js";

const PER_CATEGORY_LIMIT = 12; // candidates kept per category before combining
const MAX_OVERLAP = 0.5; // alternatives may share at most half of their items
const OUTER_MIN_TARGET = 3; // targetWarmth.upper >= 3 means feels-like <= 24°C
const MAX_ROTATION_REASONS = 2;

export interface RecommendOptions {
  date: string;
  timeZone: string;
  lat?: number;
  lon?: number;
  eventType?: EventType; // manual override (no calendar needed)
  count: number;
  reroll: number;
  weights?: Partial<Weights>;
}

interface Context {
  weather: WeatherSummary | null;
  events: ClassifiedEvent[] | null;
  eventType: EventType | null;
  calendarSource: "google" | "manual" | null;
  unavailable: Partial<Record<keyof Components, string>>;
}

async function buildContext(userId: string, opts: RecommendOptions): Promise<Context> {
  const unavailable: Context["unavailable"] = {};

  let weather: WeatherSummary | null = null;
  if (opts.lat === undefined || opts.lon === undefined) {
    unavailable.weatherFit = "Lokasi belum diatur. Isi latitude/longitude lewat PATCH /me atau query lat & lon.";
  } else {
    try {
      weather = await getDailyWeather(opts.lat, opts.lon, opts.date, opts.timeZone);
    } catch (err) {
      unavailable.weatherFit = err instanceof HttpError ? err.message : "Data cuaca tidak tersedia";
    }
  }

  let events: ClassifiedEvent[] | null = null;
  let eventType: EventType | null = null;
  let calendarSource: Context["calendarSource"] = null;
  if (opts.eventType) {
    eventType = opts.eventType;
    calendarSource = "manual";
  } else {
    try {
      events = await getDayEvents(userId, opts.date, opts.timeZone);
      if (events === null) unavailable.calendarFit = "Google Calendar belum terhubung (opsional).";
      else {
        eventType = dominantEventType(events);
        calendarSource = "google";
        if (!eventType) unavailable.calendarFit = "Tidak ada agenda di tanggal ini.";
      }
    } catch (err) {
      unavailable.calendarFit = err instanceof HttpError ? err.message : "Kalender tidak dapat dibaca";
    }
  }
  return { weather, events, eventType, calendarSource, unavailable };
}

function toScoreItem(i: WardrobeItem, vectors: Map<string, number[]>): ScoreItem {
  return {
    id: i.id,
    category: i.category,
    itemType: i.itemType,
    colors: i.colors,
    styles: i.styles,
    warmthLevel: i.warmthLevel,
    waterResistant: i.waterResistant,
    wearCount: i.wearCount,
    lastWornAt: i.lastWornAt,
    vector: vectors.get(i.qdrantVectorId ?? i.id),
  };
}

function* combinations(groups: { bases: ScoreItem[][]; shoes: Array<ScoreItem | null>; outers: Array<ScoreItem | null> }) {
  for (const base of groups.bases)
    for (const shoe of groups.shoes)
      for (const outer of groups.outers) {
        const items = [...base];
        if (outer) items.push(outer);
        if (shoe) items.push(shoe);
        yield items;
      }
}

function overlap(a: ScoreItem[], b: ScoreItem[]): number {
  const ids = new Set(a.map((i) => i.id));
  return b.filter((i) => ids.has(i.id)).length / Math.max(a.length, b.length);
}

function reasons(items: ScoreItem[], c: Components, ctx: Context, now: Date, maxWear: number): string[] {
  const out: string[] = [];
  const w = ctx.weather;
  const outer = items.find((i) => i.category === "OUTERWEAR");
  if (w) {
    const t = targetWarmth(w.feelsLikeC).upper;
    out.push(`Terasa ${Math.round(w.feelsLikeC)}°C (${w.condition})${t <= 2 ? ": pakaian ringan" : t >= 5 ? ": butuh pakaian hangat" : ""}`);
    if (outer) out.push(`${outer.itemType} ditambahkan karena ${w.rainy ? "kemungkinan hujan" : "cuaca sejuk"}`);
    if (w.rainy && items.some((i) => i.waterResistant)) out.push("Ada item tahan air untuk hujan");
  }
  if (ctx.eventType && c.calendarFit !== null && c.calendarFit >= 0.7) out.push(`Cocok untuk agenda ${ctx.eventType}`);
  if (c.styleMatch >= 0.8) out.push("Warna dan gaya serasi");
  const rarelyWorn = items
    .map((i) => ({ i, r: itemRotation(i, now, maxWear) }))
    .filter((x) => x.r >= 0.75)
    .sort((a, b) => b.r - a.r)
    .slice(0, MAX_ROTATION_REASONS);
  for (const { i } of rarelyWorn) {
    out.push(i.lastWornAt ? `${i.itemType} terakhir dipakai ${daysBetween(i.lastWornAt, now)} hari lalu` : `${i.itemType} belum pernah dipakai`);
  }
  return out;
}

export async function recommend(userId: string, opts: RecommendOptions) {
  const rows = await prisma.wardrobeItem.findMany({ where: { userId, archived: false } });
  const byCat = (c: string) => rows.filter((r) => r.category === c);
  const tops = byCat("TOP"), bottoms = byCat("BOTTOM"), dresses = byCat("DRESS");
  if (dresses.length === 0 && (tops.length === 0 || bottoms.length === 0)) {
    throw new HttpError(422, "Tambahkan minimal 1 atasan dan 1 bawahan (atau 1 dress) untuk mendapat rekomendasi");
  }

  const [ctx, vectors] = await Promise.all([
    buildContext(userId, opts),
    fetchVectors(rows.map((r) => r.qdrantVectorId ?? r.id)),
  ]);
  const now = new Date();
  const maxWear = Math.max(1, ...rows.map((r) => r.wearCount));
  const items = new Map(rows.map((r) => [r.id, toScoreItem(r, vectors)]));
  const get = (rs: WardrobeItem[]) => rs.map((r) => items.get(r.id)!);

  // Keep the most promising items per category so the number of combinations stays small.
  const prescore = (i: ScoreItem) =>
    itemRotation(i, now, maxWear) + (ctx.eventType ? itemCalendarFit(i, ctx.eventType) : 0);
  const shortlist = (xs: ScoreItem[]) => [...xs].sort((a, b) => prescore(b) - prescore(a)).slice(0, PER_CATEGORY_LIMIT);

  const T = shortlist(get(tops)), B = shortlist(get(bottoms));
  const bases: ScoreItem[][] = [...T.flatMap((t) => B.map((b) => [t, b])), ...shortlist(get(dresses)).map((d) => [d])];
  const shoesList = shortlist(get(byCat("FOOTWEAR")));
  // Outerwear only when it can help: weather unknown, rain, or cool enough (feels like <= ~24°C).
  const outerUseful = !ctx.weather || ctx.weather.rainy || targetWarmth(ctx.weather.feelsLikeC).upper >= OUTER_MIN_TARGET;
  const outersList = outerUseful ? shortlist(get(byCat("OUTERWEAR"))) : [];

  const simCache = new Map<string, number>();
  const sim = (a: ScoreItem, b: ScoreItem) => {
    const key = a.id < b.id ? `${a.id}|${b.id}` : `${b.id}|${a.id}`;
    let v = simCache.get(key);
    if (v === undefined) simCache.set(key, (v = cosine(a.vector!, b.vector!)));
    return v;
  };

  const weights = effectiveWeights(
    { ...config.scoreWeights, ...opts.weights },
    { styleMatch: true, rotationWeight: true, weatherFit: ctx.weather !== null, calendarFit: ctx.eventType !== null },
  );

  const scored = [];
  for (const outfit of combinations({
    bases,
    shoes: shoesList.length ? shoesList : [null],
    outers: [null, ...outersList],
  })) {
    const sm = styleMatch(outfit, sim);
    const components: Components = {
      styleMatch: sm.value,
      weatherFit: ctx.weather ? weatherFit(outfit, ctx.weather) : null,
      calendarFit: ctx.eventType ? calendarFit(outfit, ctx.eventType) : null,
      rotationWeight: rotationWeight(outfit, now, maxWear),
    };
    scored.push({ outfit, components, detail: sm, total: totalScore(components, weights) });
  }
  scored.sort((a, b) => b.total - a.total);

  // Greedy diverse selection: each alternative must differ enough from the ones before it.
  const needed = (opts.reroll + 1) * opts.count;
  const diverse: typeof scored = [];
  for (const s of scored) {
    if (diverse.every((d) => overlap(d.outfit, s.outfit) <= MAX_OVERLAP)) diverse.push(s);
    if (diverse.length > needed) break;
  }
  const page = diverse.slice(opts.reroll * opts.count, needed);

  const round = (x: number | null) => (x === null ? null : Math.round(x * 1000) / 1000);
  const rowById = new Map(rows.map((r) => [r.id, r]));
  return {
    date: opts.date,
    context: {
      weather: ctx.weather,
      calendar: { source: ctx.calendarSource, eventType: ctx.eventType, events: ctx.events ?? [] },
      unavailable: ctx.unavailable,
    },
    weights: Object.fromEntries(Object.entries(weights).map(([k, v]) => [k, round(v)])),
    reroll: opts.reroll,
    hasMore: diverse.length > needed,
    candidatesEvaluated: scored.length,
    outfits: page.map((s) => ({
      score: {
        total: round(s.total),
        styleMatch: round(s.components.styleMatch),
        weatherFit: round(s.components.weatherFit),
        calendarFit: round(s.components.calendarFit),
        rotationWeight: round(s.components.rotationWeight),
        styleDetail: { color: round(s.detail.color), visual: round(s.detail.visual), style: round(s.detail.style) },
      },
      reasons: reasons(s.outfit, s.components, ctx, now, maxWear),
      items: s.outfit.map((i) => {
        const r = rowById.get(i.id)!;
        return {
          id: r.id,
          category: r.category,
          itemType: r.itemType,
          colors: r.colors,
          styles: r.styles,
          warmthLevel: r.warmthLevel,
          maskedImageUrl: r.maskedImageUrl,
          lastWornAt: r.lastWornAt,
        };
      }),
    })),
  };
}
