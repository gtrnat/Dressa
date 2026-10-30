import type { FastifyRequest } from "fastify";
import { z } from "zod";
import { HttpError } from "../lib/errors.js";
import { prisma } from "../lib/prisma.js";
import { addDays, todayIn } from "../lib/time.js";
import { parse } from "../lib/validate.js";
import { EVENT_TYPES } from "../services/event-classifier.js";
import { recommend } from "../services/recommender.js";

const weight = z.coerce.number().min(0).max(1).optional();

const query = z
  .object({
    date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "date must be YYYY-MM-DD").optional(),
    lat: z.coerce.number().min(-90).max(90).optional(),
    lon: z.coerce.number().min(-180).max(180).optional(),
    eventType: z.enum(EVENT_TYPES as [string, ...string[]]).optional(),
    count: z.coerce.number().int().min(1).max(5).default(3),
    reroll: z.coerce.number().int().min(0).max(20).default(0),
    // Optional per-request weight overrides, handy for experiments
    wStyle: weight,
    wWeather: weight,
    wCalendar: weight,
    wRotation: weight,
  })
  .refine((q) => (q.lat === undefined) === (q.lon === undefined), "Send lat and lon together");

export async function getRecommendations(req: FastifyRequest) {
  const q = parse(query, req.query);
  const user = await prisma.user.findUniqueOrThrow({
    where: { id: req.userId },
    select: { timezone: true, latitude: true, longitude: true },
  });

  const today = todayIn(user.timezone);
  const date = q.date ?? today;
  if (date < today || date > addDays(today, 4)) {
    throw new HttpError(400, `date must be between ${today} and ${addDays(today, 4)} (weather forecast range)`);
  }

  const weights = Object.fromEntries(
    Object.entries({ styleMatch: q.wStyle, weatherFit: q.wWeather, calendarFit: q.wCalendar, rotationWeight: q.wRotation })
      .filter(([, v]) => v !== undefined),
  );

  return recommend(req.userId, {
    date,
    timeZone: user.timezone,
    lat: q.lat ?? user.latitude ?? undefined,
    lon: q.lon ?? user.longitude ?? undefined,
    eventType: q.eventType as (typeof EVENT_TYPES)[number] | undefined,
    count: q.count,
    reroll: q.reroll,
    weights,
  });
}
