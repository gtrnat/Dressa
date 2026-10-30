import type { FastifyReply, FastifyRequest } from "fastify";
import { z } from "zod";
import { prisma } from "../lib/prisma.js";
import { parse } from "../lib/validate.js";
import { EVENT_TYPES } from "../services/event-classifier.js";
import { deleteLog, listLogs, logWear } from "../services/outfits.js";

const score01 = z.number().min(0).max(1);

const wearBody = z
  .object({
    itemIds: z.array(z.uuid()).min(1).max(8),
    wornOn: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "wornOn must be YYYY-MM-DD").optional(),
    recommended: z.boolean().default(false),
    eventType: z.enum(EVENT_TYPES as [string, ...string[]]).optional(),
    score: score01.optional(),
    scoreDetail: z
      .object({ styleMatch: score01, weatherFit: score01.nullable(), calendarFit: score01.nullable(), rotationWeight: score01 })
      .partial()
      .strict()
      .optional(),
    weather: z
      .object({
        feelsLikeC: z.number(), tempMinC: z.number(), tempMaxC: z.number(),
        precipProbability: score01, rainMm: z.number().min(0), condition: z.string().max(60),
      })
      .partial()
      .strict()
      .optional(),
  })
  .strict();

const listQuery = z.object({
  limit: z.coerce.number().int().min(1).max(100).default(30),
  offset: z.coerce.number().int().min(0).default(0),
});

async function userTimeZone(userId: string): Promise<string> {
  const u = await prisma.user.findUniqueOrThrow({ where: { id: userId }, select: { timezone: true } });
  return u.timezone;
}

export async function wearOutfit(req: FastifyRequest, reply: FastifyReply) {
  const body = parse(wearBody, req.body);
  const log = await logWear(req.userId, await userTimeZone(req.userId), body);
  return reply.code(201).send(log);
}

export async function listOutfits(req: FastifyRequest) {
  const q = parse(listQuery, req.query);
  return listLogs(req.userId, q.limit, q.offset);
}

export async function removeOutfit(req: FastifyRequest, reply: FastifyReply) {
  const { id } = parse(z.object({ id: z.uuid() }), req.params);
  await deleteLog(req.userId, id, await userTimeZone(req.userId));
  return reply.code(204).send();
}
