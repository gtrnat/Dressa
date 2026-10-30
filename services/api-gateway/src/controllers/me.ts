import type { FastifyRequest } from "fastify";
import { z } from "zod";
import { prisma } from "../lib/prisma.js";
import { isValidTimeZone } from "../lib/time.js";
import { parse } from "../lib/validate.js";

const profileSelect = {
  id: true, email: true, displayName: true, timezone: true, latitude: true, longitude: true,
} as const;

const updateBody = z
  .object({
    displayName: z.string().trim().min(1).max(80).nullable(),
    timezone: z.string().refine(isValidTimeZone, "Unknown IANA timezone, e.g. Asia/Jakarta"),
    latitude: z.number().min(-90).max(90).nullable(),
    longitude: z.number().min(-180).max(180).nullable(),
  })
  .partial()
  .strict()
  .refine((b) => (b.latitude === undefined) === (b.longitude === undefined), "Send latitude and longitude together")
  .refine((b) => Object.keys(b).length > 0, "Body must contain at least one field");

export async function getMe(req: FastifyRequest) {
  return prisma.user.findUniqueOrThrow({ where: { id: req.userId }, select: profileSelect });
}

export async function updateMe(req: FastifyRequest) {
  const data = parse(updateBody, req.body);
  return prisma.user.update({ where: { id: req.userId }, data, select: profileSelect });
}
