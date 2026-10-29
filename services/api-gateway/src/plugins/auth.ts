import type { FastifyReply, FastifyRequest } from "fastify";
import { z } from "zod";
import { config } from "../config.js";
import { HttpError } from "../lib/errors.js";
import { prisma } from "../lib/prisma.js";

declare module "fastify" {
  interface FastifyRequest {
    userId: string;
  }
}

const userIdHeader = z.uuid();

/**
 * TEMPORARY dev auth: the caller says who they are via the X-User-Id header.
 * Anyone can impersonate anyone, so this is disabled when NODE_ENV=production.
 * Replace with real auth (JWT / Supabase Auth) before any deployment.
 */
export async function requireUser(req: FastifyRequest, _reply: FastifyReply): Promise<void> {
  if (!config.devAuth) {
    throw new HttpError(501, "Authentication is not implemented yet");
  }
  const parsed = userIdHeader.safeParse(req.headers["x-user-id"]);
  if (!parsed.success) {
    throw new HttpError(401, "Missing or invalid X-User-Id header (dev auth)");
  }
  const user = await prisma.user.findUnique({ where: { id: parsed.data }, select: { id: true } });
  if (!user) {
    throw new HttpError(401, "Unknown user");
  }
  req.userId = user.id;
}
