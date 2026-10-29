import type { FastifyInstance } from "fastify";
import { z } from "zod";
import { HttpError } from "../lib/errors.js";
import { prisma } from "../lib/prisma.js";

const createUserBody = z.object({
  email: z.email(),
  displayName: z.string().min(1).max(80).optional(),
});

/** Dev-only helpers, registered only when DEV_AUTH is on. */
export async function devRoutes(app: FastifyInstance): Promise<void> {
  app.post("/dev/users", async (req, reply) => {
    const parsed = createUserBody.safeParse(req.body);
    if (!parsed.success) throw new HttpError(400, parsed.error.issues[0]?.message ?? "Invalid body");
    const user = await prisma.user.upsert({
      where: { email: parsed.data.email },
      update: {},
      create: parsed.data,
      select: { id: true, email: true, displayName: true },
    });
    return reply.code(201).send(user);
  });
}
