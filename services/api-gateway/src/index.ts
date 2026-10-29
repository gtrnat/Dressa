import { mkdir } from "node:fs/promises";
import multipart from "@fastify/multipart";
import fastifyStatic from "@fastify/static";
import Fastify, { type FastifyError } from "fastify";
import { config } from "./config.js";
import { HttpError } from "./lib/errors.js";
import { prisma } from "./lib/prisma.js";
import { devRoutes } from "./routes/dev.js";
import { healthRoutes } from "./routes/health.js";
import { itemRoutes } from "./routes/items.js";

const app = Fastify({ logger: true, bodyLimit: 1024 * 1024 });

app.setErrorHandler((err: FastifyError | HttpError, req, reply) => {
  const status = err.statusCode ?? 500;
  if (status >= 500) req.log.error(err);
  // Don't leak internal error details to clients.
  const message = status >= 500 && !(err instanceof HttpError) ? "Internal server error" : err.message;
  return reply.code(status).send({ error: message });
});

await mkdir(config.uploadDir, { recursive: true });
await app.register(multipart, { limits: { fileSize: config.maxUploadBytes, files: 1, fields: 5 } });
await app.register(fastifyStatic, { root: config.uploadDir, prefix: "/uploads/", decorateReply: false });

await app.register(healthRoutes);
await app.register(itemRoutes);
if (config.devAuth) {
  app.log.warn("DEV_AUTH is on: X-User-Id header is trusted. Do not use in production.");
  await app.register(devRoutes);
}

const shutdown = async () => {
  await app.close();
  await prisma.$disconnect();
  process.exit(0);
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

try {
  await app.listen({ port: config.port, host: config.host });
} catch (err) {
  app.log.error(err);
  process.exit(1);
}
