import Fastify from "fastify";
import { healthRoutes } from "./routes/health.js";

const port = Number(process.env.PORT ?? process.env.API_GATEWAY_PORT ?? 3000);
const host = process.env.HOST ?? "0.0.0.0";

const app = Fastify({ logger: true });

await app.register(healthRoutes);

try {
  await app.listen({ port, host });
} catch (err) {
  app.log.error(err);
  process.exit(1);
}
