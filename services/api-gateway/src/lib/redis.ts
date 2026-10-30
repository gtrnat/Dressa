import { createClient } from "redis";
import { config } from "../config.js";

/**
 * Best-effort JSON cache. If Redis is down, every call becomes a cache miss
 * instead of failing the request.
 */
const newClient = () =>
  createClient({ url: config.redisUrl, socket: { connectTimeout: 2000, reconnectStrategy: false } });
type Client = ReturnType<typeof newClient>;
let client: Client | null = null;
let connecting: Promise<Client | null> | null = null;
let lastFailure = 0;
const RETRY_AFTER_MS = 30_000;

async function getClient(): Promise<Client | null> {
  if (client?.isReady) return client;
  if (Date.now() - lastFailure < RETRY_AFTER_MS) return null;
  connecting ??= (async () => {
    try {
      const c = newClient();
      c.on("error", () => {}); // reported via the failed command instead
      await c.connect();
      client = c;
      return c;
    } catch {
      lastFailure = Date.now();
      return null;
    } finally {
      connecting = null;
    }
  })();
  return connecting;
}

export const cache = {
  async get<T>(key: string): Promise<T | null> {
    try {
      const raw = await (await getClient())?.get(key);
      return raw ? (JSON.parse(raw) as T) : null;
    } catch {
      return null;
    }
  },

  async set(key: string, value: unknown, ttlSec: number): Promise<void> {
    try {
      await (await getClient())?.set(key, JSON.stringify(value), { EX: Math.max(1, Math.floor(ttlSec)) });
    } catch {
      /* cache is optional */
    }
  },

  async del(...keys: string[]): Promise<void> {
    try {
      if (keys.length) await (await getClient())?.del(keys);
    } catch {
      /* cache is optional */
    }
  },

  async close(): Promise<void> {
    await client?.quit().catch(() => {});
    client = null;
  },
};
