import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { config } from "../config.js";

/**
 * Local-disk image storage, served at /uploads/*.
 * Swap for S3/R2/Supabase Storage later by keeping the same two functions.
 * Keys are generated server-side (userId/itemId), never from user input.
 */
export const storage = {
  async put(key: string, data: Buffer): Promise<string> {
    const target = path.join(config.uploadDir, key);
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, data);
    return `${config.publicBaseUrl}/uploads/${key}`;
  },

  async removePrefix(prefix: string): Promise<void> {
    await rm(path.join(config.uploadDir, prefix), { recursive: true, force: true });
  },
};

export function itemPrefix(userId: string, itemId: string): string {
  return `${userId}/${itemId}`;
}
