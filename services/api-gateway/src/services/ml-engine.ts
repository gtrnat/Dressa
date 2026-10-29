import type { ProcessImageResponse } from "@dressa/shared-types";
import { config } from "../config.js";
import { HttpError } from "../lib/errors.js";

const TIMEOUT_MS = 60_000; // background removal takes several seconds on CPU

async function call<T>(path: string, init: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${config.mlEngineUrl}${path}`, { ...init, signal: AbortSignal.timeout(TIMEOUT_MS) });
  } catch (err) {
    throw new HttpError(503, `ML engine unreachable: ${(err as Error).message}`);
  }
  if (res.ok) {
    return (res.status === 204 ? undefined : await res.json()) as T;
  }
  const body = (await res.json().catch(() => ({}))) as { detail?: unknown };
  const detail = typeof body.detail === "string" ? body.detail : `ML engine error (${res.status})`;
  // 4xx from the ML engine are caused by the uploaded image; pass them through.
  // Anything else is our problem, so report a bad gateway.
  if (res.status === 400 || res.status === 413 || res.status === 415 || res.status === 422) {
    throw new HttpError(res.status, detail);
  }
  throw new HttpError(502, detail);
}

function imageForm(image: Buffer, mimeType: string, filename: string, fields: Record<string, string> = {}): FormData {
  const form = new FormData();
  for (const [k, v] of Object.entries(fields)) form.append(k, v);
  form.append("file", new Blob([new Uint8Array(image)], { type: mimeType }), filename);
  return form;
}

export const mlEngine = {
  processImage(image: Buffer, mimeType: string, filename: string): Promise<ProcessImageResponse> {
    return call("/process-image", { method: "POST", body: imageForm(image, mimeType, filename) });
  },

  /** Embeds the image and upserts it to Qdrant under point id = itemId. */
  async storeEmbedding(image: Buffer, itemId: string, userId: string, category: string): Promise<void> {
    await call("/embed", {
      method: "POST",
      body: imageForm(image, "image/png", `${itemId}.png`, { item_id: itemId, user_id: userId, category }),
    });
  },

  async deleteEmbedding(itemId: string, userId: string): Promise<void> {
    try {
      await call(`/items/${itemId}?user_id=${userId}`, { method: "DELETE" });
    } catch (err) {
      if (err instanceof HttpError && err.statusCode === 404) return; // already gone
      throw err;
    }
  },
};
