import { config } from "../config.js";

/** Fetch stored CLIP vectors from Qdrant by point id. Returns an empty map if Qdrant is unavailable. */
export async function fetchVectors(ids: string[]): Promise<Map<string, number[]>> {
  const out = new Map<string, number[]>();
  if (ids.length === 0) return out;
  try {
    const res = await fetch(`${config.qdrantUrl}/collections/${config.qdrantCollection}/points`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(config.qdrantApiKey && { "api-key": config.qdrantApiKey }) },
      body: JSON.stringify({ ids, with_vector: true, with_payload: false }),
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return out;
    const body = (await res.json()) as { result?: Array<{ id: string; vector?: number[] }> };
    for (const p of body.result ?? []) if (Array.isArray(p.vector)) out.set(String(p.id), p.vector);
  } catch {
    /* StyleMatch falls back to color + style rules */
  }
  return out;
}
