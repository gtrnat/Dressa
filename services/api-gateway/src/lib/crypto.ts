import { createCipheriv, createDecipheriv, createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { HttpError } from "./errors.js";

function key(base64: string | undefined, name: string): Buffer {
  const buf = base64 ? Buffer.from(base64, "base64") : Buffer.alloc(0);
  if (buf.length !== 32) {
    throw new HttpError(501, `${name} is not configured (needs 32 random bytes, base64)`);
  }
  return buf;
}

/** AES-256-GCM. Output: base64(iv | tag | ciphertext). */
export function encrypt(plain: string, keyB64: string | undefined): string {
  const iv = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", key(keyB64, "TOKEN_ENCRYPTION_KEY"), iv);
  const data = Buffer.concat([cipher.update(plain, "utf8"), cipher.final()]);
  return Buffer.concat([iv, cipher.getAuthTag(), data]).toString("base64");
}

export function decrypt(payload: string, keyB64: string | undefined): string {
  const buf = Buffer.from(payload, "base64");
  const decipher = createDecipheriv("aes-256-gcm", key(keyB64, "TOKEN_ENCRYPTION_KEY"), buf.subarray(0, 12));
  decipher.setAuthTag(buf.subarray(12, 28));
  return Buffer.concat([decipher.update(buf.subarray(28)), decipher.final()]).toString("utf8");
}

/** Signed, expiring token: base64url(json).base64url(hmac). Used as the OAuth state. */
export function signToken(payload: object, secretB64: string | undefined, ttlSec: number): string {
  const body = Buffer.from(
    JSON.stringify({ ...payload, exp: Math.floor(Date.now() / 1000) + ttlSec, n: randomBytes(8).toString("hex") }),
  ).toString("base64url");
  const sig = createHmac("sha256", key(secretB64, "OAUTH_STATE_SECRET")).update(body).digest("base64url");
  return `${body}.${sig}`;
}

export function verifyToken<T>(token: string, secretB64: string | undefined): T | null {
  const [body, sig] = token.split(".");
  if (!body || !sig) return null;
  const expected = createHmac("sha256", key(secretB64, "OAUTH_STATE_SECRET")).update(body).digest();
  const given = Buffer.from(sig, "base64url");
  if (given.length !== expected.length || !timingSafeEqual(given, expected)) return null;
  try {
    const data = JSON.parse(Buffer.from(body, "base64url").toString("utf8")) as T & { exp: number };
    return data.exp > Date.now() / 1000 ? data : null;
  } catch {
    return null;
  }
}
