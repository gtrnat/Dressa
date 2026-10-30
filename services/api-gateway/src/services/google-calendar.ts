import { config } from "../config.js";
import { decrypt, encrypt, signToken, verifyToken } from "../lib/crypto.js";
import { HttpError } from "../lib/errors.js";
import { prisma } from "../lib/prisma.js";
import { cache } from "../lib/redis.js";
import { dayRange } from "../lib/time.js";
import { type ClassifiedEvent, classifyEvent } from "./event-classifier.js";

const AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN_URL = "https://oauth2.googleapis.com/token";
const REVOKE_URL = "https://oauth2.googleapis.com/revoke";
const EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars";
// Least privilege: read-only access to events, nothing else.
const SCOPE = "https://www.googleapis.com/auth/calendar.events.readonly";
const STATE_TTL_SEC = 10 * 60;
const EVENTS_CACHE_TTL_SEC = 10 * 60;

function requireConfigured(): { clientId: string; clientSecret: string } {
  const { clientId, clientSecret } = config.google;
  if (!clientId || !clientSecret || !config.tokenEncryptionKey || !config.oauthStateSecret) {
    throw new HttpError(
      501,
      "Google Calendar is not configured (set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, TOKEN_ENCRYPTION_KEY, OAUTH_STATE_SECRET)",
    );
  }
  return { clientId, clientSecret };
}

export function isGoogleConfigured(): boolean {
  try {
    requireConfigured();
    return true;
  } catch {
    return false;
  }
}

export function buildAuthUrl(userId: string): string {
  const { clientId } = requireConfigured();
  // The state binds the callback to this user and expires; it's HMAC-signed so it can't be forged.
  const state = signToken({ uid: userId }, config.oauthStateSecret, STATE_TTL_SEC);
  const url = new URL(AUTH_URL);
  url.search = new URLSearchParams({
    client_id: clientId,
    redirect_uri: config.google.redirectUri,
    response_type: "code",
    scope: SCOPE,
    access_type: "offline", // returns a refresh token
    prompt: "consent", // always return a refresh token, even on reconnect
    include_granted_scopes: "true",
    state,
  }).toString();
  return url.toString();
}

interface TokenResponse {
  access_token: string;
  expires_in: number;
  refresh_token?: string;
  scope: string;
  error?: string;
  error_description?: string;
}

async function tokenRequest(params: Record<string, string>): Promise<TokenResponse> {
  let res: Response;
  try {
    res = await fetch(TOKEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams(params),
      signal: AbortSignal.timeout(10_000),
    });
  } catch (err) {
    throw new HttpError(503, `Google unreachable: ${(err as Error).message}`);
  }
  const body = (await res.json().catch(() => ({}))) as TokenResponse;
  if (!res.ok) {
    if (body.error === "invalid_grant") throw new HttpError(401, "Google access was revoked or expired; reconnect the calendar");
    throw new HttpError(502, `Google token error: ${body.error ?? res.status}`);
  }
  return body;
}

/** OAuth callback: verify state, exchange the code, store the encrypted refresh token. */
export async function handleCallback(code: string, state: string): Promise<string> {
  const { clientId, clientSecret } = requireConfigured();
  const payload = verifyToken<{ uid: string }>(state, config.oauthStateSecret);
  if (!payload) throw new HttpError(400, "Invalid or expired OAuth state; start the connection again");

  const user = await prisma.user.findUnique({ where: { id: payload.uid }, select: { id: true } });
  if (!user) throw new HttpError(400, "Unknown user");

  const tokens = await tokenRequest({
    code,
    client_id: clientId,
    client_secret: clientSecret,
    redirect_uri: config.google.redirectUri,
    grant_type: "authorization_code",
  });
  if (!tokens.refresh_token) throw new HttpError(502, "Google did not return a refresh token; remove the app's access in your Google account and try again");
  if (!tokens.scope.split(" ").includes(SCOPE)) throw new HttpError(400, "Calendar permission was not granted");

  const refreshTokenEnc = encrypt(tokens.refresh_token, config.tokenEncryptionKey);
  await prisma.calendarConnection.upsert({
    where: { userId: user.id },
    create: { userId: user.id, refreshTokenEnc, scope: tokens.scope },
    update: { refreshTokenEnc, scope: tokens.scope },
  });
  await cache.set(`gcal:token:${user.id}`, tokens.access_token, tokens.expires_in - 60);
  return user.id;
}

async function accessToken(userId: string, refreshTokenEnc: string): Promise<string> {
  const cached = await cache.get<string>(`gcal:token:${userId}`);
  if (cached) return cached;
  const { clientId, clientSecret } = requireConfigured();
  const tokens = await tokenRequest({
    refresh_token: decrypt(refreshTokenEnc, config.tokenEncryptionKey),
    client_id: clientId,
    client_secret: clientSecret,
    grant_type: "refresh_token",
  });
  await cache.set(`gcal:token:${userId}`, tokens.access_token, tokens.expires_in - 60);
  return tokens.access_token;
}

interface GoogleEvent {
  status?: string;
  summary?: string;
  description?: string;
  transparency?: string;
  start?: { dateTime?: string; date?: string };
  end?: { dateTime?: string; date?: string };
}

/** Events of one local day, classified. Returns null if the user hasn't connected a calendar. */
export async function getDayEvents(userId: string, date: string, timeZone: string): Promise<ClassifiedEvent[] | null> {
  const conn = await prisma.calendarConnection.findUnique({ where: { userId } });
  if (!conn) return null;

  const cacheKey = `gcal:events:${userId}:${date}`;
  const cached = await cache.get<ClassifiedEvent[]>(cacheKey);
  if (cached) return cached;

  const token = await accessToken(userId, conn.refreshTokenEnc);
  const { start, end } = dayRange(date, timeZone);
  const url = new URL(`${EVENTS_URL}/${encodeURIComponent(conn.calendarId)}/events`);
  url.search = new URLSearchParams({
    timeMin: start.toISOString(),
    timeMax: end.toISOString(),
    singleEvents: "true", // expand recurring events
    orderBy: "startTime",
    maxResults: "50",
    timeZone,
  }).toString();

  let res: Response;
  try {
    res = await fetch(url, { headers: { Authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(10_000) });
  } catch (err) {
    throw new HttpError(503, `Google Calendar unreachable: ${(err as Error).message}`);
  }
  if (res.status === 401) {
    await cache.del(`gcal:token:${userId}`);
    throw new HttpError(502, "Google Calendar rejected the access token; try again");
  }
  if (!res.ok) throw new HttpError(502, `Google Calendar error (${res.status})`);

  const items = ((await res.json()) as { items?: GoogleEvent[] }).items ?? [];
  const events = items
    .filter((e) => e.status !== "cancelled" && e.transparency !== "transparent") // skip "free" events (e.g. birthdays)
    .map((e): ClassifiedEvent => {
      const allDay = !e.start?.dateTime;
      const title = (e.summary ?? "(tanpa judul)").slice(0, 200);
      return {
        title,
        start: e.start?.dateTime ?? e.start?.date ?? start.toISOString(),
        end: e.end?.dateTime ?? e.end?.date ?? end.toISOString(),
        allDay,
        type: classifyEvent(title, (e.description ?? "").slice(0, 500)),
      };
    });
  await cache.set(cacheKey, events, EVENTS_CACHE_TTL_SEC);
  return events;
}

export async function disconnect(userId: string): Promise<boolean> {
  const conn = await prisma.calendarConnection.findUnique({ where: { userId } });
  if (!conn) return false;
  try {
    // Revoke at Google too, so the token can't be used even if it leaked.
    const token = decrypt(conn.refreshTokenEnc, config.tokenEncryptionKey);
    await fetch(REVOKE_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ token }),
      signal: AbortSignal.timeout(10_000),
    });
  } catch {
    /* best effort; the local copy is deleted regardless */
  }
  // Cached events need no cleanup: getDayEvents checks the connection before reading the cache.
  await prisma.calendarConnection.delete({ where: { userId } });
  await cache.del(`gcal:token:${userId}`);
  return true;
}

export async function connectionStatus(userId: string) {
  const conn = await prisma.calendarConnection.findUnique({
    where: { userId },
    select: { provider: true, calendarId: true, createdAt: true },
  });
  return { configured: isGoogleConfigured(), connected: !!conn, ...(conn ?? {}) };
}
