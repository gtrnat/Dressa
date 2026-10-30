import type { FastifyReply, FastifyRequest } from "fastify";
import { z } from "zod";
import { config } from "../config.js";
import { HttpError } from "../lib/errors.js";
import { parse } from "../lib/validate.js";
import { buildAuthUrl, connectionStatus, disconnect, handleCallback } from "../services/google-calendar.js";

export async function googleStatus(req: FastifyRequest) {
  return connectionStatus(req.userId);
}

/** Returns the Google consent URL; the client opens it in a browser. */
export async function googleAuthUrl(req: FastifyRequest) {
  return { url: buildAuthUrl(req.userId) };
}

export async function googleDisconnect(req: FastifyRequest, reply: FastifyReply) {
  const removed = await disconnect(req.userId);
  if (!removed) throw new HttpError(404, "Google Calendar is not connected");
  return reply.code(204).send();
}

const callbackQuery = z.object({
  code: z.string().min(1).max(2048).optional(),
  state: z.string().min(1).max(2048).optional(),
  error: z.string().max(200).optional(),
});

function page(title: string, message: string): string {
  const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
  return `<!doctype html><meta charset="utf-8"><title>${esc(title)}</title>
<body style="font-family:system-ui;max-width:32rem;margin:4rem auto;text-align:center">
<h2>${esc(title)}</h2><p>${esc(message)}</p></body>`;
}

/** Public: Google redirects the browser here. The signed state identifies the user. */
export async function googleCallback(req: FastifyRequest, reply: FastifyReply) {
  const q = parse(callbackQuery, req.query);
  reply.type("text/html; charset=utf-8");
  if (q.error || !q.code || !q.state) {
    return reply.code(400).send(page("Koneksi dibatalkan", "Google Calendar tidak terhubung. Silakan coba lagi dari aplikasi."));
  }
  try {
    await handleCallback(q.code, q.state);
  } catch (err) {
    const status = err instanceof HttpError ? err.statusCode : 500;
    req.log.warn({ err }, "Google OAuth callback failed");
    return reply.code(status).send(page("Gagal menghubungkan", err instanceof HttpError ? err.message : "Terjadi kesalahan."));
  }
  if (config.google.successRedirect) return reply.redirect(config.google.successRedirect);
  return reply.send(page("Google Calendar terhubung", "Kamu bisa menutup halaman ini dan kembali ke Dressa."));
}
