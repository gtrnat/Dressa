import type { FastifyReply, FastifyRequest } from "fastify";
import { z } from "zod";
import { config } from "../config.js";
import { HttpError } from "../lib/errors.js";
import { prisma } from "../lib/prisma.js";
import { createItemFromPhoto, deleteItem, getOwnedItem } from "../services/items.js";

const ALLOWED_MIME = new Set(["image/jpeg", "image/png", "image/webp"]);
const CATEGORIES = ["TOP", "BOTTOM", "OUTERWEAR", "DRESS", "FOOTWEAR", "ACCESSORY"] as const;

const idParams = z.object({ id: z.uuid() });

const listQuery = z.object({
  category: z.enum(CATEGORIES).optional(),
  archived: z.enum(["true", "false"]).optional().transform((v) => v === "true"),
  limit: z.coerce.number().int().min(1).max(100).default(50),
  offset: z.coerce.number().int().min(0).default(0),
});

const hexColor = z.string().regex(/^#[0-9a-fA-F]{6}$/, "Colors must be hex like #1a2b3c");
const tag = z.string().trim().min(1).max(32);

/** Fields the user may correct after auto-tagging. */
const updateBody = z
  .object({
    category: z.enum(CATEGORIES),
    itemType: tag,
    colors: z.array(hexColor).max(5),
    pattern: tag.nullable(),
    styles: z.array(tag).max(5),
    warmthLevel: z.number().int().min(1).max(5),
    waterResistant: z.boolean(),
    archived: z.boolean(),
  })
  .partial()
  .strict()
  .refine((b) => Object.keys(b).length > 0, "Body must contain at least one field");

function parse<T>(schema: z.ZodType<T>, value: unknown): T {
  const result = schema.safeParse(value);
  if (!result.success) {
    const issue = result.error.issues[0];
    throw new HttpError(400, issue ? `${issue.path.join(".") || "request"}: ${issue.message}` : "Invalid request");
  }
  return result.data;
}

export async function createItem(req: FastifyRequest, reply: FastifyReply) {
  const file = await req.file({ limits: { fileSize: config.maxUploadBytes, files: 1 } });
  if (!file) throw new HttpError(400, "Send the photo as multipart field 'file'");
  if (!ALLOWED_MIME.has(file.mimetype)) throw new HttpError(415, "Photo must be JPEG, PNG or WebP");

  const photo = await file.toBuffer(); // throws 413 if over the limit
  if (photo.length === 0) throw new HttpError(400, "Empty file");

  const { item, needsReview } = await createItemFromPhoto(req.userId, photo, file.mimetype, file.filename);
  return reply.code(201).send({ item, needsReview });
}

export async function listItems(req: FastifyRequest) {
  const q = parse(listQuery, req.query);
  const where = { userId: req.userId, archived: q.archived, ...(q.category && { category: q.category }) };
  const [items, total] = await Promise.all([
    prisma.wardrobeItem.findMany({ where, orderBy: { createdAt: "desc" }, take: q.limit, skip: q.offset }),
    prisma.wardrobeItem.count({ where }),
  ]);
  return { items, total, limit: q.limit, offset: q.offset };
}

export async function getItem(req: FastifyRequest) {
  const { id } = parse(idParams, req.params);
  return getOwnedItem(req.userId, id);
}

export async function updateItem(req: FastifyRequest) {
  const { id } = parse(idParams, req.params);
  const data = parse(updateBody, req.body);
  await getOwnedItem(req.userId, id);
  return prisma.wardrobeItem.update({ where: { id }, data });
}

export async function removeItem(req: FastifyRequest, reply: FastifyReply) {
  const { id } = parse(idParams, req.params);
  await deleteItem(req.userId, id);
  return reply.code(204).send();
}
