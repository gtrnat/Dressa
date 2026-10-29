import { randomUUID } from "node:crypto";
import type { Prisma, WardrobeItem } from "@prisma/client";
import { HttpError } from "../lib/errors.js";
import { prisma } from "../lib/prisma.js";
import { mlEngine } from "./ml-engine.js";
import { itemPrefix, storage } from "./storage.js";

const EXT: Record<string, string> = { "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp" };

export interface NewItemResult {
  item: WardrobeItem;
  needsReview: boolean;
}

/**
 * Add-item pipeline:
 *  1. ml-engine /process-image  -> cutout PNG + tags
 *  2. store original + cutout
 *  3. ml-engine /embed           -> vector upserted to Qdrant (point id = item id)
 *  4. insert WardrobeItem in Postgres
 * If a later step fails, earlier side effects are rolled back (files, Qdrant point).
 */
export async function createItemFromPhoto(
  userId: string,
  photo: Buffer,
  mimeType: string,
  filename: string,
): Promise<NewItemResult> {
  const processed = await mlEngine.processImage(photo, mimeType, filename);
  const cutout = Buffer.from(processed.cutout_png_base64, "base64");
  const { tags } = processed;

  const itemId = randomUUID();
  const prefix = itemPrefix(userId, itemId);
  let embeddingStored = false;

  try {
    const imageUrl = await storage.put(`${prefix}/original.${EXT[mimeType] ?? "bin"}`, photo);
    const maskedImageUrl = await storage.put(`${prefix}/cutout.png`, cutout);

    await mlEngine.storeEmbedding(cutout, itemId, userId, tags.category);
    embeddingStored = true;

    // Keep the raw ML output (minus the image) so items can be re-tagged or audited later.
    const { cutout_png_base64: _omit, ...mlMetadata } = processed;

    const item = await prisma.wardrobeItem.create({
      data: {
        id: itemId,
        userId,
        category: tags.category,
        itemType: tags.item_type.label,
        colors: tags.dominant_colors.map((c) => c.hex),
        pattern: tags.pattern.label,
        styles: tags.styles.map((s) => s.label),
        warmthLevel: tags.warmth_level,
        waterResistant: tags.water_resistant,
        imageUrl,
        maskedImageUrl,
        qdrantVectorId: itemId,
        mlMetadata: mlMetadata as unknown as Prisma.InputJsonValue,
      },
    });
    return { item, needsReview: processed.needs_review };
  } catch (err) {
    await Promise.allSettled([
      storage.removePrefix(prefix),
      embeddingStored ? mlEngine.deleteEmbedding(itemId, userId) : Promise.resolve(),
    ]);
    throw err;
  }
}

export async function getOwnedItem(userId: string, itemId: string): Promise<WardrobeItem> {
  const item = await prisma.wardrobeItem.findFirst({ where: { id: itemId, userId } });
  if (!item) throw new HttpError(404, "Item not found");
  return item;
}

export async function deleteItem(userId: string, itemId: string): Promise<void> {
  const item = await getOwnedItem(userId, itemId);
  // Qdrant first: if it fails, the DB row still exists and the delete can be retried.
  await mlEngine.deleteEmbedding(item.id, userId);
  await prisma.wardrobeItem.delete({ where: { id: item.id } });
  await storage.removePrefix(itemPrefix(userId, item.id));
}
