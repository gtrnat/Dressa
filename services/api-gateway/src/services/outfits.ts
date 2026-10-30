import type { Prisma } from "@prisma/client";
import { HttpError } from "../lib/errors.js";
import { prisma } from "../lib/prisma.js";
import { dayRange, todayIn } from "../lib/time.js";

export interface WearInput {
  itemIds: string[];
  wornOn?: string;
  recommended: boolean;
  eventType?: string;
  score?: number;
  scoreDetail?: Record<string, number | null>;
  weather?: Record<string, unknown>;
}

/** Midday of the local date, so "last worn" sorts sensibly for past dates. */
function wornAt(date: string, timeZone: string): Date {
  if (date === todayIn(timeZone)) return new Date();
  return new Date(dayRange(date, timeZone).start.getTime() + 12 * 3_600_000);
}

const logInclude = {
  items: {
    include: {
      wardrobeItem: { select: { id: true, category: true, itemType: true, maskedImageUrl: true, colors: true } },
    },
  },
} satisfies Prisma.OutfitLogInclude;

export async function logWear(userId: string, timeZone: string, input: WearInput) {
  const ids = [...new Set(input.itemIds)];
  const owned = await prisma.wardrobeItem.count({ where: { id: { in: ids }, userId } });
  if (owned !== ids.length) throw new HttpError(404, "One or more items not found");

  const date = input.wornOn ?? todayIn(timeZone);
  if (date > todayIn(timeZone)) throw new HttpError(400, "wornOn cannot be in the future");
  const at = wornAt(date, timeZone);

  return prisma.$transaction(async (tx) => {
    const log = await tx.outfitLog.create({
      data: {
        userId,
        wornOn: new Date(`${date}T00:00:00Z`),
        recommended: input.recommended,
        eventType: input.eventType,
        score: input.score,
        scoreDetail: input.scoreDetail as Prisma.InputJsonValue | undefined,
        weather: input.weather as Prisma.InputJsonValue | undefined,
        items: { create: ids.map((wardrobeItemId) => ({ wardrobeItemId })) },
      },
      include: logInclude,
    });
    await tx.wardrobeItem.updateMany({ where: { id: { in: ids } }, data: { wearCount: { increment: 1 } } });
    // Never move lastWornAt backwards when logging an older date.
    await tx.wardrobeItem.updateMany({
      where: { id: { in: ids }, OR: [{ lastWornAt: null }, { lastWornAt: { lt: at } }] },
      data: { lastWornAt: at },
    });
    return log;
  });
}

export async function listLogs(userId: string, limit: number, offset: number) {
  const [logs, total] = await Promise.all([
    prisma.outfitLog.findMany({
      where: { userId },
      orderBy: [{ wornOn: "desc" }, { createdAt: "desc" }],
      take: limit,
      skip: offset,
      include: logInclude,
    }),
    prisma.outfitLog.count({ where: { userId } }),
  ]);
  return { logs, total, limit, offset };
}

/** Undo a wear log: decrement counts and recompute lastWornAt from the remaining logs. */
export async function deleteLog(userId: string, logId: string, timeZone: string) {
  const log = await prisma.outfitLog.findFirst({ where: { id: logId, userId }, include: { items: true } });
  if (!log) throw new HttpError(404, "Outfit log not found");
  const ids = log.items.map((i) => i.wardrobeItemId);

  await prisma.$transaction(async (tx) => {
    await tx.outfitLog.delete({ where: { id: log.id } });
    await tx.wardrobeItem.updateMany({ where: { id: { in: ids }, wearCount: { gt: 0 } }, data: { wearCount: { decrement: 1 } } });
    for (const id of ids) {
      const latest = await tx.outfitLog.findFirst({
        where: { userId, items: { some: { wardrobeItemId: id } } },
        orderBy: [{ wornOn: "desc" }, { createdAt: "desc" }],
        select: { wornOn: true },
      });
      await tx.wardrobeItem.update({
        where: { id },
        data: { lastWornAt: latest ? wornAt(latest.wornOn.toISOString().slice(0, 10), timeZone) : null },
      });
    }
  });
}
