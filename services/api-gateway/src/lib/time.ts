/** Timezone helpers built on Intl (no extra dependency). Dates are "YYYY-MM-DD" strings in a given IANA zone. */

export function isValidTimeZone(tz: string): boolean {
  try {
    new Intl.DateTimeFormat("en-US", { timeZone: tz });
    return true;
  } catch {
    return false;
  }
}

export function localParts(date: Date, timeZone: string): { date: string; hour: number } {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-CA", {
      timeZone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      hourCycle: "h23",
    })
      .formatToParts(date)
      .map((p) => [p.type, p.value]),
  );
  return { date: `${parts.year}-${parts.month}-${parts.day}`, hour: Number(parts.hour) };
}

export function todayIn(timeZone: string): string {
  return localParts(new Date(), timeZone).date;
}

/** Offset of `timeZone` from UTC at `date`, in minutes. */
function offsetMinutes(date: Date, timeZone: string): number {
  const p = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", {
      timeZone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hourCycle: "h23",
    })
      .formatToParts(date)
      .map((x) => [x.type, x.value]),
  );
  const asUtc = Date.UTC(+p.year!, +p.month! - 1, +p.day!, +p.hour!, +p.minute!, +p.second!);
  return Math.round((asUtc - date.getTime()) / 60000);
}

/** UTC instants for [start of day, start of next day) of a local date. */
export function dayRange(localDate: string, timeZone: string): { start: Date; end: Date } {
  const [y, m, d] = localDate.split("-").map(Number) as [number, number, number];
  const startGuess = new Date(Date.UTC(y, m - 1, d));
  const start = new Date(startGuess.getTime() - offsetMinutes(startGuess, timeZone) * 60000);
  const endGuess = new Date(Date.UTC(y, m - 1, d + 1));
  const end = new Date(endGuess.getTime() - offsetMinutes(endGuess, timeZone) * 60000);
  return { start, end };
}

export function addDays(localDate: string, days: number): string {
  const [y, m, d] = localDate.split("-").map(Number) as [number, number, number];
  return new Date(Date.UTC(y, m - 1, d + days)).toISOString().slice(0, 10);
}

/** Whole days between two instants (b - a), floored, never negative. */
export function daysBetween(a: Date, b: Date): number {
  return Math.max(0, Math.floor((b.getTime() - a.getTime()) / 86_400_000));
}
