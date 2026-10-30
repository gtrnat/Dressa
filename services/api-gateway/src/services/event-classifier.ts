/**
 * Rule-based event type from a calendar event title/description (Indonesian + English keywords).
 * Kept as plain data so the vocabulary is easy to extend.
 */
export type EventType = "formal" | "casual" | "sport";
export const EVENT_TYPES: readonly EventType[] = ["formal", "casual", "sport"];

const KEYWORDS: Record<Exclude<EventType, "casual">, string[]> = {
  formal: [
    "meeting", "rapat", "interview", "wawancara", "presentasi", "presentation", "pitch", "sidang", "seminar",
    "konferensi", "conference", "client", "klien", "kantor", "office", "board", "kondangan", "nikahan",
    "pernikahan", "wedding", "resepsi", "wisuda", "graduation", "gala", "formal", "ujian", "exam", "kerja",
    "workshop", "townhall", "town hall", "investor", "dinner party", "akad",
  ],
  sport: [
    "gym", "fitness", "workout", "olahraga", "lari", "jogging", "running", " run ", "futsal", "sepak bola",
    "football", "soccer", "badminton", "bulu tangkis", "basket", "basketball", "renang", "swim", "yoga",
    "pilates", "tenis", "tennis", "padel", "hiking", "naik gunung", "sepeda", "cycling", "bike", "voli",
    "volleyball", "crossfit", "muay thai", "boxing", "tinju", "senam", "zumba",
  ],
};

export function classifyEvent(title: string, description = ""): EventType {
  const text = ` ${title} ${description} `.toLowerCase();
  // Title matters more than description: check title first.
  for (const source of [` ${title.toLowerCase()} `, text]) {
    if (KEYWORDS.formal.some((k) => source.includes(k))) return "formal";
    if (KEYWORDS.sport.some((k) => source.includes(k))) return "sport";
  }
  return "casual";
}

export interface ClassifiedEvent {
  title: string;
  start: string;
  end: string;
  allDay: boolean;
  type: EventType;
}

/**
 * Which outfit the day calls for. Formal wins if anything is formal (you can't wear gym clothes to
 * a meeting, but you can change for the gym). Sport only if sport takes most of the timed hours.
 */
export function dominantEventType(events: ClassifiedEvent[]): EventType | null {
  const timed = events.filter((e) => !e.allDay);
  const relevant = timed.length ? timed : events;
  if (relevant.length === 0) return null;
  if (relevant.some((e) => e.type === "formal")) return "formal";
  const hours = (t: EventType) =>
    relevant
      .filter((e) => e.type === t)
      .reduce((s, e) => s + Math.max(0.5, (Date.parse(e.end) - Date.parse(e.start)) / 3_600_000), 0);
  return hours("sport") > hours("casual") ? "sport" : "casual";
}
