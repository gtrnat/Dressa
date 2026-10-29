// Shared DTOs between api-gateway and clients.

export type ItemCategory =
  | "TOP"
  | "BOTTOM"
  | "OUTERWEAR"
  | "DRESS"
  | "FOOTWEAR"
  | "ACCESSORY";

export interface ItemTags {
  category: string;
  dominant_colors: string[];
  pattern: string;
  style: string[];
}

export interface ScoreBreakdown {
  styleMatch: number;
  weatherFit: number;
  calendarFit: number;
  rotationWeight: number;
  total: number;
}
