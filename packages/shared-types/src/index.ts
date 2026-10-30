// Shared DTOs between api-gateway and clients.

export type ItemCategory =
  | "TOP"
  | "BOTTOM"
  | "OUTERWEAR"
  | "DRESS"
  | "FOOTWEAR"
  | "ACCESSORY";

export interface ColorTag {
  hex: string;
  name: string;
  /** Share of garment pixels, 0..1 */
  ratio: number;
}

export interface LabelTag {
  label: string;
  /** Zero-shot probability, 0..1 */
  confidence: number;
}

/** ml-engine POST /process-image -> tags */
export interface ItemTags {
  category: ItemCategory;
  item_type: LabelTag;
  dominant_colors: ColorTag[];
  pattern: LabelTag;
  styles: LabelTag[];
  /** summer/fall/winter/spring; only set when the ML classifier is loaded */
  season?: LabelTag | null;
  warmth_level: number;
  water_resistant: boolean;
}

/** ml-engine POST /process-image response */
export interface ProcessImageResponse {
  filename: string;
  background_removed: boolean;
  foreground_ratio: number | null;
  cutout_png_base64: string;
  cutout_width: number;
  cutout_height: number;
  tags: ItemTags;
  /** Model that produced item_type/style/warmth */
  attribute_model: "ml-classifier" | "clip-zero-shot";
  /** How likely the cutout is a single garment (vs. people, scenes, objects), 0..1 */
  garment_probability: number;
  needs_review: boolean;
}

export interface ScoreBreakdown {
  styleMatch: number;
  weatherFit: number;
  calendarFit: number;
  rotationWeight: number;
  total: number;
}
