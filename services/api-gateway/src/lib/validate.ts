import type { z } from "zod";
import { HttpError } from "./errors.js";

export function parse<T>(schema: z.ZodType<T>, value: unknown): T {
  const result = schema.safeParse(value);
  if (!result.success) {
    const issue = result.error.issues[0];
    throw new HttpError(400, issue ? `${issue.path.join(".") || "request"}: ${issue.message}` : "Invalid request");
  }
  return result.data;
}
