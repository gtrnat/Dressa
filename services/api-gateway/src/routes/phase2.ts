import type { FastifyInstance } from "fastify";
import { googleAuthUrl, googleCallback, googleDisconnect, googleStatus } from "../controllers/integrations.js";
import { getMe, updateMe } from "../controllers/me.js";
import { listOutfits, removeOutfit, wearOutfit } from "../controllers/outfits.js";
import { getRecommendations } from "../controllers/recommendations.js";
import { requireUser } from "../plugins/auth.js";

export async function meRoutes(app: FastifyInstance): Promise<void> {
  app.addHook("preHandler", requireUser);
  app.get("/me", getMe);
  app.patch("/me", updateMe);
}

export async function recommendationRoutes(app: FastifyInstance): Promise<void> {
  app.addHook("preHandler", requireUser);
  app.get("/recommendations", getRecommendations);
}

export async function outfitRoutes(app: FastifyInstance): Promise<void> {
  app.addHook("preHandler", requireUser);
  app.post("/outfits", wearOutfit);
  app.get("/outfits", listOutfits);
  app.delete("/outfits/:id", removeOutfit);
}

export async function integrationRoutes(app: FastifyInstance): Promise<void> {
  app.get("/integrations/google", { preHandler: requireUser }, googleStatus);
  app.get("/integrations/google/auth-url", { preHandler: requireUser }, googleAuthUrl);
  app.delete("/integrations/google", { preHandler: requireUser }, googleDisconnect);
  // Public: reached by browser redirect from Google; authenticated via the signed state parameter.
  app.get("/integrations/google/callback", googleCallback);
}
