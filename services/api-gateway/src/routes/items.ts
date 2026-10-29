import type { FastifyInstance } from "fastify";
import { createItem, getItem, listItems, removeItem, updateItem } from "../controllers/items.js";
import { requireUser } from "../plugins/auth.js";

export async function itemRoutes(app: FastifyInstance): Promise<void> {
  app.addHook("preHandler", requireUser);

  app.post("/items", createItem);
  app.get("/items", listItems);
  app.get("/items/:id", getItem);
  app.patch("/items/:id", updateItem);
  app.delete("/items/:id", removeItem);
}
