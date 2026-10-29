-- CreateEnum
CREATE TYPE "ItemCategory" AS ENUM ('TOP', 'BOTTOM', 'OUTERWEAR', 'DRESS', 'FOOTWEAR', 'ACCESSORY');

-- CreateTable
CREATE TABLE "users" (
    "id" UUID NOT NULL,
    "email" TEXT NOT NULL,
    "password_hash" TEXT,
    "display_name" TEXT,
    "timezone" TEXT NOT NULL DEFAULT 'Asia/Jakarta',
    "latitude" DOUBLE PRECISION,
    "longitude" DOUBLE PRECISION,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "wardrobe_items" (
    "id" UUID NOT NULL,
    "user_id" UUID NOT NULL,
    "category" "ItemCategory" NOT NULL,
    "item_type" TEXT NOT NULL,
    "colors" TEXT[],
    "pattern" TEXT,
    "styles" TEXT[],
    "warmth_level" SMALLINT NOT NULL DEFAULT 3,
    "water_resistant" BOOLEAN NOT NULL DEFAULT false,
    "image_url" TEXT NOT NULL,
    "masked_image_url" TEXT,
    "qdrant_vector_id" UUID,
    "ml_metadata" JSONB,
    "wear_count" INTEGER NOT NULL DEFAULT 0,
    "last_worn_at" TIMESTAMP(3),
    "archived" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "wardrobe_items_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "outfit_logs" (
    "id" UUID NOT NULL,
    "user_id" UUID NOT NULL,
    "worn_on" DATE NOT NULL,
    "weather" JSONB,
    "event_type" TEXT,
    "score" DOUBLE PRECISION,
    "score_detail" JSONB,
    "recommended" BOOLEAN NOT NULL DEFAULT true,
    "rating" SMALLINT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "outfit_logs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "outfit_log_items" (
    "outfit_log_id" UUID NOT NULL,
    "wardrobe_item_id" UUID NOT NULL,

    CONSTRAINT "outfit_log_items_pkey" PRIMARY KEY ("outfit_log_id","wardrobe_item_id")
);

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");

-- CreateIndex
CREATE UNIQUE INDEX "wardrobe_items_qdrant_vector_id_key" ON "wardrobe_items"("qdrant_vector_id");

-- CreateIndex
CREATE INDEX "wardrobe_items_user_id_category_idx" ON "wardrobe_items"("user_id", "category");

-- CreateIndex
CREATE INDEX "wardrobe_items_user_id_last_worn_at_idx" ON "wardrobe_items"("user_id", "last_worn_at");

-- CreateIndex
CREATE INDEX "outfit_logs_user_id_worn_on_idx" ON "outfit_logs"("user_id", "worn_on");

-- CreateIndex
CREATE INDEX "outfit_log_items_wardrobe_item_id_idx" ON "outfit_log_items"("wardrobe_item_id");

-- AddForeignKey
ALTER TABLE "wardrobe_items" ADD CONSTRAINT "wardrobe_items_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "outfit_logs" ADD CONSTRAINT "outfit_logs_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "outfit_log_items" ADD CONSTRAINT "outfit_log_items_outfit_log_id_fkey" FOREIGN KEY ("outfit_log_id") REFERENCES "outfit_logs"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "outfit_log_items" ADD CONSTRAINT "outfit_log_items_wardrobe_item_id_fkey" FOREIGN KEY ("wardrobe_item_id") REFERENCES "wardrobe_items"("id") ON DELETE CASCADE ON UPDATE CASCADE;
