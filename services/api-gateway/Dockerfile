# Build context: repository root (needs packages/shared-types)
FROM node:22-alpine AS build
WORKDIR /repo
COPY package.json package-lock.json* ./
COPY packages/shared-types ./packages/shared-types
COPY services/api-gateway/package.json ./services/api-gateway/
RUN npm install --workspace @dressa/api-gateway --include-workspace-root=false
COPY services/api-gateway ./services/api-gateway
WORKDIR /repo/services/api-gateway
RUN npx prisma generate && npx tsc -p tsconfig.json

FROM node:22-alpine
WORKDIR /repo
ENV NODE_ENV=production
COPY --from=build /repo/node_modules ./node_modules
COPY --from=build /repo/packages ./packages
COPY --from=build /repo/services/api-gateway/dist ./services/api-gateway/dist
COPY --from=build /repo/services/api-gateway/prisma ./services/api-gateway/prisma
COPY --from=build /repo/services/api-gateway/package.json ./services/api-gateway/
WORKDIR /repo/services/api-gateway
RUN mkdir -p uploads && chown node:node uploads
USER node
EXPOSE 3000
CMD ["node", "dist/index.js"]
