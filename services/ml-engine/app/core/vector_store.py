"""Qdrant storage for wardrobe item embeddings.

Point ID = WardrobeItem.id (UUID), so WardrobeItem.qdrantVectorId equals the item id.
Every point carries user_id in its payload; all searches are filtered by it.
"""

import logging
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app import config

log = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self.collection = config.QDRANT_COLLECTION
        self._client = AsyncQdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        self._ready = False

    async def ensure_collection(self) -> None:
        if self._ready:
            return
        if not await self._client.collection_exists(self.collection):
            await self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=config.EMBEDDING_DIM, distance=qm.Distance.COSINE),
            )
            log.info("Created Qdrant collection %s", self.collection)
        for field in ("user_id", "category"):
            await self._client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        self._ready = True

    async def is_reachable(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def upsert(self, item_id: UUID, user_id: UUID, vector: list[float], category: str | None) -> None:
        await self.ensure_collection()
        payload: dict[str, str] = {"user_id": str(user_id)}
        if category:
            payload["category"] = category
        await self._client.upsert(
            collection_name=self.collection,
            points=[qm.PointStruct(id=str(item_id), vector=vector, payload=payload)],
            wait=True,
        )

    async def exists_for_user(self, item_id: UUID, user_id: UUID) -> bool:
        await self.ensure_collection()
        points = await self._client.retrieve(self.collection, ids=[str(item_id)], with_payload=True)
        return bool(points) and (points[0].payload or {}).get("user_id") == str(user_id)

    async def similar(
        self, item_id: UUID, user_id: UUID, limit: int, category: str | None
    ) -> list[tuple[str, float, dict]]:
        """Nearest items of the same user to item_id (the item itself is excluded)."""
        await self.ensure_collection()
        must = [qm.FieldCondition(key="user_id", match=qm.MatchValue(value=str(user_id)))]
        if category:
            must.append(qm.FieldCondition(key="category", match=qm.MatchValue(value=category)))
        response = await self._client.query_points(
            collection_name=self.collection,
            query=str(item_id),
            query_filter=qm.Filter(
                must=must,
                must_not=[qm.HasIdCondition(has_id=[str(item_id)])],
            ),
            limit=limit,
            with_payload=True,
        )
        return [(str(p.id), p.score, p.payload or {}) for p in response.points]

    async def delete(self, item_id: UUID) -> None:
        await self.ensure_collection()
        await self._client.delete(
            collection_name=self.collection,
            points_selector=qm.PointIdsList(points=[str(item_id)]),
            wait=True,
        )

    async def close(self) -> None:
        await self._client.close()
