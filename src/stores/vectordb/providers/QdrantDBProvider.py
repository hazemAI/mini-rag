from qdrant_client import QdrantClient, models
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
from typing import List
import logging
from models.db_schemas.minirag.schemas import RetrievedDocument


class QdrantDBProvider(VectorDBInterface):
    def __init__(self, db_client: str, default_vector_size: int = 786, distance_method: str = None, index_threshold: int = 100):

        self.client = None
        self.db_client = db_client
        self.default_vector_size = default_vector_size
        self.distance_method = models.Distance.COSINE
        self.logger = logging.getLogger("uvicorn")

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT
        elif distance_method is not None:
            self.logger.warning(
                "Unsupported distance method '%s'. Falling back to COSINE.",
                distance_method,
            )

    async def connect(self):
        self.client = QdrantClient(path=self.db_client)

    async def disconnect(self):
        self.client = None

    async def is_collection_existed(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)

    async def list_all_collections(self) -> List:
        return self.client.get_collections()

    async def get_collection_info(self, collection_name: str) -> dict:
        return self.client.get_collection(collection_name=collection_name)

    async def delete_collection(self, collection_name: str):
        if await self.is_collection_existed(collection_name=collection_name):
            self.logger.info(f"Deleting collection: {collection_name}")
            return self.client.delete_collection(collection_name=collection_name)

    async def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False):
        if do_reset:
            _ = await self.delete_collection(collection_name=collection_name)

        if not await self.is_collection_existed(collection_name=collection_name):
            self.logger.info(
                f"Creating new Qdrant collection: {collection_name}")

            _ = self.client.create_collection(collection_name=collection_name,
                                              vectors_config=models.VectorParams(
                                                  size=embedding_size,
                                                  distance=self.distance_method
                                              ))
            return True

        return False

    async def insert_one(self, collection_name: str, text: str,
                         vector: list, metadata: dict = None,
                         record_id: str = None):
        if not await self.is_collection_existed(collection_name=collection_name):
            self.logger.error(
                f"Can't insert new record to non existed collection: {collection_name}")
            return False

        _ = self.client.upload_records(
            collection_name=collection_name,
            records=[models.Record(
                id=[record_id],
                vector=vector,
                payload={
                    "text": text,
                    "metadata": metadata
                }
            )]
        )

        return True

    async def insert_many(self, collection_name: str, texts: list,
                          vectors: list, metadatas: list = None,
                          record_ids: list = None, batch_size: int = 50):
        if metadatas is None:
            metadatas = [None] * len(texts)

        if record_ids is None:
            record_ids = list(range(0, len(texts)))

        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size

            batch_texts = texts[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_metadata = metadatas[i:batch_end]
            batch_record_ids = record_ids[i:batch_end]

            batch_records = [
                models.Record(
                    id=batch_record_ids[j],
                    vector=batch_vectors[j],
                    payload={
                        "text": batch_texts[j],
                        "metadata": batch_metadata[j]
                    }
                )
                for j in range(len(batch_texts))
            ]

            try:
                _ = self.client.upload_records(
                    collection_name=collection_name,
                    records=batch_records
                )
            except Exception as e:
                self.logger.error(
                    f"Can't insert new records to collection: {collection_name}, error: {e}")
                return False

            return True

    async def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):
        if not await self.is_collection_existed(collection_name=collection_name):
            self.logger.error(
                f"Can't search in non existed collection: {collection_name}")
            return False

        results = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit
        )

        if not results or len(results) == 0:
            return None

        return [
            RetrievedDocument(**{
                "text": result.payload["text"],
                "score": result.score
            })
            for result in results
        ]
