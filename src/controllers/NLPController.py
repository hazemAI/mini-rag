from .BaseController import BaseController
from models.db_schemas.minirag.schemas import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnums
from typing import List
import json


class NLPController(BaseController):
    def __init__(self, vectordb_client, generation_client, embedding_client, template_parser):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser

    def create_collection_name(self, project_id: str):
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()

    async def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(
            project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name=collection_name)

    async def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(
            project_id=project.project_id)
        collection_info = await self.vectordb_client.get_collection_info(
            collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda o: o.__dict__)
        )

    async def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                             chunks_ids: List[int],
                             do_reset: bool = False):

        # step1: get collection name
        collection_name = self.create_collection_name(
            project_id=project.project_id)

        # step2: manage items
        texts = [c.chunk_text for c in chunks]
        metadata = [c.chunk_metadata for c in chunks]
        vectors = self.embedding_client.embed_text(text=texts, document_type=DocumentTypeEnums.DOCUMENT.value)
        
        # step3: create collection if not exists
        _ = await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset
        )

        # step4: index into vector db
        _ = await self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids
        )

        return True

    async def _search_vector_db_collection_internal(self, project: Project, query: str, limit: int = 5):
        """Internal method that returns original objects for internal processing"""
        
        # step 1: get collection name
        query_vector = None
        collection_name = self.create_collection_name(
            project_id=project.project_id)

        # step 2: get text embedding vector
        vectors = self.embedding_client.embed_text(
            text=query, document_type=DocumentTypeEnums.QUERY.value)

        if not vectors or len(vectors) == 0:
            return False

        if isinstance(vectors, list) and len(vectors) > 0:
            query_vector = vectors[0]

        if not query_vector:
            return False

        # step 3: do semantic search
        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vector,
            limit=limit
        )

        if not results or len(results) == 0:
            return False
        
        return results

    async def search_vector_db_collection(self, project: Project, query: str, limit: int = 5):
        """Public method that returns JSON serializable results for API responses"""
        
        results = await self._search_vector_db_collection_internal(project, query, limit)
        
        if not results:
            return False
        
        # Return JSON serializable data for API responses
        return json.loads(
            json.dumps(results, default=lambda o: o.__dict__)
        )

    async def answer_rag_question(self, project: Project, query: str, limit: int = 5):

        answer, full_prompt, chat_history = None, None, None
        
        # step 1: search vector db collection (using internal method for original objects)
        retrieved_docs = await self._search_vector_db_collection_internal(
            project=project,
            query=query,
            limit=limit
        )

        if not retrieved_docs or len(retrieved_docs) == 0:
            return answer, full_prompt, chat_history

        # step 2: construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "documents_prompt", {
                "doc_num": idx + 1,
                "chunk_text": self.generation_client.process_text(doc.text)
            })
            for idx, doc in enumerate(retrieved_docs)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })
        
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value
            ),
        ]
        
        full_prompt = "\n\n".join([
            documents_prompts,
            footer_prompt
        ])
        
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )
        
        return answer, full_prompt, chat_history
        