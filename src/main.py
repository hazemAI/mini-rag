from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from stores.llm.LLMFactory import LLMProviderFactory
from routes import base, data
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # MongoDB
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]

    # LLM factory
    llm_provider_factory = LLMProviderFactory(config=settings)

    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    if app.generation_client:
        app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    if app.embedding_client:
        app.embedding_client.set_embedding_model(
            model_id=settings.EMBEDDING_MODEL_ID,
            embedding_size=settings.EMBEDDING_MODEL_SIZE
        )

    # yield control to the app
    yield

    # cleanup
    app.mongo_conn.close()


app = FastAPI(lifespan=lifespan)

# Routers
app.include_router(base.base_router)
app.include_router(data.data_router)
