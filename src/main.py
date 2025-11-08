from fastapi import FastAPI
from helpers.config import get_settings
from stores.llm.LLMFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from routes import base, data, nlp
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    postgres_conn = "postgresql+asyncpg://{}:{}@{}:{}/{}".format(
        settings.POSTGRES_USERNAME,
        settings.POSTGRES_PASSWORD,
        settings.POSTGRES_HOST,
        settings.POSTGRES_PORT,
        settings.POSTGRES_MAIN_DATABASE,
    )
    
    app.db_engine = create_async_engine(postgres_conn)
    
    app.db_client = sessionmaker(
        bind=app.db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # LLM and VectorDB factories
    llm_provider_factory = LLMProviderFactory(config=settings)
    vector_db_provider_factory = VectorDBProviderFactory(config=settings, db_client=app.db_client)

    app.generation_client = llm_provider_factory.create(
        provider=settings.GENERATION_BACKEND)
    if app.generation_client:
        app.generation_client.set_generation_model(
            model_id=settings.GENERATION_MODEL_ID)

    app.embedding_client = llm_provider_factory.create(
        provider=settings.EMBEDDING_BACKEND)
    if app.embedding_client:
        app.embedding_client.set_embedding_model(
            model_id=settings.EMBEDDING_MODEL_ID,
            embedding_size=settings.EMBEDDING_MODEL_SIZE
        )

    app.vectordb_client = vector_db_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND,
    )
    await app.vectordb_client.connect()

    app.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )

    # yield control to the app
    yield

    # cleanup
    app.db_engine.dispose()
    await app.vectordb_client.disconnect()


app = FastAPI(lifespan=lifespan)

# Routers
app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
