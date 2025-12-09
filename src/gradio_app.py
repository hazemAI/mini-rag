import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import asyncio
import aiofiles

settings = None
generation_client = None
embedding_client = None
vectordb_client = None
template_parser = None
_initialized = False
_init_error = None
_postgres_conn = None


def lazy_init():
    global settings
    if settings is None:
        from helpers.config import get_settings

        settings = get_settings()


def get_fresh_db_session():
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    engine = create_async_engine(_postgres_conn, pool_pre_ping=True)
    return sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def initialize_services():
    global generation_client, embedding_client
    global vectordb_client, template_parser, _initialized, _init_error, _postgres_conn

    if _initialized:
        return True

    try:
        lazy_init()

        from stores.llm.LLMFactory import LLMProviderFactory
        from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
        from stores.llm.templates.template_parser import TemplateParser
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker

        _postgres_conn = "postgresql+asyncpg://{}:{}@{}:{}/{}".format(
            settings.POSTGRES_USERNAME,
            settings.POSTGRES_PASSWORD,
            settings.POSTGRES_HOST,
            settings.POSTGRES_PORT,
            settings.POSTGRES_MAIN_DATABASE,
        )

        db_session_factory = get_fresh_db_session()

        llm_provider_factory = LLMProviderFactory(config=settings)
        vector_db_provider_factory = VectorDBProviderFactory(
            config=settings, db_client=db_session_factory
        )

        generation_client = llm_provider_factory.create(
            provider=settings.GENERATION_BACKEND
        )
        if generation_client:
            generation_client.set_generation_model(
                model_id=settings.GENERATION_MODEL_ID
            )

        embedding_client = llm_provider_factory.create(
            provider=settings.EMBEDDING_BACKEND
        )
        if embedding_client:
            embedding_client.set_embedding_model(
                model_id=settings.EMBEDDING_MODEL_ID,
                embedding_size=settings.EMBEDDING_MODEL_SIZE,
            )

        vectordb_client = vector_db_provider_factory.create(
            provider=settings.VECTOR_DB_BACKEND
        )
        await vectordb_client.connect()

        template_parser = TemplateParser(
            language=settings.PRIMARY_LANG,
            default_language=settings.DEFAULT_LANG,
        )

        _initialized = True
        _init_error = None
        print("✅ Services initialized successfully!")
        return True

    except Exception as e:
        _init_error = str(e)
        print(f"❌ Failed to initialize services: {e}")
        return False


async def get_projects():
    if not _initialized:
        await initialize_services()

    if not _initialized:
        return []

    from models.ProjectModel import ProjectModel

    try:
        db_session = get_fresh_db_session()
        project_model = await ProjectModel.create_instance(db_client=db_session)
        result = await project_model.get_all_projects()
        if result:
            projects, _ = result
            return (
                [(f"Project {p.project_id}", p.project_id) for p in projects]
                if projects
                else []
            )
    except Exception as e:
        print(f"Error getting projects: {e}")
    return []


async def create_project(project_id: int):
    from models.ProjectModel import ProjectModel

    db_session = get_fresh_db_session()
    project_model = await ProjectModel.create_instance(db_client=db_session)
    project = await project_model.get_project_or_create_one(project_id=project_id)
    return project


async def upload_file_async(file, project_id: int):
    if not _initialized:
        await initialize_services()
    if not _initialized:
        return f"❌ Services not initialized: {_init_error}"

    if file is None:
        return "⚠️ Please select a file."
    if not project_id:
        return "⚠️ Please enter a project ID."

    file_path = file.name

    from controllers.DataController import DataController
    from controllers.ProjectController import ProjectController
    from models.AssetModel import AssetModel
    from models.db_schemas import Asset
    from models.enums.AssetTypeEnum import AssetTypeEnum

    try:
        await create_project(project_id)

        data_controller = DataController()
        ProjectController().get_project_path(project_id=project_id)

        orig_filename = os.path.basename(file_path)
        dest_path, file_id = data_controller.generate_unique_filepath(
            orig_filename=orig_filename, project_id=project_id
        )

        async with aiofiles.open(file_path, "rb") as src:
            content = await src.read()
        async with aiofiles.open(dest_path, "wb") as dst:
            await dst.write(content)

        db_session = get_fresh_db_session()
        asset_model = await AssetModel.create_instance(db_client=db_session)
        asset_resource = Asset(
            asset_project_id=project_id,
            asset_type=AssetTypeEnum.FILE.value,
            asset_name=file_id,
            asset_size=os.path.getsize(dest_path),
        )
        asset_record = await asset_model.create_asset(asset=asset_resource)

        return f"✅ File uploaded successfully! Asset ID: {asset_record.asset_id}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


async def process_and_index_async(project_id: int, chunk_size: int, overlap_size: int):
    if not _initialized:
        await initialize_services()
    if not _initialized:
        return f"❌ Services not initialized: {_init_error}"

    from controllers.ProcessController import ProcessController
    from controllers.NLPController import NLPController
    from models.ProjectModel import ProjectModel
    from models.ChunkModel import ChunkModel
    from models.AssetModel import AssetModel

    if not project_id:
        return "⚠️ Please select a project."

    try:
        db_session = get_fresh_db_session()
        project_model = await ProjectModel.create_instance(db_client=db_session)
        project = await project_model.get_project_or_create_one(project_id=project_id)

        db_session2 = get_fresh_db_session()
        asset_model = await AssetModel.create_instance(db_client=db_session2)
        from models.enums.AssetTypeEnum import AssetTypeEnum

        assets = await asset_model.get_all_project_assets(
            asset_project_id=project_id, asset_type=AssetTypeEnum.FILE.value
        )

        if not assets:
            return "⚠️ No files found in this project. Please upload files first."

        process_controller = ProcessController(project_id=project_id)
        db_session3 = get_fresh_db_session()
        chunk_model = await ChunkModel.create_instance(db_client=db_session3)

        from models.db_schemas import DataChunk

        all_datachunks = []
        for asset in assets:
            file_content = process_controller.get_file_content(file_id=asset.asset_name)
            if file_content:
                doc_chunks = process_controller.process_file_content(
                    file_content=file_content,
                    file_id=asset.asset_name,
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                )
                if doc_chunks:
                    for i, doc in enumerate(doc_chunks):
                        datachunk = DataChunk(
                            chunk_text=doc.page_content,
                            chunk_metadata=doc.metadata,
                            chunk_order=i,
                            chunk_project_id=project_id,
                            chunk_asset_id=asset.asset_id,
                        )
                        all_datachunks.append(datachunk)

        if not all_datachunks:
            return "⚠️ No content could be extracted from files."

        saved_chunks = []
        for chunk in all_datachunks:
            saved_chunk = await chunk_model.create_chunk(chunk=chunk)
            saved_chunks.append(saved_chunk)

        nlp_controller = NLPController(
            generation_client=generation_client,
            embedding_client=embedding_client,
            vectordb_client=vectordb_client,
            template_parser=template_parser,
        )

        chunk_ids = [c.chunk_id for c in saved_chunks]
        await nlp_controller.index_into_vector_db(
            project=project, chunks=saved_chunks, chunks_ids=chunk_ids, do_reset=True
        )

        return f"✅ Processed and indexed {len(saved_chunks)} chunks successfully!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


async def answer_question(question: str, project_id: int, chat_history: list):
    chat_history = chat_history or []

    if not _initialized:
        await initialize_services()

    if not _initialized:
        chat_history.append({"role": "user", "content": question})
        chat_history.append(
            {
                "role": "assistant",
                "content": f"❌ Services not initialized: {_init_error}",
            }
        )
        return chat_history, "", ""

    from models.ProjectModel import ProjectModel
    from controllers.NLPController import NLPController

    if not question.strip():
        return chat_history, "", ""

    if not project_id:
        chat_history.append({"role": "user", "content": question})
        chat_history.append(
            {"role": "assistant", "content": "⚠️ Please select a project first."}
        )
        return chat_history, "", ""

    try:
        db_session = get_fresh_db_session()
        project_model = await ProjectModel.create_instance(db_client=db_session)
        project = await project_model.get_project_or_create_one(project_id=project_id)

        if not project:
            chat_history.append({"role": "user", "content": question})
            chat_history.append(
                {"role": "assistant", "content": "⚠️ Project not found."}
            )
            return chat_history, "", ""

        nlp_controller = NLPController(
            generation_client=generation_client,
            embedding_client=embedding_client,
            vectordb_client=vectordb_client,
            template_parser=template_parser,
        )

        answer, full_prompt, _ = await nlp_controller.answer_rag_question(
            project=project, query=question, limit=5
        )

        if not answer:
            chat_history.append({"role": "user", "content": question})
            chat_history.append(
                {
                    "role": "assistant",
                    "content": "⚠️ Could not generate an answer. Please ensure documents are indexed.",
                }
            )
            return chat_history, "", ""

        retrieved_docs = await nlp_controller._search_vector_db_collection_internal(
            project=project, query=question, limit=5
        )

        docs_display = ""
        if retrieved_docs:
            docs_display = "\n\n---\n\n".join(
                [
                    f"**Document {i + 1}** (Score: {doc.score:.3f})\n\n{doc.text[:500]}..."
                    if len(doc.text) > 500
                    else f"**Document {i + 1}** (Score: {doc.score:.3f})\n\n{doc.text}"
                    for i, doc in enumerate(retrieved_docs)
                ]
            )

        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        return chat_history, "", docs_display
    except Exception as e:
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return chat_history, "", ""


async def refresh_projects():
    projects = await get_projects()
    return gr.Dropdown(choices=projects)


def get_status_message():
    if _initialized:
        return "✅ Connected to database and ready!"
    elif _init_error:
        return f"❌ Not connected: {_init_error}"
    return "⏳ Initializing (will happen on first action)..."


def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            "# 🔍 Mini-RAG Question Answering\n"
            "Upload documents, index them, and ask questions!"
        )

        gr.Textbox(
            label="🔌 Connection Status",
            value=get_status_message(),
            interactive=False,
        )

        with gr.Tabs():
            with gr.TabItem("📤 Upload & Index"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 1️⃣ Upload File")
                        project_id_input = gr.Number(
                            label="Project ID", value=1, precision=0
                        )
                        file_input = gr.File(label="Select File (PDF or TXT)")
                        upload_btn = gr.Button("📤 Upload File", variant="primary")
                        upload_status = gr.Textbox(
                            label="Upload Status", interactive=False
                        )

                    with gr.Column():
                        gr.Markdown("### 2️⃣ Process & Index")
                        chunk_size_input = gr.Number(
                            label="Chunk Size", value=64, precision=0
                        )
                        overlap_input = gr.Number(
                            label="Overlap Size", value=10, precision=0
                        )
                        process_btn = gr.Button("⚙️ Process & Index", variant="primary")
                        process_status = gr.Textbox(
                            label="Processing Status", interactive=False
                        )

                upload_btn.click(
                    fn=upload_file_async,
                    inputs=[file_input, project_id_input],
                    outputs=[upload_status],
                )

                process_btn.click(
                    fn=process_and_index_async,
                    inputs=[project_id_input, chunk_size_input, overlap_input],
                    outputs=[process_status],
                )

            with gr.TabItem("💬 Ask Questions"):
                with gr.Row():
                    with gr.Column(scale=2):
                        project_dropdown = gr.Dropdown(
                            label="📁 Select Project",
                            choices=[],
                            value=None,
                            interactive=True,
                        )
                        refresh_btn = gr.Button("🔄 Refresh Projects", size="sm")

                        chatbot = gr.Chatbot(
                            label="💬 Conversation",
                            height=450,
                        )

                        with gr.Row():
                            question_input = gr.Textbox(
                                label="Your Question",
                                placeholder="Type your question here...",
                                scale=4,
                                lines=2,
                            )
                            submit_btn = gr.Button("🚀 Ask", variant="primary", scale=1)

                        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📚 Retrieved Documents")
                        docs_display = gr.Markdown(
                            value="*Documents will appear here after asking a question.*",
                        )

                demo.load(fn=refresh_projects, outputs=[project_dropdown])
                refresh_btn.click(fn=refresh_projects, outputs=[project_dropdown])

                submit_btn.click(
                    fn=answer_question,
                    inputs=[question_input, project_dropdown, chatbot],
                    outputs=[chatbot, question_input, docs_display],
                )

                question_input.submit(
                    fn=answer_question,
                    inputs=[question_input, project_dropdown, chatbot],
                    outputs=[chatbot, question_input, docs_display],
                )

                clear_btn.click(
                    fn=lambda: (
                        [],
                        "",
                        "*Documents will appear here after asking a question.*",
                    ),
                    outputs=[chatbot, question_input, docs_display],
                )

    return demo


if __name__ == "__main__":
    print("🚀 Starting Mini-RAG...")

    print("🌐 Launching Gradio interface...")
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
