from enum import Enum


class ResponseSignal(Enum):
    FILE_VALIDATED_SUCCESS = "file_validated_successfully"
    FILE_TYPE_NOT_SUPPORTED = "file_type_not_supported"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    FILE_UPLOAD_SUCCESS = "file_upload_success"
    FILE_UPLOAD_FAILED = "file_upload_failed"
    FILE_PROCESSING_FAILED = "file_processing_failed"
    FILE_PROCESSING_SUCCESS = "file_processing_success"
    NO_FILES_FOUND = "no_files_found"
    FILE_ID_ERROR = "no_file_found_with_this_id"
    PROJECT_NOT_FOUND = "project_not_found"
    INSERT_INTO_VECTORDB_ERROR = "insert_into_vector_db_error"
    INSERT_INTO_VECTORDB_SUCCESS = "insert_into_vector_db_success"
    VECTORDB_COLLECTION_RETRIEVED = "vector_db_collection_retrieved"
    SEARCH_VECTORDB_COLLECTION_ERROR = "search_vector_db_collection_error"
    SEARCH_VECTORDB_COLLECTION_SUCCESS = "search_vector_db_collection_success"
    ANSWER_RAG_QUESTION_ERROR = "answer_rag_question_error"
    ANSWER_RAG_QUESTION_SUCCESS = "answer_rag_question_success"