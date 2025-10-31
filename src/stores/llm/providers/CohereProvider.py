from cohere.types import embedding_type
from ..LLMInterface import LLMInterface
from ..LLMEnums import CohereEnums, DocumentTypeEnums
import cohere
import logging


class CohereProvider(LLMInterface):

    def __init__(self, api_key: str,
                 default_input_max_characters: int = 1024,
                 default_generation_output_max_tokens: int = 1024,
                 default_generation_temperature: float = 0.1):
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_output_max_tokens = default_generation_output_max_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = cohere.Client(
            api_key=self.api_key
        )

        self.logger = logging.getLogger(__name__)


    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        
    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()
    
    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                      temperature: float = None):
        
        if not self.client:
            self.logger.error("Cohere client is not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Cohere generation model is not set")
            return None

        max_tokens = max_output_tokens if max_output_tokens else self.default_generation_output_max_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        chat_history.append(
            self.construct_prompt(prompt=prompt, role=CohereEnums.USER.value)
        )

        response = self.client.chat(
            model=self.generation_model_id,
            chat_history=chat_history,
            message=self.process_text(prompt=prompt),
            temperature=temperature,
            max_tokens=max_tokens
        )

        if not response or not response.text:
            self.logger.error("Failed to generate text with Cohere")
            return None

        return response.text
    
    def embed_text(self, text: str, document_type: str = None):
        
        if not self.client:
            self.logger.error("Cohere client is not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Cohere embedding model is not set")
            return None

        input_type = CohereEnums.DOCUMENT.value
        if document_type == DocumentTypeEnums.QUERY.value:
            input_type = CohereEnums.QUERY.value

        response = self.client.embed(
            model=self.embedding_model_id,
            texts=[self.process_text(text)],
            input_type=input_type,
            embedding_types=['float']
        )

        if not response or not response.embeddings or not response.embeddings.float:
            self.logger.error("Failed to embed text with Cohere")
            return None

        return response.embeddings.float[0]
    
    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt=prompt)
        }
