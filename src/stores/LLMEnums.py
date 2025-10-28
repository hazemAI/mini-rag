from enum import Enum


class LLMEnums(Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    GOOGLE = "google"
    
class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"