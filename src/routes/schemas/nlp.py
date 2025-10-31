from pydantic import BaseModel
from typing import Optional

class PushIndexRequest(BaseModel):
    do_reset: Optional[int] = 0

class SearchIndexRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
