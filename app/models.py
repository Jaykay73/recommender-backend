from pydantic import BaseModel
from typing import List

class RecommendRequest(BaseModel):
    movie_ids: List[int] = [278, 238, 155, 680, 13]
