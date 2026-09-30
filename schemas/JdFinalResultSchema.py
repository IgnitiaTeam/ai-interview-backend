from pydantic import BaseModel
from typing import List


class JdFinalResultSchema(BaseModel):
    overall_score: int
    percentage: float
    strengths: List[str]
    improvements: List[str]
    summary: str