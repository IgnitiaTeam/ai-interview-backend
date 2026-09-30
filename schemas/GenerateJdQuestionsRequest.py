from pydantic import BaseModel


class GenerateJdQuestionsRequest(BaseModel):
    candidate_id: int
    schedule_id: int