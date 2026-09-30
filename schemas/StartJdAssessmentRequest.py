from pydantic import BaseModel


class StartJdAssessmentRequest(BaseModel):
    email: str
    schedule_id: int