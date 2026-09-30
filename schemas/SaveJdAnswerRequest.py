from pydantic import BaseModel


class SaveJdAnswerRequest(BaseModel):
    answer_text: str = ""