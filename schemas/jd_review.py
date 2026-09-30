from pydantic import BaseModel
from typing import List, Optional


class JdReviewOut(BaseModel):
    sNo: int
    jdNo: str
    jdDescription: Optional[str] = None
    candidatesAttempted: int
    avgScore: Optional[float] = None
    minScore: Optional[int] = None
    maxScore: Optional[int] = None
    assessment_id: int


class CandidateReviewOut(BaseModel):
    sNo: int
    candidateId: int
    scheduleId: int   
    name: str
    email: str
    noOfAttempts: int
    status: Optional[str] = None
    score: Optional[int] = None


class AttemptSummaryOut(BaseModel):
    sNo: int
    attemptNo: int
    minScore: Optional[int] = None
    maxScore: Optional[int] = None
    avgScore: Optional[float] = None
    candidateId: int
    scheduleId: int


class AttemptQuestionOut(BaseModel):
    questionNo: int
    questionText: str
    expectedAnswer: Optional[str] = None
    answerText: Optional[str] = None
    aiAssessment: Optional[str] = None
    aiScore: Optional[int] = None

class UpdateStatusRequest(BaseModel):
    status: str