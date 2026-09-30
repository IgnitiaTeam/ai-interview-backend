from sqlalchemy import (
    Column,
    Integer,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint
)

from datetime import datetime, timezone

from database import Base


class JdAssessmentAttempt(Base):
    __tablename__ = "jd_assessment_attempts"

    __table_args__ = (
        UniqueConstraint(
            "attempt_id",
            "question_no",
            name="uq_attempt_question"
        ),
        {"extend_existing": True},
    )

    attempt_question_id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    attempt_id = Column(
        Integer,
        nullable=False,
        index=True
    )

    candidate_id = Column(
        Integer,
        ForeignKey("candidates.id"),
        nullable=False,
        index=True
    )

    schedule_id = Column(
        Integer,
        ForeignKey("jd_schedule.id"),
        nullable=False,
        index=True
    )

    attempt_no = Column(
        Integer,
        nullable=False
    )

    question_no = Column(
        Integer,
        nullable=False
    )

    question_text = Column(
        Text,
        nullable=False
    )

    expected_answer = Column(
        Text,
        nullable=False
    )

    time_limit = Column(
        Integer,
        nullable=False
    )

    answer_text = Column(
        Text,
        nullable=True
    )

    ai_assessment = Column(
        Text,
        nullable=True
    )

    ai_score = Column(
        Integer,
        nullable=True
    )

    started_at = Column(
        DateTime,
        nullable=True
    )

    answered_at = Column(
        DateTime,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )