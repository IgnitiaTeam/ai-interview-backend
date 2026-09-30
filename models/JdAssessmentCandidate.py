from sqlalchemy import (
    Column,
    Integer,
    Boolean,
    String,
    ForeignKey
)

from database import Base


class JdAssessmentCandidate(Base):
    __tablename__ = "jd_assessment_candidate"
    __table_args__ = {"extend_existing": True}

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
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

    assessment_status = Column(
        String(20),
        nullable=False,
        default="not_started"
    )

    isapply = Column(
        Boolean,
        nullable=False,
        default=False
    )