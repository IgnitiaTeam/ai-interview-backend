from sqlalchemy import Column, Integer, BigInteger, String, DateTime
from datetime import datetime, timezone
from database import Base


class JDSchedule(Base):
    __tablename__ = "jd_schedule"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    jd_id = Column(Integer, nullable=False, index=True)
    panel_id = Column(Integer, nullable=False, index=True)
    course_id = Column(BigInteger, nullable=False, index=True)
    year = Column(String(50), nullable=False)
    endfinaldate = Column(DateTime, nullable=True)   # renamed from interview_datetime
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))