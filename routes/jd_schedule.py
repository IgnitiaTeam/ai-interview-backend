from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from sqlalchemy import or_, and_
from datetime import datetime, timezone

from database import get_db
from models.JobDescription import JobDescription
from models.course_master import CourseMaster
from models.model import Panel, PanelMember
from models.jd_schedule import JDSchedule
# from models.candidate import Candidate  # <-- adjust if your file/class name differs
from models.model import Candidate
from schemas.jd_schedule import (
    JobDescriptionOut, CourseMasterOut,
    JDScheduleCreate, JDScheduleOut,
    EligibleAssessmentOut  # <-- added
)

router = APIRouter(prefix="/api", tags=["jd-schedule"])


@router.get("/job-descriptions", response_model=List[JobDescriptionOut])
def list_job_descriptions(db: Session = Depends(get_db)):
    return db.query(JobDescription).all()


@router.get("/course-master", response_model=List[CourseMasterOut])
def list_courses(db: Session = Depends(get_db)):
    return db.query(CourseMaster).all()


@router.post("/jd-schedule", response_model=List[JDScheduleOut])
def create_jd_schedule(payload: JDScheduleCreate, db: Session = Depends(get_db)):
    jd = db.query(JobDescription).filter(JobDescription.id == payload.jd_id).first()
    if not jd:
        raise HTTPException(status_code=404, detail="JD not found")

    if not payload.eligibility:
        raise HTTPException(status_code=400, detail="At least one eligibility row is required")

    # ---------------------------------------
    # Resolve panel: existing or newly created
    # ---------------------------------------
    panel_id = payload.panel_id

    if panel_id is None:
        if not payload.panel_name or not payload.member_user_ids:
            raise HTTPException(
                status_code=400,
                detail="Provide either panel_id, or panel_name + member_user_ids to create a new panel"
            )

        panel = Panel(panel_name=payload.panel_name, created_by=1)
        db.add(panel)
        db.commit()
        db.refresh(panel)

        for user_id in payload.member_user_ids:
            role = "chairman" if user_id == payload.chairman_user_id else "member"
            db.add(PanelMember(panel_id=panel.id, user_id=user_id, role=role))
        db.commit()

        panel_id = panel.id
    else:
        panel = db.query(Panel).filter(Panel.id == panel_id).first()
        if not panel:
            raise HTTPException(status_code=404, detail="Selected panel not found")

    # ---------------------------------------
    # Create one merged-table row per eligibility item
    # ---------------------------------------
    schedules = []
    for item in payload.eligibility:
        schedule = JDSchedule(
            jd_id=payload.jd_id,
            panel_id=panel_id,
            course_id=item.course_id,
            year=item.year,
            endfinaldate=payload.endfinaldate
        )
        db.add(schedule)
        schedules.append(schedule)

    db.commit()
    for schedule in schedules:
        db.refresh(schedule)

    return schedules


# ===== NEW ENDPOINT =====
@router.get(
    "/jd-schedule/eligible-by-email/{email}",
    response_model=List[EligibleAssessmentOut]
)
def get_eligible_assessments_by_email(
    email: str,
    db: Session = Depends(get_db)
):

    # ==========================================================
    # CURRENT UTC TIME
    # ==========================================================

    current_datetime = datetime.now(timezone.utc)

    # If your PostgreSQL column is TIMESTAMP WITHOUT TIME ZONE,
    # convert UTC datetime to naive UTC for comparison.
    current_datetime = current_datetime.replace(tzinfo=None)

    # ==========================================================
    # FIND CANDIDATE
    # ==========================================================

    candidates = (
        db.query(Candidate)
        .filter(Candidate.email == email)
        .all()
    )

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="Candidate not found"
        )

    # ==========================================================
    # MAP (COURSE_ID, YEAR) -> COURSE_PROGRAM
    # ==========================================================

    pair_to_program = {}

    for c in candidates:

        if c.course_id and c.year:

            pair_to_program[
                (c.course_id, c.year)
            ] = c.course_program

    if not pair_to_program:
        return []

    # ==========================================================
    # COURSE + YEAR CONDITIONS
    # ==========================================================

    conditions = [
        and_(
            JDSchedule.course_id == cid,
            JDSchedule.year == yr
        )
        for (cid, yr) in pair_to_program.keys()
    ]

    # ==========================================================
    # GET ONLY VALID / NON-EXPIRED SCHEDULES
    # ==========================================================

    results = (
        db.query(
            JDSchedule,
            JobDescription.title
        )
        .join(
            JobDescription,
            JobDescription.id == JDSchedule.jd_id
        )
        .filter(
            or_(*conditions)
        )
        .filter(
            JDSchedule.endfinaldate >= current_datetime
        )
        .order_by(
            JDSchedule.endfinaldate.asc()
        )
        .all()
    )

    # ==========================================================
    # RESPONSE
    # ==========================================================

    return [
        EligibleAssessmentOut(
            schedule_id=schedule.id,
            jd_id=schedule.jd_id,
            jd_title=title,
            panel_id=schedule.panel_id,
            course_id=schedule.course_id,
            year=schedule.year,
            course_program=pair_to_program.get(
                (schedule.course_id, schedule.year),
                ''
            ),
            endfinaldate=schedule.endfinaldate,
        )
        for schedule, title in results
    ]
