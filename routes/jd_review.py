from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from database import get_db
from models.jd_schedule import JDSchedule
from models.JobDescription import JobDescription
from models.model import Candidate
from models.JdAssessmentAttempt import JdAssessmentAttempt
from models.JdAssessmentCandidate import JdAssessmentCandidate
from schemas.jd_review import (
    JdReviewOut,
    CandidateReviewOut,
    AttemptSummaryOut,
    AttemptQuestionOut,
    UpdateStatusRequest,
)

router = APIRouter(prefix="/api/jd-review", tags=["JD Review"])


# ==========================================================
# SUMMARY: one row per JD, aggregated across all its schedules
# ==========================================================

@router.get("/summary", response_model=List[JdReviewOut])
def get_jd_review_summary(db: Session = Depends(get_db)):

    jds = db.query(JobDescription).all()
    result = []

    for idx, jd in enumerate(jds, start=1):

        schedule_ids = [
            s.id for s in db.query(JDSchedule).filter(JDSchedule.jd_id == jd.id).all()
        ]

        if not schedule_ids:
            result.append(JdReviewOut(
                sNo=idx,
                jdNo=jd.title,
                jdDescription=getattr(jd, "description", None),
                candidatesAttempted=0,
                avgScore=None,
                minScore=None,
                maxScore=None,
                assessment_id=jd.id,
            ))
            continue

        stats = (
            db.query(
                func.count(func.distinct(JdAssessmentAttempt.candidate_id)),
                func.avg(JdAssessmentAttempt.ai_score),
                func.min(JdAssessmentAttempt.ai_score),
                func.max(JdAssessmentAttempt.ai_score),
            )
            .filter(JdAssessmentAttempt.schedule_id.in_(schedule_ids))
            .first()
        )

        candidates_attempted, avg_score, min_score, max_score = stats

        result.append(JdReviewOut(
            sNo=idx,
            jdNo=jd.title,
            jdDescription=getattr(jd, "description", None),
            candidatesAttempted=candidates_attempted or 0,
            avgScore=round(avg_score, 1) if avg_score is not None else None,
            minScore=min_score,
            maxScore=max_score,
            assessment_id=jd.id,
        ))

    return result


# ==========================================================
# CANDIDATES FOR A JD (accepts jd_id, resolves all its schedules)
# ==========================================================

@router.get("/candidates/{jd_id}", response_model=List[CandidateReviewOut])
def get_candidates_by_jd(jd_id: int, db: Session = Depends(get_db)):

    schedule_ids = [
        s.id for s in db.query(JDSchedule).filter(JDSchedule.jd_id == jd_id).all()
    ]

    if not schedule_ids:
        return []

    candidate_ids = [
        row[0] for row in
        db.query(JdAssessmentAttempt.candidate_id)
        .filter(JdAssessmentAttempt.schedule_id.in_(schedule_ids))
        .distinct()
        .all()
    ]

    if not candidate_ids:
        return []

    result = []

    for idx, cand_id in enumerate(candidate_ids, start=1):

        candidate = db.query(Candidate).filter(Candidate.id == cand_id).first()
        if not candidate:
            continue

        attempts = (
            db.query(JdAssessmentAttempt)
            .filter(
                JdAssessmentAttempt.schedule_id.in_(schedule_ids),
                JdAssessmentAttempt.candidate_id == cand_id,
            )
            .all()
        )

        link = (
            db.query(JdAssessmentCandidate)
            .filter(
                JdAssessmentCandidate.schedule_id.in_(schedule_ids),
                JdAssessmentCandidate.candidate_id == cand_id,
            )
            .first()
        )

        latest_score = None
        if attempts:
            scored = [a.ai_score for a in attempts if a.ai_score is not None]
            latest_score = max(scored) if scored else None

        actual_schedule_id = link.schedule_id if link else attempts[0].schedule_id

        result.append(CandidateReviewOut(
            sNo=idx,
            candidateId=candidate.id,
            scheduleId=actual_schedule_id,
            name=candidate.name,
            email=candidate.email,
            noOfAttempts=len(set(a.attempt_no for a in attempts)),
            status=link.assessment_status if link else "not_registered",
            score=latest_score,
        ))

    return result


# ==========================================================
# ATTEMPTS FOR A CANDIDATE, ACROSS ALL SCHEDULES OF A JD
# ==========================================================

@router.get("/candidate/{jd_id}/{candidate_id}/attempts", response_model=List[AttemptSummaryOut])
def get_candidate_attempts(jd_id: int, candidate_id: int, db: Session = Depends(get_db)):

    schedule_ids = [
        s.id for s in db.query(JDSchedule).filter(JDSchedule.jd_id == jd_id).all()
    ]

    if not schedule_ids:
        return []

    attempt_rows = (
        db.query(JdAssessmentAttempt.attempt_no, JdAssessmentAttempt.schedule_id)
        .filter(
            JdAssessmentAttempt.schedule_id.in_(schedule_ids),
            JdAssessmentAttempt.candidate_id == candidate_id,
        )
        .distinct()
        .order_by(JdAssessmentAttempt.attempt_no)
        .all()
    )

    result = []

    for idx, (attempt_no, sched_id) in enumerate(attempt_rows, start=1):

        stats = (
            db.query(
                func.min(JdAssessmentAttempt.ai_score),
                func.max(JdAssessmentAttempt.ai_score),
                func.avg(JdAssessmentAttempt.ai_score),
            )
            .filter(
                JdAssessmentAttempt.schedule_id == sched_id,
                JdAssessmentAttempt.candidate_id == candidate_id,
                JdAssessmentAttempt.attempt_no == attempt_no,
            )
            .first()
        )

        min_score, max_score, avg_score = stats

        result.append(AttemptSummaryOut(
            sNo=idx,
            attemptNo=attempt_no,
            minScore=min_score,
            maxScore=max_score,
            avgScore=round(avg_score, 1) if avg_score is not None else None,
            candidateId=candidate_id,
            scheduleId=sched_id,
        ))

    return result


# ==========================================================
# QUESTIONS FOR A SPECIFIC ATTEMPT
# ==========================================================

@router.get("/candidate/{schedule_id}/{candidate_id}/attempt/{attempt_no}/questions", response_model=List[AttemptQuestionOut])
def get_attempt_questions(schedule_id: int, candidate_id: int, attempt_no: int, db: Session = Depends(get_db)):

    rows = (
        db.query(JdAssessmentAttempt)
        .filter(
            JdAssessmentAttempt.schedule_id == schedule_id,
            JdAssessmentAttempt.candidate_id == candidate_id,
            JdAssessmentAttempt.attempt_no == attempt_no,
        )
        .order_by(JdAssessmentAttempt.question_no)
        .all()
    )

    return [
        AttemptQuestionOut(
            questionNo=r.question_no,
            questionText=r.question_text,
            expectedAnswer=r.expected_answer,
            answerText=r.answer_text,
            aiAssessment=r.ai_assessment,
            aiScore=r.ai_score,
        )
        for r in rows
    ]


# ==========================================================
# UPDATE CANDIDATE STATUS (Stop / Continue)
# ==========================================================

@router.patch("/candidate/{schedule_id}/{candidate_id}/status")
def update_candidate_status(
    schedule_id: int,
    candidate_id: int,
    payload: UpdateStatusRequest,
    db: Session = Depends(get_db)
):
    link = (
        db.query(JdAssessmentCandidate)
        .filter(
            JdAssessmentCandidate.schedule_id == schedule_id,
            JdAssessmentCandidate.candidate_id == candidate_id,
        )
        .first()
    )

    if not link:
        raise HTTPException(status_code=404, detail="Candidate assessment link not found")

    link.assessment_status = payload.status
    db.commit()
    db.refresh(link)

    return {"success": True, "status": link.assessment_status}
# from fastapi import APIRouter, Depends, HTTPException
# from sqlalchemy.orm import Session
# from sqlalchemy import func
# from typing import List

# from database import get_db
# from models.jd_schedule import JDSchedule
# from models.JobDescription import JobDescription
# from models.model import Candidate
# from models.JdAssessmentAttempt import JdAssessmentAttempt
# from models.JdAssessmentCandidate import JdAssessmentCandidate
# from schemas.jd_review import (
#     JdReviewOut,
#     CandidateReviewOut,
#     AttemptSummaryOut,
#     AttemptQuestionOut,
#     UpdateStatusRequest, 
# )

# router = APIRouter(prefix="/api/jd-review", tags=["JD Review"])


# @router.get("/summary", response_model=List[JdReviewOut])
# def get_jd_review_summary(db: Session = Depends(get_db)):

#     jds = db.query(JobDescription).all()
#     result = []

#     for idx, jd in enumerate(jds, start=1):

#         schedule_ids = [
#             s.id for s in db.query(JDSchedule).filter(JDSchedule.jd_id == jd.id).all()
#         ]

#         if not schedule_ids:
#             result.append(JdReviewOut(
#                 sNo=idx,
#                 jdNo=jd.title,
#                 jdDescription=getattr(jd, "description", None),
#                 candidatesAttempted=0,
#                 avgScore=None,
#                 minScore=None,
#                 maxScore=None,
#                 assessment_id=jd.id,
#             ))
#             continue

#         stats = (
#             db.query(
#                 func.count(func.distinct(JdAssessmentAttempt.candidate_id)),
#                 func.avg(JdAssessmentAttempt.ai_score),
#                 func.min(JdAssessmentAttempt.ai_score),
#                 func.max(JdAssessmentAttempt.ai_score),
#             )
#             .filter(JdAssessmentAttempt.schedule_id.in_(schedule_ids))
#             .first()
#         )

#         candidates_attempted, avg_score, min_score, max_score = stats

#         result.append(JdReviewOut(
#             sNo=idx,
#             jdNo=jd.title,
#             jdDescription=getattr(jd, "description", None),
#             candidatesAttempted=candidates_attempted or 0,
#             avgScore=round(avg_score, 1) if avg_score is not None else None,
#             minScore=min_score,
#             maxScore=max_score,
#             assessment_id=jd.id,   # now this is job_descriptions.id
#         ))

#     return result

#     schedules = db.query(JDSchedule).all()
#     result = []

#     for idx, sched in enumerate(schedules, start=1):

#         jd = db.query(JobDescription).filter(JobDescription.id == sched.jd_id).first()

#         stats = (
#             db.query(
#                 func.count(func.distinct(JdAssessmentAttempt.candidate_id)),
#                 func.avg(JdAssessmentAttempt.ai_score),
#                 func.min(JdAssessmentAttempt.ai_score),
#                 func.max(JdAssessmentAttempt.ai_score),
#             )
#             .filter(JdAssessmentAttempt.schedule_id == sched.id)
#             .first()
#         )

#         candidates_attempted, avg_score, min_score, max_score = stats

#         result.append(JdReviewOut(
#             sNo=idx,
#             jdNo=jd.title if jd else f"JD-{sched.jd_id}",
#             jdDescription=getattr(jd, "description", None),
#             candidatesAttempted=candidates_attempted or 0,
#             avgScore=round(avg_score, 1) if avg_score is not None else None,
#             minScore=min_score,
#             maxScore=max_score,
#             assessment_id=sched.id,
#         ))

#     return result


# @router.get("/candidates/{schedule_id}", response_model=List[CandidateReviewOut])
# def get_candidates_by_jd(schedule_id: int, db: Session = Depends(get_db)):

#     links = (
#         db.query(JdAssessmentCandidate)
#         .filter(JdAssessmentCandidate.schedule_id == schedule_id)
#         .all()
#     )

#     if not links:
#         return []

#     result = []

#     for idx, link in enumerate(links, start=1):

#         candidate = db.query(Candidate).filter(Candidate.id == link.candidate_id).first()
#         if not candidate:
#             continue

#         attempts = (
#             db.query(JdAssessmentAttempt)
#             .filter(
#                 JdAssessmentAttempt.schedule_id == schedule_id,
#                 JdAssessmentAttempt.candidate_id == link.candidate_id,
#             )
#             .all()
#         )

#         latest_score = None
#         if attempts:
#             scored = [a.ai_score for a in attempts if a.ai_score is not None]
#             latest_score = max(scored) if scored else None

#         result.append(CandidateReviewOut(
#     sNo=idx,
#     candidateId=candidate.id,
#     scheduleId=link.schedule_id,   # <-- ADD THIS LINE
#     name=candidate.name,
#     email=candidate.email,
#     noOfAttempts=len(set(a.attempt_no for a in attempts)),
#     status=link.assessment_status,
#     score=latest_score,
# ))

#     return result


# @router.get("/candidate/{schedule_id}/{candidate_id}/attempts", response_model=List[AttemptSummaryOut])
# def get_candidate_attempts(schedule_id: int, candidate_id: int, db: Session = Depends(get_db)):

#     attempt_numbers = (
#         db.query(JdAssessmentAttempt.attempt_no)
#         .filter(
#             JdAssessmentAttempt.schedule_id == schedule_id,
#             JdAssessmentAttempt.candidate_id == candidate_id,
#         )
#         .distinct()
#         .order_by(JdAssessmentAttempt.attempt_no)
#         .all()
#     )

#     result = []

#     for idx, (attempt_no,) in enumerate(attempt_numbers, start=1):

#         stats = (
#             db.query(
#                 func.min(JdAssessmentAttempt.ai_score),
#                 func.max(JdAssessmentAttempt.ai_score),
#                 func.avg(JdAssessmentAttempt.ai_score),
#             )
#             .filter(
#                 JdAssessmentAttempt.schedule_id == schedule_id,
#                 JdAssessmentAttempt.candidate_id == candidate_id,
#                 JdAssessmentAttempt.attempt_no == attempt_no,
#             )
#             .first()
#         )

#         min_score, max_score, avg_score = stats

#         result.append(AttemptSummaryOut(
#             sNo=idx,
#             attemptNo=attempt_no,
#             minScore=min_score,
#             maxScore=max_score,
#             avgScore=round(avg_score, 1) if avg_score is not None else None,
#             candidateId=candidate_id,
#             scheduleId=schedule_id,
#         ))

#     return result


# @router.get("/candidate/{schedule_id}/{candidate_id}/attempt/{attempt_no}/questions", response_model=List[AttemptQuestionOut])
# def get_attempt_questions(schedule_id: int, candidate_id: int, attempt_no: int, db: Session = Depends(get_db)):

#     rows = (
#         db.query(JdAssessmentAttempt)
#         .filter(
#             JdAssessmentAttempt.schedule_id == schedule_id,
#             JdAssessmentAttempt.candidate_id == candidate_id,
#             JdAssessmentAttempt.attempt_no == attempt_no,
#         )
#         .order_by(JdAssessmentAttempt.question_no)
#         .all()
#     )

#     return [
#         AttemptQuestionOut(
#             questionNo=r.question_no,
#             questionText=r.question_text,
#             expectedAnswer=r.expected_answer,
#             answerText=r.answer_text,
#             aiAssessment=r.ai_assessment,
#             aiScore=r.ai_score,
#         )
#         for r in rows
#     ]
# @router.patch("/candidate/{schedule_id}/{candidate_id}/status")
# def update_candidate_status(
#     schedule_id: int,
#     candidate_id: int,
#     payload: UpdateStatusRequest,
#     db: Session = Depends(get_db)
# ):
#     link = (
#         db.query(JdAssessmentCandidate)
#         .filter(
#             JdAssessmentCandidate.schedule_id == schedule_id,
#             JdAssessmentCandidate.candidate_id == candidate_id,
#         )
#         .first()
#     )

#     if not link:
#         raise HTTPException(status_code=404, detail="Candidate assessment link not found")

#     link.assessment_status = payload.status
#     db.commit()
#     db.refresh(link)

#     return {"success": True, "status": link.assessment_status}