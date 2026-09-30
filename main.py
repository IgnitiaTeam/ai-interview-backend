# 🔥 LOAD ENV FIRST (VERY IMPORTANT)
import shutil
import time
import traceback

import deepface
from dotenv import load_dotenv
import os
import re

from sqlalchemy import func
import websocket

from models.InterviewVideoAnalysis import InterviewVideoAnalysis
from models.PanelCandidate import PanelCandidate
from models.course_master import CourseMaster
from models.SelfAssessmentResult import SelfAssessmentResult
from models.SelfAssessmentAnswer import SelfAssessmentAnswer
from schemas import save_answer_request
from schemas.VideoAnalysisRequest import VideoAnalysisRequest
from schemas.grading_schema import GradingSchema, FinalAssessmentSchema
from schemas.schedule_interview_request import ScheduleInterviewRequest
from models.subject_master import SubjectMaster
from routes.jd_schedule import router as jd_schedule_router
load_dotenv()


# ---------------- IMPORTS ----------------
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, APIRouter, Form, WebSocket, WebSocketDisconnect, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Query, Session

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from ultralytics import YOLO
import uuid
import random
import json
import asyncio
import websockets

from database import engine, Base, get_db, SessionLocal
from routes.answer_routes import router as answer_router
from routes.memberdashboard_routes import router as memberdashboard_router
from routes.auth_routes import router as auth_router
from core.session_store import INTERVIEW_SESSIONS
from routes.jd_route import router as jd_route
from routes.jd_review import router as jd_review_router

# 🛠️ CLEANED MODELS BLOCK
from models.question import Question, CourseProgram # Pulls your validated enum directly from question.py
from models.model import (Candidate,Panel,PanelMember,Interview,User,JDAssessment,JDQuestion)
from models.answer import Answer

# from openai import OpenAI
from google import genai
from google.genai import types



from typing import Annotated, Any, Dict, Optional, List
from pydantic import BaseModel
from datetime import datetime
from routes import auth_routes as auth

# ---------------- APP INIT ----------------
app = FastAPI()

Base.metadata.create_all(bind=engine)

# ============================================================
# AI PROCTORING MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AI_MODELS_DIR = BASE_DIR / "ai_models"

if not AI_MODELS_DIR.exists():
    AI_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

# ------------------------------------------------------------
# MODEL FILE PATHS
# ------------------------------------------------------------

FACE_DETECTOR_MODEL = (
    AI_MODELS_DIR / "face_detector.tflite"
)

FACE_LANDMARKER_MODEL = (
    AI_MODELS_DIR / "face_landmarker.task"
)

print(
    "📁 AI MODELS:",
    AI_MODELS_DIR
)

print(
    "📄 Face detector:",
    FACE_DETECTOR_MODEL
)

print(
    "📄 Face landmarker:",
    FACE_LANDMARKER_MODEL
)


# ============================================================
# VERIFY MODEL FILES
# ============================================================

if not FACE_DETECTOR_MODEL.exists():

    print(
        "❌ ERROR: face_detector.task NOT FOUND"
    )

    print(
        "Expected:",
        FACE_DETECTOR_MODEL
    )

else:

    print(
        "✅ face_detector.task found"
    )


if not FACE_LANDMARKER_MODEL.exists():

    print(
        "❌ ERROR: face_landmarker.task NOT FOUND"
    )

    print(
        "Expected:",
        FACE_LANDMARKER_MODEL
    )

else:

    print(
        "✅ face_landmarker.task found"
    )


# ============================================================
# YOLO PROCTORING MODEL
# ============================================================

print(
    "🔄 Loading YOLO proctoring model..."
)

try:

    yolo_model = YOLO(
        "yolo11n.pt"
    )

    print(
        "✅ YOLO proctoring model loaded"
    )

except Exception as e:

    print(
        "❌ Failed to load YOLO model:",
        str(e)
    )

    traceback.print_exc()

    yolo_model = None


# ============================================================
# MEDIAPIPE FACE DETECTOR
# ============================================================

print(
    "🔄 Loading MediaPipe Face Detector (optional)..."
)

face_detector = None

try:

    face_detector_options = (
        vision.FaceDetectorOptions(

            base_options=python.BaseOptions(
                model_asset_path=str(
                    FACE_DETECTOR_MODEL
                )
            ),

            min_detection_confidence=0.5
        )
    )

    face_detector = (
        vision.FaceDetector.create_from_options(
            face_detector_options
        )
    )

    print(
        "✅ MediaPipe Face Detector loaded"
    )

except Exception as e:

    print(
        "⚠️ MediaPipe Face Detector could not be loaded. "
        "FaceLandmarker fallback will be used."
    )

    print(
        str(e)
    )

    traceback.print_exc()

    face_detector = None


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================

print(
    "🔄 Loading MediaPipe Face Landmarker..."
)

face_landmarker = None

try:

    face_landmarker_options = (
        vision.FaceLandmarkerOptions(

            base_options=python.BaseOptions(
                model_asset_path=str(
                    FACE_LANDMARKER_MODEL
                )
            ),

            running_mode=(
                vision.RunningMode.IMAGE
            ),

            num_faces=2,

            min_face_detection_confidence=0.5,

            min_face_presence_confidence=0.5,

            min_tracking_confidence=0.5,

            output_face_blendshapes=True,

            output_facial_transformation_matrixes=True
        )
    )

    face_landmarker = (
        vision.FaceLandmarker.create_from_options(
            face_landmarker_options
        )
    )

    print(
        "✅ MediaPipe Face Landmarker loaded"
    )

except Exception as e:

    print(
        "❌ Failed to load MediaPipe Face Landmarker:"
    )

    print(
        str(e)
    )

    traceback.print_exc()

    face_landmarker = None


# ============================================================
# FINAL PROCTORING MODEL STATUS
# ============================================================

if face_landmarker is None:

    print(
        "❌ CRITICAL: MediaPipe Face Landmarker is unavailable. "
        "Proctoring face checks cannot run."
    )

else:

    print(
        "✅ Proctoring face analysis ready "
        "(FaceLandmarker active)."
    )

if face_detector is None:

    print(
        "ℹ️ Separate FaceDetector is unavailable; "
        "FaceLandmarker will be used for face counting "
        "and bounding-box estimation."
    )


# ============================================================
# PROCTORING CONFIGURATION
# ============================================================

PROCTORING_FACE_CONFIDENCE = 0.5

HEAD_YAW_THRESHOLD = 25.0

HEAD_PITCH_THRESHOLD = 25.0

HEAD_ROLL_THRESHOLD = 30.0

MIN_FACE_AREA_RATIO = 0.03

YOLO_CONFIDENCE = 0.45

EYE_GAZE_HORIZONTAL_THRESHOLD = 0.22
EYE_GAZE_VERTICAL_THRESHOLD = 0.22


# ---------------- GEMINI CLIENT ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

# 🔥 FFmpeg path
# os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-8.1-essentials_build\bin"
os.environ["PATH"] += os.pathsep + os.getenv("FFMPEG_PATH", "/usr/bin")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"


# 🔥 Recording path
# RECORDING_BASE_PATH = "C:/Users/saipa/OneDrive/Desktop/Recordings"
RECORDING_BASE_PATH = os.getenv("RECORDING_BASE_PATH", "/data/recording")
os.makedirs(RECORDING_BASE_PATH, exist_ok=True)

app.mount(
    "/videos",
    StaticFiles(directory=RECORDING_BASE_PATH),
    name="videos"
)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","http://172.146.1.40:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- ROUTERS ----------------
app.include_router(answer_router)
app.include_router(memberdashboard_router)
app.include_router(auth_router)
app.include_router(jd_route)
app.include_router(jd_schedule_router)
app.include_router(jd_review_router)
# app.include_router(course_route)

# mp_face_detection = mp.solutions.face_detection

# ---------------- OPENAI CLIENT ----------------
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Check your .env file.")

# client = OpenAI(api_key=api_key)

print("✅ API KEY LOADED SUCCESSFULLY")
# ---------------- Assembly Ai CLIENT ----------------

ASSEMBLY_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

if not ASSEMBLY_API_KEY:
    raise ValueError("❌ ASSEMBLYAI_API_KEY not found in .env")

# ---------------- HELPER ----------------
def is_repetitive(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 6:
        return False
    unique_words = set(words)
    ratio = 1 - (len(unique_words) / len(words))
    return ratio > 0.6

# ---------------- START INTERVIEW ----------------
@app.post("/start-interview")
def start_interview(db: Session = Depends(get_db)):

    questions = db.query(Question).all()
    if not questions:
        raise HTTPException(status_code=404, detail="No questions found")

    random.shuffle(questions)

    interview_id = str(uuid.uuid4())

    INTERVIEW_SESSIONS[interview_id] = {
        "questions": [q.id for q in questions],
        "current_index": 0,
        "completed": False
    }

    return {"interview_id": interview_id}

# 

# @app.get("/next-question/{interview_id}")
# def get_next_question(interview_id: int, db: Session = Depends(get_db)):

#     interview_key = str(interview_id)

#     # ---------------- INIT SESSION IF NOT EXISTS ----------------
#     if interview_key not in INTERVIEW_SESSIONS:

#         # 🔥 Fetch in sequential order (by ID)
#         questions = db.query(Question).order_by(Question.id).all()

#         if not questions:
#             raise HTTPException(status_code=404, detail="No questions found")

#         INTERVIEW_SESSIONS[interview_key] = {
#             "questions": [q.id for q in questions],
#             "current_index": 0,
#             "completed": False
#         }

#     session = INTERVIEW_SESSIONS[interview_key]

#     # ---------------- CHECK COMPLETION ----------------
#     if session["current_index"] >= len(session["questions"]):
#         session["completed"] = True
#         return {"message": "Interview completed"}

#     # ---------------- FETCH QUESTION ----------------
#     question_id = session["questions"][session["current_index"]]

#     question = db.query(Question).filter(Question.id == question_id).first()

#     if not question:
#         raise HTTPException(status_code=404, detail="Question not found")

#     # ---------------- INCREMENT INDEX ----------------
#     session["current_index"] += 1

#     return question

@app.get("/next-question/{interview_id}")
def get_next_question(interview_id: str, category: str, db: Session = Depends(get_db)):
    interview_key = str(interview_id)

    # ---------------- INIT SESSION IF NOT EXISTS ----------------
    if interview_key not in INTERVIEW_SESSIONS:
        # 🔥 Filter questions ONLY by the specific session_category
        questions = db.query(Question).filter(Question.category == category).order_by(Question.id).all()

        if not questions:
            raise HTTPException(status_code=404, detail="No questions found for this document session.")

        INTERVIEW_SESSIONS[interview_key] = {
            "questions": [q.id for q in questions],
            "current_index": 0,
            "completed": False
        }

    session = INTERVIEW_SESSIONS[interview_key]

    # ---------------- CHECK COMPLETION ----------------
    if session["current_index"] >= len(session["questions"]):
        session["completed"] = True
        return {"message": "Interview completed"}

    # ---------------- FETCH QUESTION ----------------
    question_id = session["questions"][session["current_index"]]
    question = db.query(Question).filter(Question.id == question_id).first()

    # Increment for next time
    session["current_index"] += 1
 
    return question


# ---------------- SAVE FULL VIDEO ----------------
# @app.post("/save-video/{interview_id}")
# async def save_video(
#     interview_id: int,
#     file: UploadFile = File(...),
#     db: Session = Depends(get_db)
# ):
#     try:
#         # 🔥 Ensure directory exists
#         os.makedirs(RECORDING_BASE_PATH, exist_ok=True)

#         # 🔥 Create filename (IMPORTANT CHANGE)
#         filename = f"{interview_id}.webm"

#         # 🔥 Build full path
#         video_path = os.path.join(RECORDING_BASE_PATH, filename)

#         # 🔥 Normalize (OS safe)
#         video_path = os.path.normpath(video_path)

#         # 🔥 Save file
#         contents = await file.read()
#         with open(video_path, "wb") as f:
#             f.write(contents)

#         # ---------------- FETCH INTERVIEW ----------------
#         interview = db.query(Interview).filter(Interview.id == interview_id).first()

#         if not interview:
#             raise HTTPException(status_code=404, detail="Interview not found")

#         # ---------------- UPDATE INTERVIEW ----------------
#         # ✅ STORE ONLY FILE NAME (NOT FULL PATH)
#         interview.video_path = filename
#         interview.status = "Completed"

#         # ---------------- UPDATE CANDIDATE ----------------
#         candidate = db.query(Candidate).filter(
#             Candidate.id == interview.candidate_id
#         ).first()

#         if candidate:
#             candidate.video_path = filename

#         db.commit()

#         # ✅ Return full accessible URL (optional but useful)
#         video_url = f"http://127.0.0.1:8000/videos/{filename}"

#         return {
#             "message": "Video saved and updated successfully",
#             "video_filename": filename,
#             "video_url": video_url
#         }

#     except Exception as e:
#         print("❌ SAVE VIDEO ERROR:", e)
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/ask-ai")
# async def ask_ai(question: str, answer: str):

#     try:
#         prompt = f"""
# You are an expert technical assistant.

# A candidate answered a question, but the speech-to-text output has errors.

# Your job:
# - Fix incorrect words (like "R pictures" → "microservices")
# - Correct grammar
# - Remove repetition
# - Keep the original meaning
# - Do NOT add new information

# QUESTION:
# {question}

# RAW ANSWER:
# {answer}

# Return ONLY the corrected sentence.
# """

#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You fix speech recognition errors using context."},
#                 {"role": "user", "content": prompt}
#             ]
#         )

#         corrected = response.choices[0].message.content.strip()

#         return {
#             "corrected_text": corrected
#         }

#     except Exception as e:
#         print("❌ ChatGPT Error:", e)
#         return {"corrected_text": answer}
           

# ============================================================
# SIMPLE FACE DETECTION
# ============================================================

@app.post("/detect-face")
async def detect_face(
    file: UploadFile = File(...)
):

    try:

        frame = await read_uploaded_frame(file)

        analysis = analyze_face(frame)

        return {
            "success": True,
            "face_detected": analysis["face_visible"],
            "face_count": analysis["face_count"],
            "face_confidence": analysis["face_confidence"]
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "❌ Face detection error:",
            str(e)
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Face detection failed: {str(e)}"
        )


app.include_router(auth.router, prefix="/api/auth")


# mp_face_detection = mp.solutions.face_detection

# ============================================
# PYDANTIC MODELS
# ============================================

class InterviewRequest(BaseModel):
    applied_role: str
    interview_date: str
    start_time: str
    end_time: str
    
    # Panel - either existing or new
    panel_id: Optional[int] = None
    panel_name: Optional[str] = None
    chairman_user_id: Optional[int] = None
    member_user_ids: Optional[List[int]] = None
    
    # Student details
    student_full_name: str
    student_father_mother_name: Optional[str] = None
    student_dob: Optional[str] = None
    student_gender: Optional[str] = None
    student_category: Optional[str] = None
    student_mobile: Optional[str] = None
    student_alt_mobile: Optional[str] = None
    student_email: Optional[str] = None
    student_current_address: Optional[str] = None
    student_permanent_address: Optional[str] = None
    student_course_program: Optional[str] = None
    student_department_branch: Optional[str] = None
    student_university: Optional[str] = None
    student_enrollment_no: Optional[str] = None
    student_academic_year: Optional[str] = None
    student_cgpa: Optional[str] = None
    student_skills: Optional[str] = None
    student_certifications: Optional[str] = None
    student_projects: Optional[str] = None
    student_experience: Optional[str] = None
    student_strengths: Optional[str] = None
    student_weaknesses: Optional[str] = None
    student_career_objective: Optional[str] = None
    student_declaration: bool
    interview_category: Optional[str] = None

# ============================================
# HELPER FUNCTIONS
# ============================================

def generate_interview_id():
    year = datetime.now().year
    num = random.randint(1000, 9999)
    return f"IVW-{year}-{num}"

# ============================================
# ROUTES
# ============================================

@app.get("/")
def root():
    return {"message": "Interview Scheduling API", "status": "running"}

# ============================================
# USER ROUTES (for panel members)
# ============================================

@app.get("/api/users/interviewers")
def get_interviewers(db: Session = Depends(get_db)):
    users = db.query(User).all()
    result = []
    for user in users:
        result.append({
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "designation": user.designation,
            "role": user.role
        })
    return {"success": True, "data": result}

# ============================================
# PANEL ROUTES
# ============================================

@app.get("/api/panels")
def get_panels(db: Session = Depends(get_db)):
    panels = db.query(Panel).all()
    result = []
    for panel in panels:
        members = db.query(PanelMember).filter(PanelMember.panel_id == panel.id).all()
        result.append({
            "id": panel.id,
            "panel_name": panel.panel_name,
            "member_count": len(members),
            "created_at": panel.created_at
        })
    return {"success": True, "data": result}

# ============================================
# INTERVIEW ROUTES
# ============================================

# @app.post("/api/schedule-interview")
# def schedule_interview(data: InterviewRequest, db: Session = Depends(get_db)):
#     try:
#         # Generate the unique identifier tracking key for the session handshake
#         interview_id = generate_interview_id()
        
#         # 🛠️ Validate and convert incoming frontend string to your Python CourseProgram Enum
#         db_course = None
#         if data.student_course_program:
#             try:
#                 db_course = CourseProgram(data.student_course_program)
#             except ValueError:
#                 raise HTTPException(
#                     status_code=400, 
#                     detail=f"Invalid course program validation match. Choose from: {[c.value for c in CourseProgram]}"
#                 )

#         # 1. Create the candidate with the course program assigned directly to its proper model field
#         candidate = Candidate(
#             name=data.student_full_name,
#             email=data.student_email,
#             phone=data.student_mobile,
#             interview_id=interview_id,
#             student_father_mother_name=data.student_father_mother_name,
#             student_dob=data.student_dob,
#             student_gender=data.student_gender,
#             student_category=data.student_category,
#             student_alt_mobile=data.student_alt_mobile,
#             student_current_address=data.student_current_address,
#             student_permanent_address=data.student_permanent_address,
#             student_course_program=data.student_course_program,
            
#             # 🔥 SAVED DIRECTLY IN CANDIDATE TABLE AS ENUM
#             course_program=db_course,
            
#             student_department_branch=data.student_department_branch,
#             student_university=data.student_university,
#             student_enrollment_no=data.student_enrollment_no,
#             student_academic_year=data.student_academic_year,
#             student_cgpa=data.student_cgpa,
#             student_skills=data.student_skills,
#             student_certifications=data.student_certifications,
#             student_projects=data.student_projects,
#             student_experience=data.student_experience,
#             student_strengths=data.student_strengths,
#             student_weaknesses=data.student_weaknesses,
#             student_career_objective=data.student_career_objective,
#             student_declaration=data.student_declaration
#         )
        
#         db.add(candidate)
#         db.commit()
#         db.refresh(candidate)
        
#         # 2. Handle routing / scheduling panel structures
#         if data.panel_id:
#             panel = db.query(Panel).filter(Panel.id == data.panel_id).first()
#             if not panel:
#                 raise HTTPException(status_code=404, detail="Selected evaluation panel not found")
#         else:
#             # Fallback named generation matching the course value
#             fallback_name = f"{db_course.value if db_course else 'General'} Panel"
#             panel = Panel(
#                 panel_name=data.panel_name or fallback_name,
#                 created_by=1  
#             )
#             db.add(panel)
#             db.commit()
#             db.refresh(panel)
            
#             # Add corresponding assigned panel members
#             if data.member_user_ids:
#                 for user_id in data.member_user_ids:
#                     role = "chairman" if user_id == data.chairman_user_id else "member"
#                     panel_member = PanelMember(
#                         panel_id=panel.id,
#                         user_id=user_id,
#                         role=role
#                     )
#                     db.add(panel_member)
#                 db.commit()
        
#         # 3. Complete structural interview event generation setup mapping to your candidate
#         interview = Interview(
#             candidate_id=candidate.id,
#             panel_id=panel.id,
#             scheduled_at=f"{data.interview_date} {data.start_time}",
#             status="Scheduled",
#             created_by=1,
#             interview_category=data.interview_category or (db_course.value if db_course else "General"),   
#             interview_id=interview_id,    
#         )
        
#         db.add(interview)
#         db.commit()
#         db.refresh(interview)
        
#         return {
#             "success": True,
#             "message": "Interview scheduled successfully and course assigned directly to candidate record.",
#             "interview_id": interview_id,
#             "data": {
#                 "candidate_id": candidate.id,
#                 "panel_id": panel.id,
#                 "interview_id": interview_id
#             }
#         }
        
#     except HTTPException as he:
#         db.rollback()
#         raise he
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/interviews")
def get_all_interviews(db: Session = Depends(get_db)):
    interviews = db.query(Interview).all()
    result = []
    
    for interview in interviews:
        candidate = db.query(Candidate).filter(Candidate.id == interview.candidate_id).first()
        panel = db.query(Panel).filter(Panel.id == interview.panel_id).first()
        
        result.append({
            "id": interview.id,
            "interview_id": candidate.interview_id if candidate else None,
            "candidate_name": candidate.name if candidate else None,
            "email": candidate.email if candidate else None,
            "phone": candidate.phone if candidate else None,
            "panel_name": panel.panel_name if panel else None,
            "scheduled_at": interview.scheduled_at,
            "status": interview.status,
            "created_at": interview.created_at.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return {"success": True, "data": result}

@app.get("/api/interviews/latest")
def get_latest_interview(db: Session = Depends(get_db)):

    latest = db.query(Interview).order_by(Interview.id.desc()).first()

    if not latest:
        return {"success": False, "message": "No interviews found"}

    candidate = latest.candidate
    panel = latest.panel
    members = db.query(PanelMember).filter(
        PanelMember.panel_id == panel.id
    ).all()

    return {
        "success": True,
        "data": {
            "candidate": candidate,
            "interview": latest,
            "panel": panel,
            "members": members
        }
    }


@app.get("/api/interviews/{id}")
def get_interview(id: int, db: Session = Depends(get_db)):

    interview = db.query(Interview).filter(Interview.id == id).first()

    if not interview:
        return {"success": False, "message": "Interview not found"}

    candidate = interview.candidate
    panel = interview.panel
    members = db.query(PanelMember).filter(
        PanelMember.panel_id == panel.id
    ).all()

    return {
        "success": True,
        "data": {
            "candidate": candidate,
            "interview": interview,
            "panel": panel,
            "members": members
        }
    }


@app.delete("/api/interviews/{interview_id}")
def delete_interview(interview_id: str, db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.interview_id == interview_id).first()
    if not candidate:
        return {"success": False, "message": "Interview not found"}
    
    interview = db.query(Interview).filter(Interview.candidate_id == candidate.id).first()
    
    db.delete(interview)
    db.delete(candidate)
    db.commit()
    
    return {"success": True, "message": "Interview deleted successfully"}


# BASE_URL = "http://127.0.0.1:8000"
# ✅ Use env variable
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
# ============================================
# FILE UPLOAD ROUTES
# ============================================

@app.post("/api/candidates/{candidate_id}/upload")
async def upload_candidate_file(
    candidate_id: int,
    file_type: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 🔍 Check candidate exists
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # 📁 Create uploads folder if not exists
    os.makedirs("uploads", exist_ok=True)

    # 🧠 Create proper saved filename
    saved_filename = f"{candidate_id}_{file_type}_{file.filename}"
    file_location = os.path.join("uploads", saved_filename)

    # 💾 Save file
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)

    # 🗂️ Update DB with correct filename
    if file_type == "photo":
        candidate.photo_filename = saved_filename
    elif file_type == "resume":
        candidate.resume_filename = saved_filename
    elif file_type == "idproof":
        candidate.idproof_filename = saved_filename
    elif file_type == "certificates":
        candidate.certificates_filename = saved_filename
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

    db.commit()

    # 🔗 Return file URL (important for frontend)
    file_url = f"{BASE_URL}/uploads/{saved_filename}"

    return {
        "success": True,
        "message": "File uploaded successfully",
        "file_url": file_url
    }


@app.get("/interview/{interview_id}")
def get_interview(interview_id: int, db: Session = Depends(get_db)):

    interview = db.query(Interview).filter(Interview.id == interview_id).first()

    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")

    return {
        "id": interview.id,

        # 🔥 candidate info
        "candidate": {
            "id": interview.candidate.id,
            "name": interview.candidate.name,
            "email": interview.candidate.email,
            "video_path": interview.candidate.video_path
        } if interview.candidate else None,

        # 🔥 panel info
        "panel": {
            "id": interview.panel.id,
            "panel_name": interview.panel.panel_name
        } if interview.panel else None,

        # 🔥 main fields
        "candidate_id": interview.candidate_id,
        "panel_id": interview.panel_id,
        "scheduled_at": interview.scheduled_at,
        "status": interview.status,
        "video_path": interview.video_path
    }   

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):

    await websocket.accept()
    print("✅ Angular client connected")

    assembly_url = (
        "wss://streaming.assemblyai.com/v3/ws"
        "?sample_rate=16000&speech_model=u3-rt-pro"
    )

    silence_task = None
    candidate_started = False

    try:

        async with websockets.connect(
            assembly_url,
            additional_headers={
                "Authorization": ASSEMBLY_API_KEY
            }
        ) as assembly_ws:

            print("✅ Connected to AssemblyAI")

            # ============================================================
            # SILENCE COUNTDOWN
            # ============================================================

            async def silence_countdown():

                try:

                    await asyncio.sleep(10)

                    print(
                        "⏰ Candidate silent for 10 seconds"
                    )

                    await websocket.send_text(
                        json.dumps({
                            "type": "silence_timeout"
                        })
                    )

                except asyncio.CancelledError:

                    print(
                        "🔄 Silence timer reset"
                    )

                except Exception:

                    print(
                        "❌ Error inside silence timer"
                    )

                    traceback.print_exc()

            # ============================================================
            # SEND AUDIO
            # Angular → Backend → AssemblyAI
            # ============================================================

            async def send_audio():

                nonlocal silence_task

                try:

                    while True:

                        chunk = (
                            await websocket.receive_bytes()
                        )

                        if not chunk:
                            continue

                        print(
                            f"📤 Audio Chunk: "
                            f"{len(chunk)} bytes"
                        )

                        await assembly_ws.send(
                            chunk
                        )

                except WebSocketDisconnect:

                    print(
                        "❌ Angular WebSocket disconnected"
                    )

                except Exception:

                    print(
                        "❌ send_audio() crashed"
                    )

                    traceback.print_exc()

                finally:

                    if (
                        silence_task
                        and not silence_task.done()
                    ):
                        silence_task.cancel()

            # ============================================================
            # RECEIVE TRANSCRIPTS
            # AssemblyAI → Backend → Angular
            # ============================================================

            async def receive_transcript():

                nonlocal silence_task
                nonlocal candidate_started

                try:

                    async for message in assembly_ws:

                        # ------------------------------------------------
                        # PARSE ASSEMBLYAI MESSAGE
                        # ------------------------------------------------

                        try:

                            data = json.loads(
                                message
                            )

                        except Exception:

                            print(
                                "⚠️ Invalid JSON received"
                            )

                            print(message)

                            continue

                        print(
                            "📩 AssemblyAI:",
                            data
                        )

                        # ------------------------------------------------
                        # ONLY PROCESS TURN MESSAGES
                        # ------------------------------------------------

                        if data.get("type") != "Turn":
                            continue

                        transcript = (
                            data.get(
                                "transcript",
                                ""
                            )
                            .strip()
                        )

                        end_of_turn = bool(
                            data.get(
                                "end_of_turn",
                                False
                            )
                        )

                        # ------------------------------------------------
                        # NO TRANSCRIPT
                        # ------------------------------------------------

                        if not transcript:

                            # Still notify Angular if
                            # AssemblyAI indicates end of turn.
                            if end_of_turn:

                                await websocket.send_text(
                                    json.dumps({
                                        "type": "transcript",
                                        "transcript": "",
                                        "end_of_turn": True,
                                        "final": True
                                    })
                                )

                                print(
                                    "🏁 Empty end_of_turn received"
                                )

                            continue

                        # ------------------------------------------------
                        # CANDIDATE HAS STARTED SPEAKING
                        # ------------------------------------------------

                        if not candidate_started:

                            candidate_started = True

                            print(
                                "🎤 Candidate started speaking"
                            )

                        # ------------------------------------------------
                        # LOG TRANSCRIPT
                        # ------------------------------------------------

                        if end_of_turn:

                            print(
                                "✅ FINAL STT:",
                                transcript
                            )

                        else:

                            print(
                                "📝 PARTIAL STT:",
                                transcript
                            )

                        # ------------------------------------------------
                        # SEND TO ANGULAR IMMEDIATELY
                        #
                        # IMPORTANT:
                        # Do NOT wait for end_of_turn here.
                        # Angular receives the latest partial transcript
                        # while candidate is still speaking.
                        # ------------------------------------------------

                        await websocket.send_text(
                            json.dumps({

                                "type": "transcript",

                                "transcript":
                                    transcript,

                                "end_of_turn":
                                    end_of_turn,

                                "final":
                                    end_of_turn

                            })
                        )

                        print(
                            "📤 Sending STT to Angular:",
                            transcript,
                            "| final:",
                            end_of_turn
                        )

                        # ------------------------------------------------
                        # END OF TURN
                        # ------------------------------------------------

                        if end_of_turn:

                            print(
                                "🏁 AssemblyAI end_of_turn"
                            )

                            # ------------------------------------------------
                            # IMPORTANT:
                            #
                            # Do NOT append anything here.
                            #
                            # Angular will use this message to replace
                            # the current partial transcript with the
                            # finalized transcript.
                            # ------------------------------------------------

                except websockets.ConnectionClosed as e:

                    print(
                        f"❌ AssemblyAI closed connection "
                        f"Code={e.code} "
                        f"Reason={e.reason}"
                    )

                except Exception:

                    print(
                        "❌ receive_transcript() crashed"
                    )

                    traceback.print_exc()

                finally:

                    if (
                        silence_task
                        and not silence_task.done()
                    ):

                        silence_task.cancel()

            # ============================================================
            # RUN AUDIO + TRANSCRIPT TASKS
            # ============================================================

            await asyncio.gather(
                send_audio(),
                receive_transcript()
            )

    except Exception:

        print(
            "❌ websocket_audio() crashed"
        )

        traceback.print_exc()

    finally:

        print(
            "🔴 websocket_audio() finished"
        )
        
@app.post("/api/candidate/login")
def candidate_login(payload: dict, db: Session = Depends(get_db)):
    email = payload.get("email")
    phone = payload.get("phone")

    # 🔒 Basic validation
    if not email or not phone:
        return {
            "success": False,
            "message": "Email and phone are required"
        }

    # 🔍 Check if candidate exists
    candidate = db.query(Candidate).filter(Candidate.email == email).first()

    # =========================
    # ✅ EXISTING USER LOGIN
    # =========================
    if candidate:
        if candidate.phone != phone:
            return {
                "success": False,
                "message": "Invalid phone number"
            }

        return {
            "success": True,
            "message": "Login successful",
            "data": {
                "id": candidate.id,
                "name": candidate.name,
                "email": candidate.email,
            }
        }

    # =========================
    # 🔥 NEW USER → AUTO CREATE
    # =========================
    import uuid

    new_candidate = Candidate(
        name=email.split("@")[0],
        email=email,
        phone=phone,
        interview_id=str(uuid.uuid4())
    )

    db.add(new_candidate)
    db.commit()
    db.refresh(new_candidate)

    return {
        "success": True,
        "message": "User created & login successful",
        "data": {
            "id": new_candidate.id,
            "name": new_candidate.name,
            "email": new_candidate.email,
            "interview_id": new_candidate.interview_id
        }
    }

# Use the correct production model strings
PRIMARY_MODEL = "gemini-3-flash-preview"  # ✅ Updated to latest flash model for best performance
# PRIMARY_MODEL = "gemini-2.5-flash-lite"  # ✅ Updated to latest flash model for best performance
# PRIMARY_MODEL = "gemini-2.5-flash"  # ✅ Updated to latest flash model for best performance

# ================= ENFORCED JSON SCHEMAS =================
class QuestionItem(BaseModel):
    question_text: str
    expected_answer: str
    difficulty: str = "Medium"

class QuestionGenerationSchema(BaseModel):
    detected_subject: str
    questions: List[QuestionItem]

class GradingSchema(BaseModel):
    score: int
    feedback: str

@app.post("/api/grade-answer/{answer_id}")
async def grade_answer_with_pdf(
    answer_id: int, 
    pdf_file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    try:
        # 1. Fetch the answer and question from DB
        answer_record = db.query(Answer).filter(Answer.id == answer_id).first()
        if not answer_record:
            raise HTTPException(status_code=404, detail="Answer record not found")
        
        question_record = db.query(Question).filter(Question.id == answer_record.question_id).first()

        # 2. Read PDF bytes
        pdf_content = await pdf_file.read()

        # 3. Construct the prompt
        prompt = f"""
        You are an expert interviewer. Grade the candidate's answer based ONLY on the provided PDF document.
        
        QUESTION: {question_record.question_text}
        EXPECTED KEY POINTS: {question_record.expected_answer}
        CANDIDATE'S ACTUAL ANSWER: {answer_record.answer_text}
        
        TASK:
        - Compare the candidate's answer with the technical facts in the PDF.
        - Assign a score from 0 to 10.
        - Provide a short, professional feedback sentence.
        
        Return ONLY a JSON object:
        {{
            "score": integer,
            "feedback": "string"
        }}
        """

        # 4. Call Gemini 2.0 Flash
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=pdf_content, mime_type="application/pdf"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        # 5. Parse Gemini response
        grading_result = json.loads(response.text)

        # 6. Update Database
        answer_record.ai_score = grading_result.get("score")
        answer_record.ai_response = grading_result.get("feedback")
        
        db.commit()
        db.refresh(answer_record)

        return {
            "success": True,
            "score": answer_record.ai_score,
            "feedback": answer_record.ai_response
        }

    except Exception as e:
        db.rollback()
        print(f"❌ Grading Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def grade_with_expected_answer(answer_id: int, question_id: int, candidate_answer: str):
    db = SessionLocal()
    try:
        question = db.query(Question).filter(Question.id == question_id).first()
        if not question:
            print(f"❓ Question ID {question_id} missing from lookup bank.")
            return

        expected = question.expected_answer or "No baseline verification reference value established."

        prompt = f"""
        You are a seasoned technical interviewer grading a candidate's answer response string.
        
        QUESTION CONTEXT: {question.question_text}
        EXPECTED KEY ANSWER VALUES: {expected}
        CANDIDATE ACTUAL RESPONSE STRING: {candidate_answer}

        Task Core Guidelines:
        - Deeply evaluate semantic concept alignment logic over exact textual sequence matches.
        - Output an integer score grading evaluation on a strict 0 to 10 scale.
        - Populate a clean feedback summary statement mapping observations fairly.
        """

        response = genai_client.models.generate_content(
            model=PRIMARY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GradingSchema
            )
        )

        result = json.loads(response.text)
        final_score = int(result.get("score", 0))
        final_feedback = result.get("feedback", "Grading evaluation cycle finalized.")

        db.query(Answer).filter(Answer.id == answer_id).update({
            "ai_score": final_score,
            "ai_response": final_feedback
        })
        db.commit()
        print(f"✅ AI Grading Complete for Answer Record ID: {answer_id} | Assigned Evaluation Score: {final_score}")

    except Exception as e:
        db.rollback()
        print(f"❌ Critical Core AI Background Thread Failure: {str(e)}")
        db.query(Answer).filter(Answer.id == answer_id).update({
            "ai_response": f"AI Evaluation Pipeline Drop: {str(e)[:45]}..."
        })
        db.commit()
    finally:
        db.close()


# --- THE UPDATED ROUTE ---
@app.post("/api/submit-and-grade")
async def submit_and_grade(
    background_tasks: BackgroundTasks,
    candidate_id: int = Form(...),
    question_id: int = Form(...),
    interview_id: str = Form(...),
    answer_text: str = Form(...),
    panel_id: Optional[int] = Form(None),
    db: Session = Depends(get_db)
):

    new_ans = Answer(
        candidate_id=candidate_id,
        question_id=question_id,
        interview_id=interview_id,
        answer_text=answer_text,
        panel_id=panel_id,
        ai_response="Processing..."
    )

    db.add(new_ans)
    db.commit()
    db.refresh(new_ans)

    # Trigger background evaluation
    background_tasks.add_task(
        grade_with_expected_answer,
        new_ans.id,
        question_id,
        answer_text
    )

    return {
        "success": True,
        "message": "Answer received!",
        "answer_id": new_ans.id
    }


# @app.post("/api/upload-and-generate-questions")
# async def upload_pdf_and_generate(
#     file: UploadFile = File(...), 
#     num_questions: int = Form(3),        # 👈 Changed to Form so it cleanly extracts from Angular FormData
#     course: CourseProgram = Form(...),   # 👈 Automatically validates against your 6 options!
#     db: Session = Depends(get_db) 
# ):
#     try:
#         # Read the file stream incoming from the upload pipeline
#         pdf_bytes = await file.read()
        
#         prompt = f"""
#         Analyze this PDF document cleanly.
#         1. Identify the high-level professional subject domain matching a {course.value} program curriculum.
#         2. Generate exactly {num_questions} structured interview questions anchored directly to the source text data.
#         3. For each item, provide a comprehensive 'expected_answer' context block.
#         4. Infer a structural difficulty tier ('Easy', 'Medium', or 'Hard') based on the complexity of the section topic parsed.
#         """

#         # Call Gemini using the structured JSON schema rules setup
#         response = genai_client.models.generate_content(
#             model=PRIMARY_MODEL,
#             contents=[
#                 types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
#                 prompt
#             ],
#             config=types.GenerateContentConfig(
#                 response_mime_type="application/json",
#                 response_schema=QuestionGenerationSchema  # Enforces structural text output consistency
#             )
#         )

#         # Parse output payload from model pipeline strings
#         data = json.loads(response.text)
#         subject = data.get("detected_subject", "General AI Concept").upper()
#         session_category = f"{subject}_{uuid.uuid4().hex[:4]}"

#         # Iterate over structural dictionary lists 
#         for q in data.get('questions', []):
#             new_question = Question(
#                 question_text=q['question_text'].strip(),
#                 expected_answer=q['expected_answer'].strip(),
#                 category=session_category,
#                 course=course,                   # 👈 Saves natively as your clean enum key
#                 difficulty=q.get('difficulty', 'Medium').strip(), # Captures AI inferred complexity level
#                 time_limit=120                   # Standard default limit (seconds) for AI questions
#             )
#             db.add(new_question)
        
#         # Commit transactional operations block safely to your database 
#         db.commit()

#         return {
#             "success": True, 
#             "category": session_category, 
#             "course": course.value,              # 👈 Returns clean string ("B-Tech", "MCA", etc.)
#             "display_subject": subject,
#             "count": len(data.get('questions', []))
#         }

#     except Exception as e:
#         db.rollback()                            # Flushes changes out of scope bounds upon drop anomalies
#         print(f"❌ ERROR inside generation sequence: {str(e)}")
#         return {"success": False, "error": str(e)}   

# @app.get("/api/get-questions-by-candidate/{candidate_id}")
# def get_questions_by_candidate(candidate_id: int, db: Session = Depends(get_db)):
#     # 1. Fetch candidate context profile matching the incoming primary key ID
#     candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
#     if not candidate:
#         raise HTTPException(status_code=404, detail="Candidate profile data not found")
    
#     # 2. Extract course variable (Check the SQLEnum column first, fallback to standard String field)
#     raw_course = candidate.course_program or candidate.student_course_program
#     if not raw_course:
#         raise HTTPException(status_code=400, detail="No course program assignment mapping found on this student profile")
    
#     # Extract string value if it's a structural Enum instance
#     course_str = raw_course.value if hasattr(raw_course, 'value') else str(raw_course)
    
#     # 3. Pull all matching sequential questions out of the question_bank table
#     questions = db.query(Question).filter(Question.course == course_str).all()
    
#     # Optional Fallback Rule: If no items exist under the explicit tag, pull "General" items
#     if not questions:
#         questions = db.query(Question).filter(Question.category.ilike("%General%")).all()
        
#     return {
#         "success": True,
#         "candidate_id": candidate.id,
#         "candidate_name": candidate.name,
#         "matched_course": course_str,
#         "total_questions": len(questions),
#         "questions": [
#             {
#                 "id": q.id,
#                 "question_text": q.question_text,
#                 "expected_answer": q.expected_answer,
#                 "category": q.category,
#                 "difficulty": q.difficulty,
#                 "time_limit": q.time_limit if q.time_limit else 50
#             }
#             for q in questions
#         ]
#     }

class StudentItem(BaseModel):
    name: str
    email: str
    phone: str
    department_branch: str

class BulkRegistrationPayload(BaseModel):
    course_program: str
    students: List[StudentItem]

@app.post("/api/save-bulk-students")
def save_bulk_students(payload: BulkRegistrationPayload, db: Session = Depends(get_db)):
    try:
        inserted_count = 0
        
        # Loop through the parsed array rows extracted by your Angular app
        for student in payload.students:
            # Prepare an individual table record object mapping properties
            db_candidate = Candidate(
                name=student.name,
                email=student.email,
                phone=student.phone,
                course_program=payload.course_program,       # Applied globally from your select dropdown
                department_branch=student.department_branch  # Extracted row-by-row from Excel
            )
            db.add(db_candidate)
            inserted_count += 1
            
        # Commit the transaction to save all records cleanly
        db.commit()
        
        print(f"🎉 Successfully persisted {inserted_count} student records inside 'candidates' table.")
        return {"success": True, "inserted_count": inserted_count}

    except Exception as e:
        db.rollback()  # Rollback if an unexpected breakdown happens to prevent corrupted tables
        print(f"❌ Critical Database Write Anomaly: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database operational write failure: {str(e)}")
    
# 💡 FIX: Make sure it's attached directly to @app if written inside main.py
@app.get("/api/candidate-courses/{email}")
async def get_candidate_courses(
    email: str,
    db: Session = Depends(get_db)
):
    """
    Returns all courses that the candidate (identified by email)
    is registered for.
    """

    registrations = (
        db.query(Candidate, CourseMaster)
        .join(
            CourseMaster,
            Candidate.course_id == CourseMaster.course_id
        )
        .filter(
            Candidate.email == email
        )
        .all()
    )

    if not registrations:

        raise HTTPException(
            status_code=404,
            detail=f"No registration history found for email address: {email}"
        )

    allowed_courses = []

    seen_course_ids = set()

    for candidate, course in registrations:

        if course.course_id in seen_course_ids:
            continue

        seen_course_ids.add(course.course_id)

        allowed_courses.append({

            "course_id": course.course_id,

            "course_name": course.course_name,

            "branch_name": course.branch_name,

            "course_code": course.course_code

        })

    return {

        "success": True,

        "candidate_name": registrations[0][0].name,

        "allowed_courses": allowed_courses

    }

@app.get("/api/get-questions-by-candidate/{candidate_id}")
async def get_self_assessment_questions(
    candidate_id: int,
    course_id: int,
    db: Session = Depends(get_db)
):
    """
    1. Resolve the candidate.
    2. Verify the candidate's email is registered for the selected course_id.
    3. Fetch 5 random questions for that course_id.
    """

    # -----------------------------------------------------------------
    # STEP 1: GET CURRENT CANDIDATE
    # -----------------------------------------------------------------

    current_candidate = (
        db.query(Candidate)
        .filter(
            Candidate.id == candidate_id
        )
        .first()
    )

    if not current_candidate:
        raise HTTPException(
            status_code=404,
            detail="Candidate profile not found."
        )

    # -----------------------------------------------------------------
    # STEP 2: VERIFY EMAIL IS REGISTERED FOR THIS COURSE
    # -----------------------------------------------------------------

    registration_check = (
        db.query(Candidate)
        .filter(
            Candidate.email == current_candidate.email,
            Candidate.course_id == course_id
        )
        .first()
    )

    if not registration_check:

        raise HTTPException(
            status_code=403,
            detail="Candidate is not registered for the selected course."
        )

    # -----------------------------------------------------------------
    # STEP 3: FETCH RANDOM QUESTIONS
    # -----------------------------------------------------------------

    questions = (
        db.query(Question)
        .filter(
            Question.course_id == course_id
        )
        .order_by(func.random())
        .limit(5)
        .all()
    )

    if not questions:

        raise HTTPException(
            status_code=404,
            detail="No questions found for the selected course."
        )

    # -----------------------------------------------------------------
    # STEP 4: FETCH COURSE DETAILS
    # -----------------------------------------------------------------

    course = (
        db.query(CourseMaster)
        .filter(
            CourseMaster.course_id == course_id
        )
        .first()
    )

    # -----------------------------------------------------------------
    # STEP 5: RESPONSE
    # -----------------------------------------------------------------

    return {

        "success": True,

        "candidate_id": registration_check.id,

        "candidate_name": registration_check.name,

        "course_id": course_id,

        "course_name": course.course_name if course else None,

        "branch_name": course.branch_name if course else None,

        "total_questions": len(questions),

        "questions": [

            {

                "id": q.id,

                "question_text": q.question_text,

                "time_limit": q.time_limit,

                "difficulty": q.difficulty,

                "expected_answer": q.expected_answer,

                "category": q.category

            }

            for q in questions

        ]

    }


# @app.post("/api/self-assessment/submit-answer")
# async def submit_self_assessment_answer(
#     background_tasks: BackgroundTasks,
#     candidate_id: int = Form(...),
#     assessment_id: str = Form(...),
#     question_id: int = Form(...),
#     answer_text: str = Form(...),
#     course: str = Form(...),
#     db: Session = Depends(get_db)
# ):

#     new_answer = SelfAssessmentAnswer(
#         candidate_id=candidate_id,
#         assessment_id=assessment_id,
#         question_id=question_id,
#         answer_text=answer_text,
#         course=course,
#         ai_response="Processing...",
#         status="Processing"
#     )

#     db.add(new_answer)
#     db.commit()
#     db.refresh(new_answer)

#     # Trigger AI grading in background
#     background_tasks.add_task(
#         grade_self_assessment_answer,
#         new_answer.id,
#         question_id,
#         answer_text
#     )

#     return {
#         "success": True,
#         "message": "Answer received successfully.",
#         "answer_id": new_answer.id,
#         "assessment_id": assessment_id
#     }

@app.post("/api/self-assessment/submit-answer")
async def submit_self_assessment_answer(
    background_tasks: BackgroundTasks,
    candidate_id: int = Form(...),
    assessment_id: str = Form(...),
    question_text: str = Form(...),
    expected_answer: str = Form(...),
    answer_text: str = Form(...),
    course: str = Form(...),
    db: Session = Depends(get_db)
):

    try:

        # --------------------------------------------------------
        # SAVE CANDIDATE ANSWER
        # --------------------------------------------------------

        new_answer = SelfAssessmentAnswer(
            candidate_id=candidate_id,
            assessment_id=assessment_id,
            question_text=question_text,
            expected_answer=expected_answer,
            answer_text=answer_text,
            course=course,
            ai_response="Processing...",
            status="Processing"
        )

        db.add(new_answer)
        db.commit()
        db.refresh(new_answer)

        print(
            f"📝 Self Assessment Answer Saved | "
            f"Answer ID: {new_answer.id}"
        )

        # --------------------------------------------------------
        # START AI GRADING
        # --------------------------------------------------------

        background_tasks.add_task(
            grade_self_assessment_answer,
            new_answer.id,
            question_text,
            expected_answer,
            answer_text
        )

        print(
            f"🤖 AI Grading Task Started | "
            f"Answer ID: {new_answer.id}"
        )

        return {
            "success": True,
            "message": "Answer received successfully.",
            "answer_id": new_answer.id,
            "assessment_id": assessment_id
        }

    except Exception as e:

        db.rollback()

        print(
            f"❌ Failed to save self-assessment answer: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save answer: {str(e)}"
        )


async def grade_self_assessment_answer(
    answer_id: int,
    question_text: str,
    expected_answer: str,
    candidate_answer: str
):
    """
    Grade a self-assessment answer using Gemini.

    Tries multiple Gemini models in order.
    If one model fails, the next model is attempted.
    """

    db = SessionLocal()

    # ------------------------------------------------------------
    # GEMINI MODEL FALLBACK LIST
    # ------------------------------------------------------------

    models_to_try = [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    try:

        # --------------------------------------------------------
        # VALIDATE INPUT
        # --------------------------------------------------------

        if not candidate_answer or not candidate_answer.strip():

            db.query(SelfAssessmentAnswer).filter(
                SelfAssessmentAnswer.id == answer_id
            ).update({
                "ai_score": 0,
                "ai_response": "Candidate did not provide an answer.",
                "status": "Completed"
            })

            db.commit()

            print(
                f"⚠️ Empty answer | Answer ID: {answer_id}"
            )

            return

        # --------------------------------------------------------
        # BUILD PROMPT
        # --------------------------------------------------------

        prompt = f"""
You are a seasoned technical interviewer evaluating a candidate's
answer to a technical interview question.

QUESTION:
{question_text}

EXPECTED ANSWER:
{expected_answer}

CANDIDATE ANSWER:
{candidate_answer}

Evaluate the candidate answer against the expected answer.

IMPORTANT RULES:
- Evaluate conceptual understanding.
- Do not require exact wording.
- Equivalent explanations should receive credit.
- Do not penalize the candidate for using different terminology
  if the underlying concept is correct.
- Identify important missing concepts.
- Give a fair score from 0 to 10.
- Give concise but useful feedback.
- Do not invent information that is not present in the candidate answer.

SCORING GUIDELINES:

0-2:
The answer is completely incorrect, irrelevant, or demonstrates
almost no understanding.

3-4:
The answer demonstrates limited understanding but contains
significant mistakes or missing concepts.

5-6:
The answer demonstrates partial understanding and is mostly
correct but misses some important concepts.

7-8:
The answer demonstrates good understanding and is substantially
correct with only minor omissions or inaccuracies.

9:
The answer demonstrates very strong understanding with almost
everything important covered.

10:
The answer is completely correct, clear, and demonstrates
excellent conceptual understanding.

Return ONLY valid JSON in this format:

{{
    "score": 8,
    "feedback": "The candidate demonstrates good understanding..."
}}
"""

        # --------------------------------------------------------
        # TRY GEMINI MODELS ONE BY ONE
        # --------------------------------------------------------

        last_error = None

        for model_name in models_to_try:

            try:

                print(
                    f"🤖 Trying Gemini model: {model_name} | "
                    f"Answer ID: {answer_id}"
                )

                response = genai_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GradingSchema
                    )
                )

                # ------------------------------------------------
                # VALIDATE RESPONSE
                # ------------------------------------------------

                if not response or not response.text:

                    raise Exception(
                        f"{model_name} returned an empty response."
                    )

                print(
                    f"✅ Gemini response received from {model_name}"
                )

                result = json.loads(response.text)

                final_score = int(
                    result.get("score", 0)
                )

                final_feedback = result.get(
                    "feedback",
                    "Evaluation completed."
                )

                # ------------------------------------------------
                # SAFETY CHECK SCORE
                # ------------------------------------------------

                if final_score < 0:
                    final_score = 0

                if final_score > 10:
                    final_score = 10

                # ------------------------------------------------
                # SAVE RESULT
                # ------------------------------------------------

                db.query(SelfAssessmentAnswer).filter(
                    SelfAssessmentAnswer.id == answer_id
                ).update({

                    "ai_score": final_score,

                    "ai_response": final_feedback,

                    "status": "Completed"

                })

                db.commit()

                print(
                    f"✅ Self Assessment Graded Successfully | "
                    f"Answer ID: {answer_id} | "
                    f"Model: {model_name} | "
                    f"Score: {final_score}"
                )

                # ------------------------------------------------
                # IMPORTANT:
                # Stop trying other models because this worked.
                # ------------------------------------------------

                return

            except Exception as model_error:

                last_error = model_error

                print(
                    f"❌ Gemini model failed: {model_name}"
                )

                print(
                    f"   Error: {str(model_error)}"
                )

                # ------------------------------------------------
                # Try next model
                # ------------------------------------------------

                continue

        # --------------------------------------------------------
        # ALL MODELS FAILED
        # --------------------------------------------------------

        error_message = str(last_error)

        print(
            f"❌ ALL GEMINI MODELS FAILED | "
            f"Answer ID: {answer_id}"
        )

        db.query(SelfAssessmentAnswer).filter(
            SelfAssessmentAnswer.id == answer_id
        ).update({

            "ai_score": 0,

            "ai_response":
                "AI evaluation could not be completed. "
                f"All Gemini models failed. "
                f"Last error: {error_message[:200]}",

            "status": "Failed"

        })

        db.commit()

    except Exception as e:

        db.rollback()

        error_message = str(e)

        print(
            f"❌ Self Assessment Grading Error | "
            f"Answer ID: {answer_id}"
        )

        print(
            f"Error: {error_message}"
        )

        try:

            db.query(SelfAssessmentAnswer).filter(
                SelfAssessmentAnswer.id == answer_id
            ).update({

                "ai_score": 0,

                "ai_response":
                    f"AI Evaluation Error: "
                    f"{error_message[:200]}",

                "status": "Failed"

            })

            db.commit()

        except Exception as db_error:

            print(
                f"❌ Failed to update grading error in database: "
                f"{str(db_error)}"
            )

    finally:

        db.close()


@app.post("/api/self-assessment/generate-final-result/{assessment_id}")
async def generate_final_self_assessment_result(
    assessment_id: str,
    db: Session = Depends(get_db)
):

    answers = db.query(SelfAssessmentAnswer)\
        .filter(
            SelfAssessmentAnswer.assessment_id == assessment_id
        )\
        .all()

    if not answers:
        raise HTTPException(
            status_code=404,
            detail="No answers found for this assessment."
        )

    candidate_id = answers[0].candidate_id

    total_marks = sum(
        (a.ai_score or 0)
        for a in answers
    )

    total_questions = len(answers)

    # ⭐ NEW
    maximum_marks = total_questions * 10

    percentage = round(
        (total_marks / maximum_marks) * 100,
        2
    ) if maximum_marks > 0 else 0

    feedback_text = "\n".join([
        a.ai_response or ""
        for a in answers
    ])

    course = answers[0].course

    try:

        prompt = f"""
        You are an expert technical interviewer.

        Assessment Statistics:

        Total Questions: {total_questions}

        Total Marks: {total_marks} / {maximum_marks}

        Percentage: {percentage}%

        Individual Feedback:

        {feedback_text}

        Return ONLY JSON.

        Required JSON Structure:

        {{
            "strengths": [
                "strength 1",
                "strength 2"
            ],
            "improvements": [
                "improvement 1",
                "improvement 2"
            ],
            "summary": "overall assessment summary"
        }}
        """

        response = genai_client.models.generate_content(
            model=PRIMARY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FinalAssessmentSchema
            )
        )

        final_response = response.text

    except Exception as e:

        print(
            f"⚠️ Failed generating overall feedback: {str(e)}"
        )

        final_response = (
            "Assessment completed successfully. "
            "Overall AI summary unavailable."
        )

    existing = db.query(SelfAssessmentResult)\
        .filter(
            SelfAssessmentResult.assessment_id == assessment_id
        )\
        .first()

    if existing:

        existing.final_marks = total_marks
        existing.final_response = final_response

        db.commit()

        return {
            "success": True,
            "message": "Result updated.",
            "result_id": existing.id,
            "final_marks": total_marks,
            "maximum_marks": maximum_marks,
            "percentage": percentage
        }

    result = SelfAssessmentResult(
        candidate_id=candidate_id,
        assessment_id=assessment_id,
        course=course,
        final_marks=total_marks,
        final_response=final_response
    )

    db.add(result)
    db.commit()
    db.refresh(result)

    return {
        "success": True,
        "message": "Final result generated.",
        "result_id": result.id,
        "final_marks": total_marks,
        "maximum_marks": maximum_marks,
        "percentage": percentage
    }

# @app.post("/api/self-assessment/generate-final-result/{assessment_id}")
# async def generate_final_self_assessment_result(
#     assessment_id: str,
#     db: Session = Depends(get_db)
# ):

#     answers = db.query(SelfAssessmentAnswer)\
#         .filter(
#             SelfAssessmentAnswer.assessment_id == assessment_id
#         )\
#         .all()

#     if not answers:
#         raise HTTPException(
#             status_code=404,
#             detail="No answers found for this assessment."
#         )

#     candidate_id = answers[0].candidate_id

#     total_marks = sum(
#         (a.ai_score or 0)
#         for a in answers
#     )

#     total_questions = len(answers)

#     average_score = (
#         total_marks / total_questions
#         if total_questions > 0
#         else 0
#     )

#     course = "Unknown"

#     strengths = []
#     improvements = []

#     if average_score >= 8:

#         strengths = [
#             "Strong understanding of core concepts",
#             "Consistent performance across questions",
#             "Demonstrated good technical knowledge"
#         ]

#     elif average_score >= 5:

#         strengths = [
#             "Basic conceptual understanding",
#             "Able to answer most questions"
#         ]

#         improvements = [
#             "Needs deeper technical knowledge",
#             "Should improve explanation quality"
#         ]

#     else:

#         improvements = [
#             "Requires additional preparation",
#             "Needs improvement in fundamentals",
#             "Should practice more technical concepts"
#         ]

#     final_response = json.dumps({
#         "strengths": strengths,
#         "improvements": improvements,
#         "summary":
#             f"Candidate scored {total_marks} marks across "
#             f"{total_questions} questions with an average score "
#             f"of {round(average_score, 2)}."
#     })

#     existing = db.query(SelfAssessmentResult)\
#         .filter(
#             SelfAssessmentResult.assessment_id == assessment_id
#         )\
#         .first()

#     if existing:

#         existing.final_marks = total_marks
#         existing.final_response = final_response

#         db.commit()

#         return {
#             "success": True,
#             "message": "Result updated.",
#             "result_id": existing.id,
#             "final_marks": total_marks
#         }

#     result = SelfAssessmentResult(
#         candidate_id=candidate_id,
#         assessment_id=assessment_id,
#         course=course,
#         final_marks=total_marks,
#         final_response=final_response
#     )

#     db.add(result)
#     db.commit()
#     db.refresh(result)

#     return {
#         "success": True,
#         "message": "Final result generated.",
#         "result_id": result.id,
#         "final_marks": total_marks
#     }

@app.get("/api/self-assessment/result/{assessment_id}")
async def get_self_assessment_result(
    assessment_id: str,
    db: Session = Depends(get_db)
):

    result = db.query(SelfAssessmentResult)\
        .filter(
            SelfAssessmentResult.assessment_id == assessment_id
        )\
        .first()

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Assessment result not found."
        )

    # ---------------------------------------------
    # Total Questions Assigned
    # ---------------------------------------------
    total_questions = db.query(SelfAssessmentAnswer)\
        .filter(
            SelfAssessmentAnswer.assessment_id == assessment_id
        )\
        .count()

    # ---------------------------------------------
    # Questions Actually Attempted
    # ---------------------------------------------
    attempted_questions = db.query(SelfAssessmentAnswer)\
        .filter(
            SelfAssessmentAnswer.assessment_id == assessment_id,
            SelfAssessmentAnswer.answer_text.isnot(None),
            SelfAssessmentAnswer.answer_text != "",
            ~SelfAssessmentAnswer.answer_text.contains(
                "Candidate did not answer this question"
            )
        )\
        .count()

    # ---------------------------------------------
    # Marks & Percentages
    # ---------------------------------------------
    maximum_marks = total_questions * 10

    percentage = round(
        (result.final_marks / maximum_marks) * 100,
        2
    ) if maximum_marks > 0 else 0

    attempt_percentage = round(
        (attempted_questions / total_questions) * 100,
        2
    ) if total_questions > 0 else 0

    # ---------------------------------------------
    # Ensure final_response is always valid JSON
    # ---------------------------------------------
    final_response = result.final_response

    try:
        json.loads(final_response)
    except Exception:

        average_score = round(
            result.final_marks / total_questions,
            2
        ) if total_questions > 0 else 0

        final_response = json.dumps({
            "strengths": [],
            "improvements": [],
            "summary":
                f"Candidate scored {result.final_marks} marks out of "
                f"{maximum_marks} across {total_questions} questions "
                f"with an average score of {average_score} per question."
        })

    return {
        "success": True,
        "result": {
            "candidate_id": result.candidate_id,
            "assessment_id": result.assessment_id,
            "course": result.course,

            "final_marks": result.final_marks,
            "maximum_marks": maximum_marks,
            "percentage": percentage,

            # Questionnaire Statistics
            "assigned_questions": total_questions,
            "attempted_questions": attempted_questions,
            "attempt_percentage": attempt_percentage,

            "final_response": final_response,
            "completed_at": result.completed_at
        }
    }

class SelfAssessmentRequest(BaseModel):
    course_id: int

@app.post("/api/self-assessment/generate-questions")
def generate_self_assessment_questions(
    payload: SelfAssessmentRequest,
    db: Session = Depends(get_db)
):

    # ----------------------------------------------------
    # Fetch Course Details
    # ----------------------------------------------------
    course = (
        db.query(CourseMaster)
        .filter(CourseMaster.course_id == payload.course_id)
        .first()
    )

    if not course:
        raise HTTPException(
            status_code=404,
            detail="Course not found."
        )

    course_name = course.course_name
    branch_name = course.branch_name or ""

    print(f"📘 Course : {course_name}")
    print(f"🌿 Branch : {branch_name}")

    # ----------------------------------------------------
    # Prompt
    # ----------------------------------------------------
    prompt = f"""
You are an experienced technical interviewer.

Generate EXACTLY 10 interview questions for a self-assessment technical interview.

Course:
{course_name}

Branch:
{branch_name}

Rules:

- Questions must be suitable for a spoken AI interview.
- Mix beginner, intermediate, and advanced difficulty levels.
- Questions should be answerable within approximately 50 seconds.
- Include both conceptual and practical interview questions.
- Avoid coding/program-writing questions.
- Avoid duplicate or very similar questions.
- Ensure the questions are relevant to both the selected course and branch.
- For every question, generate an expected answer that represents what an ideal candidate should answer.
- The expected answer should be technically accurate, concise, and approximately 80-150 words.
- The expected answer should cover the important concepts an interviewer would expect.
- Do not include numbering, headings, explanations, or markdown.
- Return ONLY valid JSON.

Return format:

[
    {{
        "question": "Question 1",
        "expected_answer": "Expected answer for Question 1."
    }},
    {{
        "question": "Question 2",
        "expected_answer": "Expected answer for Question 2."
    }},
    {{
        "question": "Question 3",
        "expected_answer": "Expected answer for Question 3."
    }}
]
"""

    try:

        response = genai_client.models.generate_content(
            model=PRIMARY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        text = response.text.strip()

        # Gemini sometimes wraps JSON in markdown
        text = re.sub(r"```json", "", text)
        text = re.sub(r"```", "", text).strip()

        questions = json.loads(text)

        if not isinstance(questions, list):
            raise Exception("Gemini returned an invalid JSON array.")

        return {
            "success": True,
            "course_id": course.course_id,
            "course_name": course.course_name,
            "branch_name": course.branch_name,
            "total_questions": len(questions),
            "questions": questions
        }

    except Exception as e:

        print(f"❌ Gemini Question Generation Failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate interview questions. {str(e)}"
        )
  

# @app.get("/api/interviews/check-schedule/{email}")
# async def check_interview_schedule(
#     email: str,
#     db: Session = Depends(get_db)
# ):

#     # Find all candidate records for this email
#     candidate_ids = [
#         c.id
#         for c in db.query(Candidate)
#         .filter(Candidate.email == email)
#         .all()
#     ]

#     if not candidate_ids:
#         return {
#             "success": True,
#             "status": "none",
#             "message": "Candidate not found."
#         }

#     # Find latest interview among all candidate records
#     interview = (
#         db.query(Interview)
#         .filter(
#             Interview.candidate_id.in_(candidate_ids)
#         )
#         .order_by(Interview.id.desc())
#         .first()
#     )

#     if not interview:
#         return {
#             "success": True,
#             "status": "none",
#             "message": "No interview assigned."
#         }

#     # ----------------------------------
#     # COMPLETED
#     # ----------------------------------

#     if interview.status == "Completed":
#         return {
#             "success": True,
#             "status": "completed",
#             "message": "Interview already completed.",
#             "interview": {
#                 "candidate_id": interview.candidate_id,
#                 "interview_id": interview.interview_id,
#                 "panel_id": interview.panel_id,
#                 "course": interview.interview_category,
#                 "scheduled_at": interview.scheduled_at
#             }
#         }

#     # ----------------------------------
#     # CANCELLED
#     # ----------------------------------

#     if interview.status == "Cancelled":
#         return {
#             "success": True,
#             "status": "cancelled",
#             "message": "Interview was cancelled.",
#             "interview": {
#                 "candidate_id": interview.candidate_id,
#                 "interview_id": interview.interview_id,
#                 "panel_id": interview.panel_id,
#                 "course": interview.interview_category,
#                 "scheduled_at": interview.scheduled_at
#             }
#         }

#     # ----------------------------------
#     # DATE PARSE
#     # ----------------------------------

#     try:

#         scheduled_dt = datetime.strptime(
#             interview.scheduled_at,
#             "%Y-%m-%d %H:%M"
#         )

#     except Exception:

#         print(
#             "Scheduled Date Parse Error:",
#             interview.scheduled_at
#         )

#         raise HTTPException(
#             status_code=500,
#             detail="Invalid scheduled_at format."
#         )

#     now = datetime.now()

#     # ----------------------------------
#     # NOT TODAY
#     # ----------------------------------

#     if scheduled_dt.date() != now.date():

#         return {
#             "success": True,
#             "status": "none",
#             "message": "No interview scheduled today."
#         }

#     diff_minutes = (
#         scheduled_dt - now
#     ).total_seconds() / 60

#     # ----------------------------------
#     # UPCOMING
#     # ----------------------------------

#     if diff_minutes > 15:

#         return {
#             "success": True,
#             "status": "upcoming",
#             "message": "Interview scheduled later today.",
#             "interview": {
#                 "candidate_id": interview.candidate_id,
#                 "interview_id": interview.interview_id,
#                 "panel_id": interview.panel_id,
#                 "course": interview.interview_category,
#                 "scheduled_at": interview.scheduled_at
#             }
#         }

#     # ----------------------------------
#     # READY
#     # ----------------------------------

#     if 0 <= diff_minutes <= 15:

#         return {
#             "success": True,
#             "status": "ready",
#             "message": "Interview ready to join.",
#             "interview": {
#                 "candidate_id": interview.candidate_id,
#                 "interview_id": interview.interview_id,
#                 "panel_id": interview.panel_id,
#                 "course": interview.interview_category,
#                 "scheduled_at": interview.scheduled_at
#             }
#         }

#     # ----------------------------------
#     # EXPIRED
#     # ----------------------------------

#     return {
#         "success": True,
#         "status": "expired",
#         "message": "Interview time has passed.",
#         "interview": {
#             "candidate_id": interview.candidate_id,
#             "interview_id": interview.interview_id,
#             "panel_id": interview.panel_id,
#             "course": interview.interview_category,
#             "scheduled_at": interview.scheduled_at
#         }
#     }

@app.get("/api/interviews/check-schedule/{email}")
async def check_interview_schedule(
    email: str,
    db: Session = Depends(get_db)
):

    # ---------------------------------------------------
    # Fetch all candidate records for the email
    # ---------------------------------------------------

    candidates = (
        db.query(Candidate)
        .filter(Candidate.email == email)
        .all()
    )

    if not candidates:
        return {
            "success": True,
            "status": "none",
            "message": "Candidate not found."
        }

    candidate_ids = [c.id for c in candidates]

    # ---------------------------------------------------
    # Fetch all panel mappings
    # ---------------------------------------------------

    panel_candidates = (
        db.query(PanelCandidate)
        .filter(
            PanelCandidate.candidate_id.in_(candidate_ids)
        )
        .all()
    )

    if not panel_candidates:
        return {
            "success": True,
            "status": "none",
            "message": "No interviews assigned."
        }

    interview_ids = list({
        pc.interview_id
        for pc in panel_candidates
    })

    # ---------------------------------------------------
    # Fetch interviews
    # ---------------------------------------------------

    interviews = (
        db.query(Interview)
        .filter(
            Interview.interview_id.in_(interview_ids)
        )
        .all()
    )

    if not interviews:
        return {
            "success": True,
            "status": "none",
            "message": "No interviews found."
        }

    now = datetime.now()

    available_interviews = []

    # ---------------------------------------------------
    # Filter today's active interviews
    # ---------------------------------------------------

    for interview in interviews:

        if interview.status in ["Completed", "Cancelled"]:
            continue

        try:

            start_dt = datetime.strptime(
                interview.scheduled_at,
                "%Y-%m-%d %H:%M"
            )

            end_dt = datetime.strptime(
                interview.scheduled_end_at,
                "%Y-%m-%d %H:%M"
            )

        except Exception:

            print(
                f"Invalid schedule format for interview {interview.interview_id}"
            )

            continue

        # Only today's interviews
        if start_dt.date() != now.date():
            continue

        # Interview already ended
        if now > end_dt:
            continue

        available_interviews.append({
            "interview_id": interview.interview_id,
            "panel_id": interview.panel_id,
            "course_id": interview.course_id,
            "interview_type": interview.interview_type,
            "interview_name": interview.interview_name,
            "scheduled_at": interview.scheduled_at,
            "scheduled_end_at": interview.scheduled_end_at,
            "status": interview.status
        })

    # ---------------------------------------------------
    # Response
    # ---------------------------------------------------

    if not available_interviews:
        return {
            "success": True,
            "status": "none",
            "message": "No active interviews available today."
        }

    available_interviews.sort(
        key=lambda x: x["scheduled_at"]
    )

    return {
        "success": True,
        "status": "available",
        "message": "Active interviews found.",
        "interviews": available_interviews
    }


# @app.get("/api/interviews/load-questions/{candidate_id}/{course}")
# async def load_scheduled_interview_questions(
#     candidate_id: int,
#     course: str,
#     db: Session = Depends(get_db)
# ):

#     # Verify interview exists for this candidate/course
#     interview = (
#         db.query(Interview)
#         .filter(
#             Interview.candidate_id == candidate_id,
#             Interview.interview_category == course
#         )
#         .order_by(Interview.id.desc())
#         .first()
#     )

#     if not interview:
#         raise HTTPException(
#             status_code=404,
#             detail="No scheduled interview found."
#         )

#     questions = (
#         db.query(Question)
#         .filter(
#             Question.course == course
#         )
#         .order_by(Question.id.asc())
#         .all()
#     )

#     if not questions:
#         raise HTTPException(
#             status_code=404,
#             detail=f"No questions found for course {course}"
#         )

#     return {
#         "success": True,
#         "candidate_id": candidate_id,
#         "interview_id": interview.interview_id,
#         "course": course,
#         "total_questions": len(questions),
#         "questions": [
#             {
#                 "id": q.id,
#                 "question_text": q.question_text,
#                 "difficulty": q.difficulty,
#                 "time_limit": q.time_limit
#             }
#             for q in questions
#         ]
#     }

@app.post("/api/interviews/save-video/{interview_code}")
async def save_video(
    interview_code: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:

        # --------------------------------------------------
        # Find Interview using Interview Code
        # --------------------------------------------------

        interview = (
            db.query(Interview)
            .filter(
                Interview.interview_id == interview_code
            )
            .first()
        )

        if not interview:
            raise HTTPException(
                status_code=404,
                detail="Interview not found."
            )

        # --------------------------------------------------
        # Create Recording Folder (if not exists)
        # --------------------------------------------------

        os.makedirs(RECORDING_BASE_PATH, exist_ok=True)

        # --------------------------------------------------
        # File Name
        # Example:
        # IVW-2026-6686.webm
        # --------------------------------------------------

        filename = f"{interview_code}.webm"

        full_path = os.path.join(
            RECORDING_BASE_PATH,
            filename
        )

        full_path = os.path.normpath(full_path)

        # --------------------------------------------------
        # Save Video
        # --------------------------------------------------

        contents = await file.read()

        with open(full_path, "wb") as f:
            f.write(contents)

        # --------------------------------------------------
        # Update Interview
        # --------------------------------------------------

        interview.video_path = filename
        interview.status = "Completed"

        db.commit()

        # --------------------------------------------------
        # Response
        # --------------------------------------------------

        # video_url = f"http://127.0.0.1:8000/videos/{filename}" monika
        video_url = f"{BASE_URL}/videos/{filename}"

        return {
            "success": True,
            "message": "Interview completed successfully.",
            "interview_id": interview.interview_id,
            "candidate_id": interview.candidate_id,
            "status": interview.status,
            "video_filename": filename,
            "video_path": filename,
            "video_url": video_url
        }

    except Exception as e:

        db.rollback()

        print("❌ SAVE VIDEO ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save interview recording: {str(e)}"
        )
    
@app.get("/api/candidates/courses")
async def get_available_courses(db: Session = Depends(get_db)):
    courses = (
        db.query(Candidate.course_program)
        .distinct()
        .order_by(Candidate.course_program.asc())
        .all()
    )

    return {
        "success": True,
        "courses": [c[0] for c in courses if c[0]]
    }

@app.get("/api/candidates/course/{course}")
async def get_candidates_by_course(
    course: str,
    db: Session = Depends(get_db)
):

    candidates = (
        db.query(Candidate)
        .filter(
            Candidate.course_program == course
        )
        .order_by(Candidate.name.asc())
        .all()
    )

    return {
        "success": True,
        "course": course,
        "total_candidates": len(candidates),
        "candidates": [
            {
                "id": c.id,
                "name": c.name,
                "email": c.email,
                "phone": c.phone,
                "course_program": c.course_program,
                "department_branch": c.department_branch,
                "video_path": c.video_path
            }
            for c in candidates
        ]
    }

@app.post("/api/interviews/schedule")
async def schedule_interview(
    request: ScheduleInterviewRequest,
    db: Session = Depends(get_db)
):
    try:

        panel_id = request.panel_id

        # ---------------------------------------
        # Create New Panel (if required)
        # ---------------------------------------

        if panel_id is None:

            panel = Panel(
                panel_name=request.panel_name,
                created_by=1
            )

            db.add(panel)
            db.commit()
            db.refresh(panel)

            for member_id in request.member_user_ids:

                role = (
                    "chairman"
                    if member_id == request.chairman_user_id
                    else "member"
                )

                db.add(
                    PanelMember(
                        panel_id=panel.id,
                        user_id=member_id,
                        role=role
                    )
                )

            db.commit()

            panel_id = panel.id

        # ---------------------------------------
        # Validate Candidates
        # ---------------------------------------

        candidates = (
            db.query(Candidate)
            .filter(
                Candidate.id.in_(request.candidate_ids)
            )
            .all()
        )

        if len(candidates) != len(request.candidate_ids):

            raise HTTPException(
                status_code=404,
                detail="One or more candidates not found."
            )

        # ---------------------------------------
        # Interview Type
        # ---------------------------------------

        interview_type = (
            "interview"
            if request.subject_id is None
            else str(request.subject_id)
        )

        scheduled_at = (
            f"{request.interview_date} "
            f"{request.start_time}"
        )

        scheduled_end_at = (
            f"{request.interview_date} "
            f"{request.end_time}"
        )

        # ---------------------------------------
        # Create Interview
        # ---------------------------------------

        interview = Interview(

            panel_id=panel_id,

            course_id=request.course_id,

            scheduled_at=scheduled_at,

            scheduled_end_at=scheduled_end_at,

            status="Scheduled",

            created_by=1,

            interview_type=interview_type,

            interview_id=request.interview_id,

            interview_name=request.interview_name,

        )

        db.add(interview)
        db.commit()
        db.refresh(interview)

        # ---------------------------------------
        # Create Panel Candidate Mapping
        # ---------------------------------------

        for candidate_id in request.candidate_ids:

            db.add(

                PanelCandidate(

                    panel_id=panel_id,

                    candidate_id=candidate_id,

                    interview_id=request.interview_id

                )

            )

        db.commit()

        return {

            "success": True,

            "message": "Interview scheduled successfully.",

            "interview_id": interview.interview_id,

            "interview_db_id": interview.id,

            "panel_id": panel_id

        }

    except Exception as ex:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )
    

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):

    original_name = file.filename or f"video_{uuid.uuid4()}.mp4"
    file_path = os.path.join(RECORDING_BASE_PATH, original_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        os.remove(file_path)
        return {"error": "Could not open video file"}

    emotion_counts = {
        "happy": 0, "neutral": 0, "sad": 0,
        "angry": 0, "fear": 0, "disgust": 0, "surprise": 0
    }
    frame_count = 0
    total_analyzed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:
            try:
                result = deepface.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False
                )
                emotion = result[0]["dominant_emotion"]
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1
                total_analyzed += 1
            except:
                pass

    cap.release()

    if os.path.exists(file_path):
        os.remove(file_path)

    dominant = max(emotion_counts, key=emotion_counts.get) if total_analyzed > 0 else "neutral"

    return {
        "video": original_name,
        "dominant_emotion": dominant,
        "total_analyzed_frames": total_analyzed,
        "happy_frames": emotion_counts["happy"],
        "neutral_frames": emotion_counts["neutral"],
        "sad_frames": emotion_counts["sad"],
        "angry_frames": emotion_counts["angry"],
		
        "fear_frames": emotion_counts["fear"],
        "disgust_frames": emotion_counts["disgust"],
        "surprise_frames": emotion_counts["surprise"]
    }

@app.get("/health")
def health():
    return {"status": "Face Analysis Service is running"}

class SemesterAssignItem(BaseModel):
    candidate_id: int
    semester: int
    year: Optional[str] = None   # 🆕

class SemesterAssignItem(BaseModel):
    candidate_id: int
    semester: int
    year: Optional[str] = None 
# 🆕 ============================================
# 🆕 NEW — COURSE MASTER (for new-schedule dropdown)
# ============================================
 
@app.get("/course-master")
def get_course_master(db: Session = Depends(get_db)):
    courses = db.query(CourseMaster).order_by(CourseMaster.course_id.asc()).all()
    return [
        {
            "course_id":   c.course_id,
            "course_code": c.course_code,
            "course_name": c.course_name,
            "branch_name": c.branch_name
        }
        for c in courses
    ]
  
class CourseMasterItem(BaseModel):
    course_code: str
    course_name: str
    branch_name: str

class BulkCourseMasterPayload(BaseModel):
    courses: List[CourseMasterItem]

@app.post("/course-master/bulk-upload")
def bulk_upload_course_master(
    payload: BulkCourseMasterPayload,
    db: Session = Depends(get_db)
):
    inserted_count = 0
    skipped = []

    for item in payload.courses:
        existing = db.query(CourseMaster).filter(
            CourseMaster.course_code == item.course_code
        ).first()

        if existing:
            skipped.append(item.course_code)
            continue

        db_course = CourseMaster(
            course_code=item.course_code,
            course_name=item.course_name,
            branch_name=item.branch_name
        )
        db.add(db_course)
        inserted_count += 1

    db.commit()

    return {
        "success": True,
        "inserted_count": inserted_count,
        "skipped_existing": skipped
    }
 
# ============================================
# 🆕 NEW — FETCH CANDIDATES BY COURSE CODE (Fetch Candidates button)
# ============================================
 
@app.get("/candidates/by-course/{course_id}")
def get_candidates_by_course(
    course_id: int,
    db: Session = Depends(get_db)
):
    candidates = (
        db.query(Candidate)
        .filter(Candidate.course_id == course_id)
        .order_by(Candidate.name.asc())
        .all()
    )

    return [
        {
            "id": c.id,
            "name": c.name,
            "email": c.email,
            "phone": c.phone,
            "course_program": c.course_program,
            "course_id": c.course_id,
            "department_branch": c.department_branch,
            "semester": c.semester,
            "year": c.year
        }
        for c in candidates
    ]

#Monika code of add-question file api
# ============================================================
# REPLACE /subjects/by-course/{course_id} in main.py
# Matches exact SQL query confirmed working in pgAdmin
# ============================================================

@app.get("/subjects/by-course/{course_id}")
def get_subjects_by_course(course_id: int, db: Session = Depends(get_db)):
    """
    SELECT sbm.subject_code, sbm.subject_name, sm.semester_no, sm.semester_name,
           ssm.course_id, ssm.semester_id
    FROM semester_subject_mapping ssm
    JOIN semester_master sm ON ssm.semester_id = sm.semester_id
    JOIN subject_master sbm ON ssm.subject_id = sbm.subject_id
    WHERE ssm.course_id = :course_id
    """
    from sqlalchemy import text

    query = text("""
        SELECT DISTINCT
            sbm.subject_id,
            sbm.subject_code,
            sbm.subject_name,
            sm.semester_no,
            sm.semester_name,
            ssm.course_id,
            ssm.semester_id
        FROM public.semester_subject_mapping ssm
        JOIN public.semester_master sm ON ssm.semester_id = sm.semester_id
        JOIN public.subject_master sbm ON ssm.subject_id = sbm.subject_id
        WHERE ssm.course_id = :course_id
        ORDER BY sm.semester_no ASC, sbm.subject_code ASC
    """)

    results = db.execute(query, {"course_id": course_id}).fetchall()

    return [
        {
            "subject_id":    row.subject_id,
            "subject_code":  row.subject_code,
            "subject_name":  row.subject_name,
            "semester_no":   row.semester_no,
            "semester_name": row.semester_name,
            "course_id":     row.course_id,
            "semester_id":   row.semester_id
        }
        for row in results
    ]


# ============================================
# 🆕 NEW — ASSIGN SEMESTER BULK (Commit to Database button)
# ============================================
 
@app.post("/candidates/assign-semester")
def assign_semester_bulk(
    payload: List[SemesterAssignItem],
    db: Session = Depends(get_db)
):
    for item in payload:
        update_data = {"semester": item.semester}
        if item.year is not None:
            update_data["year"] = item.year
        db.query(Candidate).filter(
            Candidate.id == item.candidate_id
        ).update(update_data)
    db.commit()
    return {"success": True, "updated": len(payload)}
	

def call_gemini_with_retry(contents, config, max_retries=3):
    for attempt in range(max_retries):
        try:
            return genai_client.models.generate_content(
                model=PRIMARY_MODEL,
                contents=contents,
                config=config
            )
        except Exception as e:
            err_str = str(e)
            if ("503" in err_str or "UNAVAILABLE" in err_str) and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"⚠️ Gemini 503, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            raise

@app.post("/api/upload-and-generate-questions")
async def upload_pdf_and_generate(
    file: UploadFile = File(...), 
    num_questions: int = Form(3),
    course: str = Form(...),          # 👈 now plain string (course_code like "BTCS"), not the enum
    course_id: int = Form(...),       # 👈 new — from course_master
    subject_id: Optional[int] = Form(None),  # 👈 new — optional, from subject_master
    db: Session = Depends(get_db) 
):
    try:
        # Map dynamic course_code → fixed CourseProgram enum required by question_bank.course
        course_code_to_enum = {
            "BTCS": CourseProgram.B_TECH,
            "BBA":  CourseProgram.BBA,
            "MBA":  CourseProgram.MBA,
            "MCA":  CourseProgram.MCA,
            "BCOM": CourseProgram.B_COM,
            "MCOM": CourseProgram.M_COM,
        }
        course_enum = course_code_to_enum.get(course.upper())
        if not course_enum:
            raise HTTPException(
                status_code=400,
                detail=f"Unrecognized course code '{course}'. Add it to course_code_to_enum mapping in main.py."
            )

        pdf_bytes = await file.read()

        prompt = f"""
        Analyze this PDF document cleanly.
        1. Identify the high-level professional subject domain matching a {course_enum.value} program curriculum.
        2. Generate exactly {num_questions} structured interview questions anchored directly to the source text data.
        3. For each item, provide a comprehensive 'expected_answer' context block.
        4. Infer a structural difficulty tier ('Easy', 'Medium', or 'Hard') based on the complexity of the section topic parsed.
        """

        response = genai_client.models.generate_content(
            model=PRIMARY_MODEL,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QuestionGenerationSchema
            )
        )

        data = json.loads(response.text)
        subject = data.get("detected_subject", "General AI Concept").upper()
        session_category = f"{subject}_{uuid.uuid4().hex[:4]}"

        for q in data.get('questions', []):
            new_question = Question(
                question_text=q['question_text'].strip(),
                expected_answer=q['expected_answer'].strip(),
                category=session_category,
                course=course_enum,
                course_id=course_id,        # 🆕 saved
                subject_id=subject_id,      # 🆕 saved (None if AI covers all subjects)
                difficulty=q.get('difficulty', 'Medium').strip(),
                time_limit=120
            )
            db.add(new_question)

        db.commit()

        return {
            "success": True, 
            "category": session_category, 
            "course": course_enum.value,
            "course_id": course_id,
            "subject_id": subject_id,
            "display_subject": subject,
            "count": len(data.get('questions', []))
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ ERROR inside generation sequence: {str(e)}")
        return {"success": False, "error": str(e)}  



# ///////////////////////////////////////////////
@app.post("/api/save-bulk-students")
def save_bulk_students(payload: BulkRegistrationPayload, db: Session = Depends(get_db)):
    try:
        # ✅ Move lookup INSIDE the try block
        course_master_row = db.query(CourseMaster).filter(
            CourseMaster.course_code == payload.course_program
        ).first()
        resolved_course_id = course_master_row.course_id if course_master_row else None

        print(f"🔍 Resolved course_id: {resolved_course_id} for course_code: {payload.course_program}")

        inserted_count = 0
        for student in payload.students:
            db_candidate = Candidate(
                name=student.name,
                email=student.email,
                phone=student.phone,
                course_program=payload.course_program,
                course_id=resolved_course_id,        # ✅ Now in scope
                department_branch=student.department_branch,
                year=student.year  
            )
            db.add(db_candidate)
            inserted_count += 1

        db.commit()
        print(f"🎉 Successfully persisted {inserted_count} student records.")
        return {"success": True, "inserted_count": inserted_count}

    except Exception as e:
        db.rollback()
        print(f"❌ Critical Database Write Anomaly: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database write failure: {str(e)}")
    

@app.get("/api/panels/{panel_id}/members")
def get_panel_members(
    panel_id: int,
    db: Session = Depends(get_db)
):
    panel = (
        db.query(Panel)
        .filter(Panel.id == panel_id)
        .first()
    )

    if not panel:
        raise HTTPException(
            status_code=404,
            detail="Panel not found."
        )

    members = (
        db.query(PanelMember)
        .filter(PanelMember.panel_id == panel_id)
        .all()
    )

    return {
        "success": True,
        "panel": {
            "id": panel.id,
            "panel_name": panel.panel_name
        },
        "chairman_user_id": next(
            (
                m.user_id
                for m in members
                if m.role and m.role.lower() == "chairman"
            ),
            None
        ),
        "members": [
            {
                "id": m.user.id,
                "name": m.user.name,
                "email": m.user.email,
                "designation": m.user.designation,
                "role": m.role
            }
            for m in members
        ]
    }
    
@app.get("/api/interviews/load-questions/{candidate_id}/{interview_id}")
async def load_scheduled_interview_questions(
    candidate_id: int,
    interview_id: str,
    db: Session = Depends(get_db)
):

    # ----------------------------------------------------
    # Candidate
    # ----------------------------------------------------

    candidate = (
        db.query(Candidate)
        .filter(
            Candidate.id == candidate_id
        )
        .first()
    )

    if not candidate:

        raise HTTPException(
            status_code=404,
            detail="Candidate not found."
        )

    # ----------------------------------------------------
    # Interview
    # ----------------------------------------------------

    interview = (
        db.query(Interview)
        .filter(
            Interview.interview_id == interview_id
        )
        .first()
    )

    if not interview:

        raise HTTPException(
            status_code=404,
            detail="Interview not found."
        )

    # ----------------------------------------------------
    # Fetch Questions
    # ----------------------------------------------------

    if interview.interview_type == "interview":

        # Entire Course Interview

        questions = (
            db.query(Question)
            .filter(
                Question.course_id == interview.course_id
            )
            .order_by(Question.id)
            .all()
        )

    else:

        # Subject Viva

        try:

            subject_id = int(
                interview.interview_type
            )

        except ValueError:

            raise HTTPException(
                status_code=400,
                detail="Invalid interview type."
            )

        questions = (
            db.query(Question)
            .filter(
                Question.subject_id == subject_id
            )
            .order_by(Question.id)
            .all()
        )

    # ----------------------------------------------------
    # No Questions
    # ----------------------------------------------------

    if not questions:

        raise HTTPException(
            status_code=404,
            detail="No questions found for this interview."
        )

    # ----------------------------------------------------
    # Response
    # ----------------------------------------------------

    return {

        "success": True,

        "candidate_id": candidate.id,

        "candidate_name": candidate.name,

        "interview_id": interview.interview_id,

        "interview_name": interview.interview_name,

        "interview_type": interview.interview_type,

        "course_id": interview.course_id,

        "total_questions": len(questions),

        "questions": questions

    }

class GenerateGlobalQuestionsRequest(BaseModel):
    course_id: int
    subject_id: int
    question_count: int

@app.post("/api/generate-global-questions")
async def generate_global_questions(
    request: GenerateGlobalQuestionsRequest,
    db: Session = Depends(get_db)
):
    try:

        # -------------------------------------------------------
        # Fetch Course
        # -------------------------------------------------------
        course = db.query(CourseMaster).filter(
            CourseMaster.course_id == request.course_id
        ).first()

        if not course:
            raise HTTPException(
                status_code=404,
                detail="Course not found."
            )

        # -------------------------------------------------------
        # Fetch Subject
        # -------------------------------------------------------
        subject = db.query(SubjectMaster).filter(
            SubjectMaster.subject_id == request.subject_id
        ).first()

        if not subject:
            raise HTTPException(
                status_code=404,
                detail="Subject not found."
            )

        # -------------------------------------------------------
        # Prompt
        # -------------------------------------------------------
        prompt = f"""
You are an expert technical interviewer.

Generate exactly {request.question_count} interview questions.

Course:
{course.course_name}

Branch:
{course.branch_name}

Subject:
{subject.subject_name}

Instructions:

1. Generate ONLY technical interview questions.
2. Cover all important concepts of the subject.
3. Mix Easy, Medium and Hard questions.
4. Avoid duplicate questions.
5. Return ONLY JSON.
6. Do not return markdown.

Return JSON exactly like this:

{{
  "questions":[
    {{
      "question_text":"Question",
      "expected_answer":"Answer",
      "difficulty":"Easy"
    }}
  ]
}}
"""

        # -------------------------------------------------------
        # Gemini
        # -------------------------------------------------------
        response = genai_client.models.generate_content(
            model=PRIMARY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        data = json.loads(response.text)

        return {
            "success": True,
            "course": course.course_name,
            "branch": course.branch_name,
            "subject": subject.subject_name,
            "course_id": course.course_id,
            "subject_id": subject.subject_id,
            "questions": data.get("questions", [])
        }

    except HTTPException:
        raise

    except Exception as e:
        print(e)

        return {
            "success": False,
            "error": str(e)
        }


class GeneratedQuestionItem(BaseModel):
    question_text: str
    expected_answer: str
    difficulty: str
 
class SaveGlobalQuestionsRequest(BaseModel):
    course: str
    course_id: int
    subject_id: int
    questions: List[GeneratedQuestionItem]

@app.post("/api/save-global-questions")
async def save_global_questions(
    request: SaveGlobalQuestionsRequest,
    db: Session = Depends(get_db)
):
    try:

        course_code_to_enum = {
            "BTCS": CourseProgram.B_TECH,
            "BBA": CourseProgram.BBA,
            "MBA": CourseProgram.MBA,
            "MCA": CourseProgram.MCA,
            "BCOM": CourseProgram.B_COM,
            "MCOM": CourseProgram.M_COM,
        }

        course_enum = course_code_to_enum.get(request.course.upper())

        if not course_enum:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown course {request.course}"
            )

        subject = db.query(SubjectMaster).filter(
            SubjectMaster.subject_id == request.subject_id
        ).first()

        if not subject:
            raise HTTPException(
                status_code=404,
                detail="Subject not found."
            )

        saved = 0

        for q in request.questions:

            question = Question(
                question_text=q.question_text.strip(),
                expected_answer=q.expected_answer.strip(),
                category=subject.subject_name,
                difficulty=q.difficulty,
                course=course_enum,
                course_id=request.course_id,
                subject_id=request.subject_id,
                time_limit=120
            )

            db.add(question)
            saved += 1

        db.commit()

        return {
            "success": True,
            "saved": saved
        }

    except HTTPException:
        raise

    except Exception as e:
        db.rollback()

        print(e)

        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/candidates")
async def get_all_candidates(
    db: Session = Depends(get_db)
):
    candidates = (
        db.query(Candidate)
        .order_by(Candidate.name.asc())
        .all()
    )

    return {
        "success": True,
        "total_candidates": len(candidates),
        "candidates": [
            {
                "id": candidate.id,
                "name": candidate.name,
                "email": candidate.email,
                "phone": candidate.phone,
                "course_program": candidate.course_program,
                "department_branch": candidate.department_branch
            }
            for candidate in candidates
        ]
    }

@app.post("/api/admin/jd-assessment/generate")
async def generate_jd_assessment(
    candidate_id: int = Form(...),
    resume: UploadFile = File(...),
    num_questions: int = Form(10),
    db: Session = Depends(get_db)
):
    resume_path = None

    try:

        # ==========================================================
        # 1. FETCH CANDIDATE
        # ==========================================================

        candidate = (
            db.query(Candidate)
            .filter(Candidate.id == candidate_id)
            .first()
        )

        if not candidate:
            raise HTTPException(
                status_code=404,
                detail="Candidate not found."
            )

        # ==========================================================
        # 2. VALIDATE RESUME
        # ==========================================================

        if not resume.filename:
            raise HTTPException(
                status_code=400,
                detail="Resume file is required."
            )

        file_extension = (
            os.path.splitext(resume.filename)[1]
            .lower()
        )

        if file_extension != ".pdf":
            raise HTTPException(
                status_code=400,
                detail="Currently only PDF resumes are supported."
            )

        # ==========================================================
        # 3. READ RESUME
        # ==========================================================

        pdf_bytes = await resume.read()

        if not pdf_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded resume is empty."
            )

        # ==========================================================
        # 4. SAVE RESUME
        # ==========================================================

        resume_directory = os.path.join(
            "uploads",
            "resumes"
        )

        os.makedirs(
            resume_directory,
            exist_ok=True
        )

        safe_filename = (
            f"candidate_{candidate.id}_"
            f"{uuid.uuid4().hex}"
            f"{file_extension}"
        )

        resume_path = os.path.join(
            resume_directory,
            safe_filename
        )

        with open(resume_path, "wb") as buffer:
            buffer.write(pdf_bytes)

        # ==========================================================
        # 5. GEMINI PROMPT
        # ==========================================================

        prompt = f"""
You are an expert technical interviewer.

You are given the resume of a candidate.

Candidate Name:
{candidate.name}

Candidate Course/Program:
{candidate.course_program}

Candidate Department/Branch:
{candidate.department_branch}

Your task is to create a personalized technical interview
based specifically on this candidate's resume.

Generate exactly {num_questions} interview questions.

IMPORTANT RULES:

1. Questions MUST be based on information present in the resume.

2. Focus on the candidate's actual:
   - Skills
   - Technologies
   - Projects
   - Work experience
   - Education
   - Responsibilities

3. Ask questions that allow the interviewer to verify whether
   the candidate genuinely understands the experience claimed
   in the resume.

4. Do NOT create generic questions unrelated to the resume.

5. Mix Easy, Medium and Hard questions.

6. Avoid duplicate questions.

7. Expected answers must contain the important technical
   points that a strong candidate should mention.

8. Do NOT invent technologies, projects or experience that
   are not present in the resume.

9. Return exactly {num_questions} questions.

Return the result using the required JSON structure.
"""

        # ==========================================================
        # 6. GEMINI
        # ==========================================================

        response = genai_client.models.generate_content(
            model=PRIMARY_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf"
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QuestionGenerationSchema
            )
        )

        # ==========================================================
        # 7. PARSE RESPONSE
        # ==========================================================

        data = json.loads(response.text)

        generated_questions = data.get(
            "questions",
            []
        )

        if not generated_questions:
            raise HTTPException(
                status_code=500,
                detail="Gemini did not generate any questions."
            )

        # ==========================================================
        # 8. CREATE JD ASSESSMENT
        # ==========================================================

        assessment = JDAssessment(
            candidate_id=candidate.id,
            resume_path=resume_path,
            status="Generated"
        )

        db.add(assessment)

        # Get assessment ID before creating questions
        db.flush()

        # ==========================================================
        # 9. SAVE QUESTIONS
        # ==========================================================

        saved_questions = []

        for index, question in enumerate(
            generated_questions,
            start=1
        ):

            jd_question = JDQuestion(
                assessment_id=assessment.id,

                question_text=(
                    question["question_text"]
                    .strip()
                ),

                expected_answer=(
                    question["expected_answer"]
                    .strip()
                ),

                difficulty=(
                    question.get(
                        "difficulty",
                        "Medium"
                    ).strip()
                ),

                question_order=index
            )

            db.add(jd_question)

            saved_questions.append({
                "id": jd_question.id,
                "question_order": index,
                "question_text": jd_question.question_text,
                "difficulty": jd_question.difficulty
            })

        # ==========================================================
        # 10. COMMIT
        # ==========================================================

        db.commit()

        db.refresh(assessment)

        return {
            "success": True,
            "message": "JD assessment generated successfully.",

            "assessment_id": assessment.id,

            "candidate_id": candidate.id,

            "candidate_name": candidate.name,

            "resume_path": resume_path,

            "total_questions": len(
                saved_questions
            ),

            "questions": saved_questions
        }

    except HTTPException:

        db.rollback()

        if (
            resume_path
            and os.path.exists(resume_path)
        ):
            os.remove(resume_path)

        raise

    except Exception as e:

        db.rollback()

        if (
            resume_path
            and os.path.exists(resume_path)
        ):
            os.remove(resume_path)

        print(
            f"❌ JD Assessment Generation Error: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to generate JD assessment: "
                f"{str(e)}"
            )
        )

@app.post("/api/candidate/jd-assessment/submit-answer")
async def submit_jd_answer(
    assessment_id: int = Form(...),
    question_id: int = Form(...),
    candidate_id: int = Form(...),
    answer_text: str = Form(...),
    db: Session = Depends(get_db)
):

    try:

        # ==========================================================
        # 1. VERIFY CANDIDATE
        # ==========================================================

        candidate = (
            db.query(Candidate)
            .filter(
                Candidate.id == candidate_id
            )
            .first()
        )

        if not candidate:

            raise HTTPException(
                status_code=404,
                detail="Candidate not found."
            )


        # ==========================================================
        # 2. VERIFY ASSESSMENT
        # ==========================================================

        assessment = (
            db.query(JDAssessment)
            .filter(
                JDAssessment.id == assessment_id,
                JDAssessment.candidate_id == candidate_id
            )
            .first()
        )

        if not assessment:

            raise HTTPException(
                status_code=404,
                detail="JD Assessment not found."
            )


        # ==========================================================
        # 3. FIND QUESTION
        # ==========================================================

        question = (
            db.query(JDQuestion)
            .filter(
                JDQuestion.id == question_id,
                JDQuestion.assessment_id == assessment_id
            )
            .first()
        )

        if not question:

            raise HTTPException(
                status_code=404,
                detail="JD Question not found."
            )


        # ==========================================================
        # 4. VALIDATE ANSWER
        # ==========================================================

        if not answer_text or not answer_text.strip():

            raise HTTPException(
                status_code=400,
                detail="Answer cannot be empty."
            )


        # ==========================================================
        # 5. SAVE CANDIDATE ANSWER
        # ==========================================================

        question.candidate_answer = (
            answer_text.strip()
        )


        # ==========================================================
        # 6. GRADE ANSWER
        # ==========================================================

        grading = grade_jd_answer(
            candidate_answer=answer_text,
            expected_answer=question.expected_answer
        )


        question.ai_score = grading["score"]

        question.ai_feedback = grading["feedback"]


        # ==========================================================
        # 7. SAVE
        # ==========================================================

        db.commit()

        db.refresh(question)


        # ==========================================================
        # 8. RESPONSE
        # ==========================================================

        return {

            "success": True,

            "message":
                "JD answer saved and graded successfully.",

            "assessment_id":
                assessment_id,

            "question_id":
                question_id,

            "candidate_answer":
                question.candidate_answer,

            "ai_score":
                question.ai_score,

            "ai_feedback":
                question.ai_feedback
        }


    except HTTPException:

        db.rollback()

        raise


    except Exception as e:

        db.rollback()

        print(
            "❌ JD Answer Submission Error:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save JD answer: "
                f"{str(e)}"
            )
        )

JD_STOP_WORDS = {
    "the",
    "is",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "on",
    "with",
    "used",
    "use",
    "this",
    "that",
    "are",
    "was",
    "were",
    "be",
    "by",
    "as",
    "it",
    "from",
    "which",
    "can",
    "has",
    "have",
    "their",
    "they",
    "them"
}


def extract_jd_keywords(text: str):

    if not text:
        return set()

    words = re.findall(
        r"[a-zA-Z0-9+#.-]+",
        text.lower()
    )

    return {
        word
        for word in words
        if word not in JD_STOP_WORDS
        and len(word) > 2
    }


def grade_jd_answer(
    candidate_answer: str,
    expected_answer: str
):

    if not candidate_answer:

        return {
            "score": 0,
            "feedback": "No answer was provided."
        }


    if not expected_answer:

        return {
            "score": 0,
            "feedback":
                "Expected answer is not available."
        }


    candidate_keywords = extract_jd_keywords(
        candidate_answer
    )

    expected_keywords = extract_jd_keywords(
        expected_answer
    )


    if not expected_keywords:

        return {
            "score": 0,
            "feedback":
                "Unable to evaluate this answer."
        }


    matched_keywords = (
        candidate_keywords &
        expected_keywords
    )


    score = (
        len(matched_keywords)
        /
        len(expected_keywords)
    ) * 100


    score = min(
        round(score),
        100
    )


    # ==========================================================
    # FEEDBACK
    # ==========================================================

    if score >= 80:

        feedback = (
            "Excellent answer. The candidate "
            "covered most of the important "
            "technical concepts."
        )

    elif score >= 60:

        feedback = (
            "Good answer. The candidate covered "
            "several important technical concepts, "
            "but some details could be improved."
        )

    elif score >= 40:

        feedback = (
            "Average answer. The candidate "
            "mentioned some relevant concepts, "
            "but several important points were missing."
        )

    else:

        feedback = (
            "Weak answer. The candidate did not "
            "cover enough of the important technical "
            "concepts."
        )


    return {
        "score": score,
        "feedback": feedback
    }

@app.post(
    "/api/candidate/jd-assessment/{assessment_id}/complete"
)
def complete_jd_assessment(
    assessment_id: int,
    candidate_id: int,
    db: Session = Depends(get_db)
):

    try:

        # ==========================================================
        # 1. FIND ASSESSMENT
        # ==========================================================

        assessment = (
            db.query(JDAssessment)
            .filter(
                JDAssessment.id == assessment_id,
                JDAssessment.candidate_id == candidate_id
            )
            .first()
        )

        if not assessment:

            raise HTTPException(
                status_code=404,
                detail="JD Assessment not found."
            )


        # ==========================================================
        # 2. CHECK QUESTIONS
        # ==========================================================

        questions = (
            db.query(JDQuestion)
            .filter(
                JDQuestion.assessment_id == assessment_id
            )
            .all()
        )


        if not questions:

            raise HTTPException(
                status_code=400,
                detail="No questions found for this assessment."
            )


        # ==========================================================
        # 3. CHECK ANSWERS
        # ==========================================================

        unanswered_questions = [

            question.id

            for question in questions

            if not question.candidate_answer
            or not question.candidate_answer.strip()
        ]


        if unanswered_questions:

            raise HTTPException(
                status_code=400,
                detail=(
                    "All questions must be answered "
                    "before completing the assessment."
                )
            )


        # ==========================================================
        # 4. UPDATE STATUS
        # ==========================================================

        assessment.status = "Completed"


        db.commit()

        db.refresh(assessment)


        # ==========================================================
        # 5. CALCULATE RESULT
        # ==========================================================

        total_questions = len(questions)

        total_score = sum(
            question.ai_score or 0
            for question in questions
        )


        percentage = round(
            total_score / total_questions,
            2
        )


        # ==========================================================
        # 6. RESULT GRADE
        # ==========================================================

        if percentage >= 80:

            grade = "Excellent"

        elif percentage >= 60:

            grade = "Good"

        elif percentage >= 40:

            grade = "Average"

        else:

            grade = "Needs Improvement"


        return {

            "success": True,

            "message":
                "JD Assessment completed successfully.",

            "assessment_id":
                assessment.id,

            "candidate_id":
                candidate_id,

            "status":
                assessment.status,

            "total_questions":
                total_questions,

            "total_score":
                total_score,

            "percentage":
                percentage,

            "grade":
                grade
        }


    except HTTPException:

        db.rollback()

        raise


    except Exception as e:

        db.rollback()

        print(
            "❌ JD Assessment Completion Error:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to complete JD assessment: "
                f"{str(e)}"
            )
        )

@app.get("/api/candidate/jd-assessment/{candidate_id}")
def get_candidate_jd_assessment(
    candidate_id: int,
    db: Session = Depends(get_db)
):

    try:

        # ==========================================================
        # 1. VERIFY CANDIDATE
        # ==========================================================

        candidate = (
            db.query(Candidate)
            .filter(
                Candidate.id == candidate_id
            )
            .first()
        )

        if not candidate:
            raise HTTPException(
                status_code=404,
                detail="Candidate not found."
            )


        # ==========================================================
        # 2. FIND JD ASSESSMENT
        # ==========================================================

        assessment = (
            db.query(JDAssessment)
            .filter(
                JDAssessment.candidate_id == candidate_id
            )
            .order_by(
                JDAssessment.created_at.desc()
            )
            .first()
        )

        if not assessment:
            raise HTTPException(
                status_code=404,
                detail="No JD Assessment found for this candidate."
            )


        # ==========================================================
        # 3. GET QUESTIONS
        # ==========================================================

        questions = (
            db.query(JDQuestion)
            .filter(
                JDQuestion.assessment_id == assessment.id
            )
            .order_by(
                JDQuestion.question_order.asc()
            )
            .all()
        )


        if not questions:
            raise HTTPException(
                status_code=404,
                detail="No questions found for this JD Assessment."
            )


        # ==========================================================
        # 4. RETURN QUESTIONS
        # ==========================================================

        return {

            "success": True,

            "assessment_id":
                assessment.id,

            "candidate_id":
                candidate.id,

            "candidate_name":
                candidate.name,

            "status":
                assessment.status,

            "total_questions":
                len(questions),

            "questions": [

                {
                    "id":
                        question.id,

                    "question_order":
                        question.question_order,

                    "question_text":
                        question.question_text,

                    "difficulty":
                        question.difficulty

                }

                for question in questions
            ]
        }


    except HTTPException:

        raise


    except Exception as e:

        print(
            "❌ Failed to fetch JD Assessment:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to fetch JD Assessment: "
                f"{str(e)}"
            )
        )

@app.post("/api/interviews/save-video-analysis")
def save_video_analysis(
    request: VideoAnalysisRequest,
    db: Session = Depends(get_db)
):

    analysis = request.analysis

    record = InterviewVideoAnalysis(

        interview_id=request.interview_id,

        candidate_id=request.candidate_id,

        interview_mode=request.interview_mode,

        video=analysis.get("video"),

        dominant_emotion=analysis.get("dominant_emotion"),

        total_analyzed_frames=analysis.get("total_analyzed_frames"),

        happy_frames=analysis.get("happy_frames"),

        neutral_frames=analysis.get("neutral_frames"),

        sad_frames=analysis.get("sad_frames"),

        angry_frames=analysis.get("angry_frames"),

        fear_frames=analysis.get("fear_frames"),

        disgust_frames=analysis.get("disgust_frames"),

        surprise_frames=analysis.get("surprise_frames")

    )

    db.add(record)

    db.commit()

    db.refresh(record)

    return {
        "success": True,
        "message": "Video analysis saved successfully.",
        "id": record.id
    }

# ============================================================
# IMAGE HELPER
# ============================================================

async def read_uploaded_frame(file: UploadFile):

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Empty image received."
        )

    np_arr = np.frombuffer(
        contents,
        np.uint8
    )

    frame = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image frame."
        )

    return frame

# ============================================================
# MEDIAPIPE IMAGE CONVERSION
# ============================================================

def convert_to_mp_image(frame):

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


# ============================================================
# FACE BOUNDING BOX FROM LANDMARKS
# ============================================================

def get_landmark_bbox(face_landmarks, frame_width, frame_height):

    if not face_landmarks:
        return None

    xs = []
    ys = []

    for landmark in face_landmarks:

        x = max(0.0, min(1.0, float(landmark.x)))
        y = max(0.0, min(1.0, float(landmark.y)))

        xs.append(x * frame_width)
        ys.append(y * frame_height)

    if not xs or not ys:
        return None

    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)

    return {
        "x": left,
        "y": top,
        "width": max(0.0, right - left),
        "height": max(0.0, bottom - top)
    }


# ============================================================
# MEDIAPIPE FACE ANALYSIS
# ============================================================

def analyze_face(frame):

    # FaceLandmarker is the required detector for this backend.
    # The separate FaceDetector is optional and, if available,
    # provides a confidence score and native bounding boxes.
    if face_landmarker is None:

        raise RuntimeError(
            "MediaPipe Face Landmarker is not initialized. "
            "Check the face_landmarker.task model and startup logs."
        )

    mp_image = convert_to_mp_image(frame)

    # --------------------------------------------------------
    # LANDMARKER ALSO DETECTS FACES
    # --------------------------------------------------------

    landmark_result = face_landmarker.detect(mp_image)

    landmark_faces = (
        landmark_result.face_landmarks
        if landmark_result
        else []
    )

    face_count = len(landmark_faces)

    face_visible = face_count > 0

    # --------------------------------------------------------
    # OPTIONAL NATIVE FACE DETECTOR
    # --------------------------------------------------------

    detections = []

    if face_detector is not None:

        try:

            detection_result = face_detector.detect(mp_image)

            detections = (
                detection_result.detections
                if detection_result
                else []
            )

        except Exception as e:

            print(
                "⚠️ FaceDetector failed during frame analysis; "
                "using FaceLandmarker fallback:",
                str(e)
            )

    # --------------------------------------------------------
    # FACE CONFIDENCE
    # --------------------------------------------------------

    face_confidence = 0.0

    if detections:

        confidence_values = []

        for detection in detections:

            if not detection.categories:
                continue

            confidence_values.append(
                float(detection.categories[0].score)
            )

        if confidence_values:
            face_confidence = max(confidence_values)

    elif face_count > 0:

        # FaceLandmarker has already accepted the face.
        # There is no FaceDetector score in the fallback path.
        face_confidence = 1.0

    # --------------------------------------------------------
    # FACE SIZE
    # --------------------------------------------------------

    frame_height, frame_width = frame.shape[:2]

    face_area_ratio = 0.0

    if face_count == 1:

        if detections:

            bbox = detections[0].bounding_box

            face_width = float(bbox.width)
            face_height = float(bbox.height)

        else:

            bbox = get_landmark_bbox(
                landmark_faces[0],
                frame_width,
                frame_height
            )

            face_width = (
                bbox["width"]
                if bbox
                else 0.0
            )

            face_height = (
                bbox["height"]
                if bbox
                else 0.0
            )

        face_area = face_width * face_height
        frame_area = frame_width * frame_height

        if frame_area > 0:
            face_area_ratio = face_area / frame_area

    face_too_small = (
        face_count == 1
        and face_area_ratio < MIN_FACE_AREA_RATIO
    )

    return {
        "face_count": face_count,
        "face_visible": face_visible,
        "face_confidence": round(face_confidence, 3),
        "face_area_ratio": round(face_area_ratio, 4),
        "face_too_small": face_too_small,
        "landmark_result": landmark_result,
        "landmark_faces": landmark_faces
    }


# ============================================================
# HEAD POSE
# ============================================================

def calculate_head_pose(landmark_result):

    default_pose = {
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0
    }

    if landmark_result is None:
        return default_pose

    matrices = (
        landmark_result
        .facial_transformation_matrixes
    )

    if not matrices:
        return default_pose

    try:

        matrix = np.array(
            matrices[0],
            dtype=np.float64
        )

        # ----------------------------------------------------
        # MediaPipe normally provides a 4x4 transformation
        # matrix.
        # ----------------------------------------------------

        if matrix.shape != (4, 4):

            return default_pose

        rotation_matrix = matrix[:3, :3]

        # OpenCV Euler decomposition
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(
            rotation_matrix
        )

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        return {
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll": round(roll, 2)
        }

    except Exception as e:

        print(
            "⚠️ Head pose calculation error:",
            str(e)
        )

        return default_pose

# ============================================================
# LOOKING AWAY DETECTION
# ============================================================

def detect_looking_away(head_pose):

    yaw = abs(
        float(head_pose.get("yaw", 0))
    )

    pitch = abs(
        float(head_pose.get("pitch", 0))
    )

    roll = abs(
        float(head_pose.get("roll", 0))
    )

    yaw_away = (
        yaw > HEAD_YAW_THRESHOLD
    )

    pitch_away = (
        pitch > HEAD_PITCH_THRESHOLD
    )

    roll_away = (
        roll > HEAD_ROLL_THRESHOLD
    )

    looking_away = (
        yaw_away
        or pitch_away
        or roll_away
    )

    return {
        "looking_away": looking_away,

        "yaw_exceeded": yaw_away,

        "pitch_exceeded": pitch_away,

        "roll_exceeded": roll_away
    }

# ============================================================
# YOLO PROCTORING DETECTION
# ============================================================

def analyze_yolo_objects(frame):

    result = {
        "person_count": 0,
        "phone_detected": False,
        "phone_confidence": 0.0,
        "objects": []
    }

    if yolo_model is None:

        return result

    try:

        predictions = yolo_model(
            frame,
            verbose=False,
            conf=YOLO_CONFIDENCE
        )

        for prediction in predictions:

            if prediction.boxes is None:
                continue

            for box in prediction.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                class_name = (
                    yolo_model.names[class_id]
                    if class_id in yolo_model.names
                    else str(class_id)
                )

                result["objects"].append({
                    "class": class_name,
                    "confidence": round(
                        confidence,
                        3
                    )
                })

                # --------------------------------------------
                # PERSON
                # --------------------------------------------

                if class_name == "person":

                    result["person_count"] += 1

                # --------------------------------------------
                # MOBILE PHONE
                # --------------------------------------------

                if class_name == "cell phone":

                    result["phone_detected"] = True

                    result["phone_confidence"] = max(
                        result["phone_confidence"],
                        confidence
                    )

    except Exception as e:

        print(
            "❌ YOLO proctoring error:",
            str(e)
        )

    return result

# ============================================================
# COMPLETE PROCTORING ANALYSIS
# ============================================================

def analyze_proctoring_frame(frame):

    alerts = []

    severity = "NORMAL"

    # ========================================================
    # FACE
    # ========================================================

    face_analysis = analyze_face(frame)

    face_count = face_analysis[
        "face_count"
    ]

    face_visible = face_analysis[
        "face_visible"
    ]

    face_confidence = face_analysis[
        "face_confidence"
    ]

    face_too_small = face_analysis[
        "face_too_small"
    ]

    # ========================================================
    # FACE NOT DETECTED
    # ========================================================

    if not face_visible:

        alerts.append({
            "code": "FACE_NOT_VISIBLE",
            "message": (
                "Candidate face is not visible."
            ),
            "severity": "HIGH"
        })

        severity = "HIGH"

    # ========================================================
    # MULTIPLE FACES
    # ========================================================

    if face_count > 1:

        alerts.append({
            "code": "MULTIPLE_FACES",
            "message": (
                "Multiple faces detected."
            ),
            "severity": "HIGH"
        })

        severity = "HIGH"

    # ========================================================
    # FACE TOO SMALL / FAR AWAY
    # ========================================================

    if face_too_small:

        alerts.append({
            "code": "FACE_TOO_FAR",
            "message": (
                "Please move closer to the camera."
            ),
            "severity": "WARNING"
        })

        if severity == "NORMAL":
            severity = "WARNING"

    # ========================================================
    # HEAD POSE
    # ========================================================

    head_pose = {
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0
    }

    looking_away = False

    pose_status = {
        "yaw_exceeded": False,
        "pitch_exceeded": False,
        "roll_exceeded": False
    }

    if (
        face_count == 1
        and len(face_analysis["landmark_faces"]) == 1
    ):

        head_pose = calculate_head_pose(
            face_analysis["landmark_result"]
        )

        pose_status = detect_looking_away(
            head_pose
        )

        looking_away = pose_status[
            "looking_away"
        ]

        # ----------------------------------------------------
        # HEAD LOOKING AWAY
        # ----------------------------------------------------

        if looking_away:

            alerts.append({
                "code": "LOOKING_AWAY",
                "message": (
                    "Candidate appears to be "
                    "looking away from the camera."
                ),
                "severity": "WARNING"
            })

            if severity == "NORMAL":
                severity = "WARNING"

    # ========================================================
    # EYE GAZE
    # ========================================================

    eye_gaze = {
        "available": False,
        "looking_away": False,
        "direction": "CENTER",
        "horizontal": 0.0,
        "vertical": 0.0,
        "left_eye": None,
        "right_eye": None
    }

    if (
        face_count == 1
        and len(face_analysis["landmark_faces"]) == 1
    ):

        eye_gaze = detect_eye_gaze(
            face_analysis["landmark_result"]
        )

        # ----------------------------------------------------
        # EYE MOVEMENT / GAZE AWAY
        # ----------------------------------------------------

        if eye_gaze["looking_away"]:

            direction = eye_gaze.get(
                "direction",
                "away"
            )

            alerts.append({
                "code": "EYE_GAZE_AWAY",
                "message": (
                    "Candidate appears to be "
                    f"looking {direction.lower()} "
                    "away from the camera."
                ),
                "severity": "WARNING",
                "direction": direction
            })

            if severity == "NORMAL":
                severity = "WARNING"

    # ========================================================
    # YOLO
    # ========================================================

    yolo_analysis = analyze_yolo_objects(
        frame
    )

    person_count = yolo_analysis[
        "person_count"
    ]

    phone_detected = yolo_analysis[
        "phone_detected"
    ]

    phone_confidence = yolo_analysis[
        "phone_confidence"
    ]

    # ========================================================
    # MULTIPLE PERSONS
    # ========================================================

    if person_count > 1:

        alerts.append({
            "code": "MULTIPLE_PERSONS",
            "message": (
                "More than one person is visible."
            ),
            "severity": "HIGH"
        })

        severity = "HIGH"

    # ========================================================
    # PHONE
    # ========================================================

    if phone_detected:

        alerts.append({
            "code": "PHONE_DETECTED",
            "message": (
                "Mobile phone detected."
            ),
            "severity": "HIGH",
            "confidence": round(
                phone_confidence,
                3
            )
        })

        severity = "HIGH"

    # ========================================================
    # FINAL RESULT
    # ========================================================

    result = {

        "success": True,

        # ----------------------------------------------------
        # FACE
        # ----------------------------------------------------

        "face_count":
            face_count,

        "face_visible":
            face_visible,

        "face_confidence":
            face_confidence,

        "face_too_small":
            face_too_small,

        # ----------------------------------------------------
        # HEAD
        # ----------------------------------------------------

        "head_pose":
            head_pose,

        "looking_away":
            looking_away,

        "pose_status":
            pose_status,

        # ----------------------------------------------------
        # EYE GAZE
        # ----------------------------------------------------

        "eye_gaze":
            eye_gaze,

        # ----------------------------------------------------
        # OBJECTS
        # ----------------------------------------------------

        "person_count":
            person_count,

        "phone_detected":
            phone_detected,

        "phone_confidence":
            round(
                phone_confidence,
                3
            ),

        "objects":
            yolo_analysis["objects"],

        # ----------------------------------------------------
        # ALERTS
        # ----------------------------------------------------

        "alerts":
            alerts,

        "alert_count":
            len(alerts),

        "severity":
            severity
    }

    return make_json_safe(result)
# ============================================================
# PROCTORING - SINGLE FRAME
# ============================================================

@app.post("/api/proctoring/check-frame")
async def check_proctoring_frame(
    file: UploadFile = File(...)
):

    try:

        frame = await read_uploaded_frame(
            file
        )

        result = analyze_proctoring_frame(
            frame
        )

        return result

    except HTTPException:
        raise

    except Exception as e:

        print(
            "❌ Proctoring frame error:",
            str(e)
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Proctoring analysis failed: {str(e)}"
            )
        )

# ============================================================
# PROCTORING - PRE CHECK
# ============================================================

@app.post("/api/proctoring/precheck")
async def proctoring_precheck(
    file: UploadFile = File(...)
):

    try:

        frame = await read_uploaded_frame(file)

        analysis = analyze_proctoring_frame(frame)

        # ====================================================
        # PRECHECK CONDITIONS
        # ====================================================

        face_check = bool(
            analysis["face_count"] == 1
        )

        no_phone = bool(
            not analysis["phone_detected"]
        )

        no_extra_person = bool(
            analysis["person_count"] <= 1
        )

        face_visible = bool(
            analysis["face_visible"]
        )

        face_size_ok = bool(
            not analysis["face_too_small"]
        )

        looking_at_camera = bool(
            not analysis["looking_away"]
        )

        # ====================================================
        # ALL CHECKS
        # ====================================================

        checks = {

            "face_detected":
                face_visible,

            "single_face":
                face_check,

            "face_size_ok":
                face_size_ok,

            "looking_at_camera":
                looking_at_camera,

            "no_phone":
                no_phone,

            "no_extra_person":
                no_extra_person
        }

        # ====================================================
        # PRECHECK RESULT
        # ====================================================

        precheck_passed = all(
            checks.values()
        )

        # ====================================================
        # USER MESSAGE
        # ====================================================

        if not face_visible:

            message = (
                "Please position your face "
                "inside the camera frame."
            )

        elif not face_check:

            message = (
                "Only one face should be visible "
                "during the assessment."
            )

        elif not face_size_ok:

            message = (
                "Please move closer to the camera."
            )

        elif not looking_at_camera:

            message = (
                "Please look directly at the camera."
            )

        elif not no_extra_person:

            message = (
                "Only the candidate should be "
                "visible."
            )

        elif not no_phone:

            message = (
                "Please remove your mobile phone "
                "from the camera view."
            )

        else:

            message = (
                "All camera and environment "
                "checks passed."
            )

        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        response = {

            "success": True,

            "precheck_passed":
                bool(precheck_passed),

            "message":
                message,

            "checks":
                checks,

            "analysis":
                analysis
        }

        return make_json_safe(response)

    except HTTPException:
        raise

    except Exception as e:

        print(
            "❌ Proctoring precheck error:",
            str(e)
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Precheck failed: {str(e)}"
        )


# ============================================================
# EYE GAZE DETECTION
# ============================================================

def detect_eye_gaze(landmark_result):

    result = {
        "available": False,
        "looking_away": False,
        "direction": "CENTER",
        "horizontal": 0.0,
        "vertical": 0.0,
        "left_eye": None,
        "right_eye": None
    }

    if landmark_result is None:
        return result

    if not landmark_result.face_landmarks:
        return result

    try:

        landmarks = landmark_result.face_landmarks[0]

        # MediaPipe iris landmarks
        #
        # Left iris: 468 - 472
        # Right iris: 473 - 477

        if len(landmarks) < 478:
            return result

        # ----------------------------------------------------
        # IRIS CENTERS
        # ----------------------------------------------------

        left_iris_points = landmarks[468:473]
        right_iris_points = landmarks[473:478]

        left_iris_x = np.mean(
            [p.x for p in left_iris_points]
        )

        left_iris_y = np.mean(
            [p.y for p in left_iris_points]
        )

        right_iris_x = np.mean(
            [p.x for p in right_iris_points]
        )

        right_iris_y = np.mean(
            [p.y for p in right_iris_points]
        )

        # ----------------------------------------------------
        # EYE CORNERS
        # ----------------------------------------------------

        # Left eye
        left_inner = landmarks[133]
        left_outer = landmarks[33]

        # Right eye
        right_inner = landmarks[362]
        right_outer = landmarks[263]

        # ----------------------------------------------------
        # EYE TOP / BOTTOM
        # ----------------------------------------------------

        left_top = landmarks[159]
        left_bottom = landmarks[145]

        right_top = landmarks[386]
        right_bottom = landmarks[374]

        # ----------------------------------------------------
        # HORIZONTAL NORMALIZATION
        # ----------------------------------------------------

        left_eye_width = abs(
            left_outer.x - left_inner.x
        )

        right_eye_width = abs(
            right_outer.x - right_inner.x
        )

        if (
            left_eye_width <= 0
            or right_eye_width <= 0
        ):
            return result

        left_horizontal = (
            left_iris_x - min(
                left_inner.x,
                left_outer.x
            )
        ) / left_eye_width

        right_horizontal = (
            right_iris_x - min(
                right_inner.x,
                right_outer.x
            )
        ) / right_eye_width

        horizontal = (
            left_horizontal +
            right_horizontal
        ) / 2.0

        # ----------------------------------------------------
        # VERTICAL NORMALIZATION
        # ----------------------------------------------------

        left_eye_height = abs(
            left_bottom.y -
            left_top.y
        )

        right_eye_height = abs(
            right_bottom.y -
            right_top.y
        )

        if (
            left_eye_height <= 0
            or right_eye_height <= 0
        ):
            return result

        left_vertical = (
            left_iris_y -
            min(
                left_top.y,
                left_bottom.y
            )
        ) / left_eye_height

        right_vertical = (
            right_iris_y -
            min(
                right_top.y,
                right_bottom.y
            )
        ) / right_eye_height

        vertical = (
            left_vertical +
            right_vertical
        ) / 2.0

        # ----------------------------------------------------
        # CENTER = ~0.5
        # ----------------------------------------------------

        horizontal_offset = (
            horizontal - 0.5
        )

        vertical_offset = (
            vertical - 0.5
        )

        # ----------------------------------------------------
        # DETERMINE DIRECTION
        # ----------------------------------------------------

        horizontal_away = (
            abs(horizontal_offset)
            > EYE_GAZE_HORIZONTAL_THRESHOLD
        )

        vertical_away = (
            abs(vertical_offset)
            > EYE_GAZE_VERTICAL_THRESHOLD
        )

        direction = "CENTER"

        if horizontal_away:

            if horizontal_offset < 0:
                direction = "LEFT"
            else:
                direction = "RIGHT"

        elif vertical_away:

            if vertical_offset < 0:
                direction = "UP"
            else:
                direction = "DOWN"

        looking_away = (
            horizontal_away
            or vertical_away
        )

        result.update({
            "available": True,
            "looking_away": bool(looking_away),
            "direction": str(direction),

            "horizontal": float(
                round(float(horizontal_offset), 3)
            ),

            "vertical": float(
                round(float(vertical_offset), 3)
            ),

            "left_eye": {
                "horizontal": float(
                    round(float(left_horizontal), 3)
                ),
                "vertical": float(
                    round(float(left_vertical), 3)
                )
            },

            "right_eye": {
                "horizontal": float(
                    round(float(right_horizontal), 3)
                ),
                "vertical": float(
                    round(float(right_vertical), 3)
                )
            }
        })

        return result

    except Exception as e:

        print(
            "⚠️ Eye gaze detection error:",
            str(e)
        )

        return result

def make_json_safe(obj):
    """
    Convert NumPy / OpenCV / MediaPipe values
    into standard JSON-serializable Python values.
    """

    if isinstance(obj, dict):
        return {
            str(key): make_json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            make_json_safe(value)
            for value in obj
        ]

    # NumPy scalar -> Python scalar
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    # NumPy array -> Python list
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass

    return obj

