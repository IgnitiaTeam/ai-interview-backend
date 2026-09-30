from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import List
import io
import json
import os
import uuid

from google import genai
from google.genai import types

from sqlalchemy import func
from sqlalchemy.orm import Session
from database import engine, Base, get_db, SessionLocal

from models.JdAssessmentAttempt import JdAssessmentAttempt
from models.JdAssessmentCandidate import JdAssessmentCandidate
from models.JobDescription import JobDescription
# from models.Candidate import Candidate
# from models.JDSchedule import JDSchedule

import fitz

from models.jd_schedule import JDSchedule
from models.model import Candidate
from schemas.JdFinalResultSchema import JdFinalResultSchema
from schemas.StartJdAssessmentRequest import StartJdAssessmentRequest  # PyMuPDF
from schemas.SaveJdAnswerRequest import SaveJdAnswerRequest
from schemas.GenerateJdQuestionsRequest import GenerateJdQuestionsRequest
from schemas.grading_schema import GradingSchema


router = APIRouter(
    prefix="/jd",
    tags=["Job Description"]
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not configured."
    )

gemini_client = genai.Client(
    api_key=GOOGLE_API_KEY
)

PRIMARY_MODEL = "gemini-3-flash-preview"  # ✅ Updated to latest flash model for best performance
# PRIMARY_MODEL = "gemini-2.5-flash-lite"  # ✅ Updated to latest flash model for best performance
# PRIMARY_MODEL = "gemini-2.5-flash"  # ✅ Updated to latest flash model for best performance
# ============================================================
# EXTRACT KEYWORDS FROM JD PDF
# ============================================================

Base.metadata.create_all(bind=engine)

@router.post("/extract-keywords")
async def extract_keywords(
    file: UploadFile = File(...)
):
    """
    Upload a Job Description PDF and extract relevant keywords
    using Gemini.

    Works for any domain such as:
    IT, Accounts, Finance, MBA, HR, Marketing, Sales,
    Operations, Mechanical, Civil, Healthcare, Education, Legal, etc.
    """

    # ============================================================
    # VALIDATE FILE
    # ============================================================

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # ============================================================
    # READ PDF
    # ============================================================

    try:
        file_content = await file.read()

        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded PDF is empty."
            )

        pdf_document = fitz.open(
            stream=file_content,
            filetype="pdf"
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read PDF: {str(e)}"
        )

    # ============================================================
    # EXTRACT TEXT FROM ALL PAGES
    # ============================================================

    extracted_text = ""

    try:

        for page in pdf_document:

            page_text = page.get_text()

            if page_text:
                extracted_text += page_text + "\n"

        pdf_document.close()

    except Exception as e:

        try:
            pdf_document.close()
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail=f"Unable to extract PDF text: {str(e)}"
        )

    # ============================================================
    # CHECK EXTRACTED TEXT
    # ============================================================

    if not extracted_text.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text was found in the PDF. "
                "The PDF may be scanned/image based."
            )
        )

    # ============================================================
    # SEND EXTRACTED JD TEXT TO GEMINI
    # ============================================================

    prompt = f"""
You are an expert Job Description analyzer and recruitment specialist.

Analyze the following Job Description and extract the most relevant
keywords that describe the role.

The Job Description can belong to ANY professional domain.

Possible domains include, but are not limited to:

- Software / IT
- Accounts
- Finance
- MBA / Business Administration
- HR
- Marketing
- Sales
- Operations
- Mechanical
- Civil
- Healthcare
- Education
- Legal
- Banking
- Supply Chain
- Manufacturing
- Any other professional domain

Extract meaningful keywords such as:

- Technical skills
- Non-technical skills
- Qualifications
- Certifications
- Tools
- Technologies
- Software
- Accounting concepts
- Business concepts
- Industry knowledge
- Job responsibilities
- Professional skills
- Domain-specific terminology
- Important methodologies or processes

IMPORTANT RULES:

1. Do NOT assume the JD is a software/IT job.
2. Identify the relevant domain from the actual JD.
3. Extract keywords ONLY from the provided JD.
4. Do NOT invent skills or technologies that are not present in the JD.
5. Prefer specific and meaningful keywords over generic words.
6. Avoid generic words such as "job", "candidate", "company", etc.
7. Avoid duplicate keywords.
8. Preserve commonly used abbreviations such as GST, MBA, SQL, HR, etc.
9. Return between 5 and 30 keywords depending on the JD.
10. If the JD contains fewer than 5 meaningful keywords, return only
    the keywords that are actually supported by the JD.
11. Do not force keywords just to reach a particular count.

Return ONLY valid JSON in exactly this format:

{{
    "keywords": [
        "keyword 1",
        "keyword 2",
        "keyword 3"
    ]
}}

JOB DESCRIPTION:
{extracted_text}
"""

    try:

        response = gemini_client.models.generate_content(
            model=PRIMARY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        if not response or not response.text:
            raise Exception(
                "Gemini returned an empty response."
            )

        # ========================================================
        # PARSE GEMINI RESPONSE
        # ========================================================

        data = json.loads(response.text)

        keywords = data.get("keywords", [])

        if not isinstance(keywords, list):
            raise Exception(
                "Gemini returned an invalid keyword format."
            )

        # ========================================================
        # CLEAN KEYWORDS
        # ========================================================

        cleaned_keywords = []
        seen = set()

        for keyword in keywords:

            if not keyword:
                continue

            keyword = str(keyword).strip()

            if not keyword:
                continue

            normalized = keyword.lower()

            if normalized in seen:
                continue

            seen.add(normalized)

            cleaned_keywords.append(keyword)

        # ========================================================
        # RESPONSE
        # ========================================================

        return {
            "status": "success",
            "filename": file.filename,
            "keywords": cleaned_keywords
        }

    except json.JSONDecodeError as e:

        print(
            f"❌ Gemini returned invalid JSON: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Gemini returned an invalid keyword response."
        )

    except Exception as e:

        print(
            f"❌ Gemini keyword extraction failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract keywords using Gemini: {str(e)}"
        )
    
# ============================================================
# BASIC JD KEYWORD EXTRACTION
# ============================================================

def extract_jd_keywords(text: str) -> List[str]:

    """
    Temporary keyword extraction.

    Later this function will use the AI model to identify
    meaningful JD-specific skills/keywords.
    """

    known_keywords = [

        # Programming
        "Python",
        "Java",
        "JavaScript",
        "TypeScript",
        "C",
        "C++",
        "C#",

        # Backend
        "FastAPI",
        "Django",
        "Flask",
        "Spring Boot",
        "REST API",
        "Microservices",

        # Frontend
        "Angular",
        "React",
        "Vue",

        # Database
        "PostgreSQL",
        "MySQL",
        "MongoDB",
        "SQL",
        "NoSQL",

        # Cloud / DevOps
        "AWS",
        "Azure",
        "GCP",
        "Docker",
        "Kubernetes",
        "Git",
        "GitHub",

        # AI / ML
        "Machine Learning",
        "Deep Learning",
        "Artificial Intelligence",
        "NLP",
        "LLM",

        # General
        "Data Structures",
        "Algorithms",
        "Object Oriented Programming",
        "System Design"
    ]


    text_lower = text.lower()

    found_keywords = []


    for keyword in known_keywords:

        if keyword.lower() in text_lower:

            found_keywords.append(keyword)


    return found_keywords

@router.post("/save")
async def save_jd(
    jd_title: str = Form(...),
    keywords: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    # ========================================================
    # VALIDATE JD TITLE
    # ========================================================

    jd_title = jd_title.strip()

    if not jd_title:
        raise HTTPException(
            status_code=400,
            detail="JD title is required"
        )


    # ========================================================
    # VALIDATE FILE
    # ========================================================

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="JD PDF is required"
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )


    # ========================================================
    # CONVERT KEYWORDS JSON STRING TO LIST
    # ========================================================

    try:

        keyword_list = json.loads(keywords)

    except json.JSONDecodeError:

        raise HTTPException(
            status_code=400,
            detail="Invalid keywords format"
        )


    if not isinstance(keyword_list, list):

        raise HTTPException(
            status_code=400,
            detail="Keywords must be a list"
        )


    # ========================================================
    # CLEAN KEYWORDS
    # ========================================================

    cleaned_keywords = []

    seen_keywords = set()

    for keyword in keyword_list:

        if not keyword:
            continue

        keyword = str(keyword).strip()

        if not keyword:
            continue

        # Prevent duplicate keywords
        normalized_keyword = keyword.lower()

        if normalized_keyword in seen_keywords:
            continue

        seen_keywords.add(normalized_keyword)

        cleaned_keywords.append(keyword)


    if not cleaned_keywords:

        raise HTTPException(
            status_code=400,
            detail="At least one keyword is required"
        )


    # ========================================================
    # CONVERT KEYWORDS TO TEXT
    # ========================================================

    keywords_text = ", ".join(cleaned_keywords)


    # ========================================================
    # READ PDF
    # ========================================================

    try:

        pdf_content = await file.read()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to read PDF: {str(e)}"
        )


    if not pdf_content:

        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty"
        )


    # ========================================================
    # CREATE UPLOAD DIRECTORY
    # ========================================================

    jd_upload_directory = os.path.join(
        "uploads",
        "jd"
    )

    os.makedirs(
        jd_upload_directory,
        exist_ok=True
    )


    # ========================================================
    # CREATE UNIQUE FILE NAME
    # ========================================================

    unique_filename = (
        f"{uuid.uuid4().hex}_"
        f"{file.filename}"
    )

    file_path = os.path.join(
        jd_upload_directory,
        unique_filename
    )


    # ========================================================
    # SAVE PDF
    # ========================================================

    try:

        with open(
            file_path,
            "wb"
        ) as pdf_file:

            pdf_file.write(pdf_content)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to save PDF: {str(e)}"
        )


    # ========================================================
    # SAVE JD TO DATABASE
    # ========================================================

    try:

        jd = JobDescription(

            title=jd_title,

            pdf_filename=file.filename,

            pdf_path=file_path,

            keywords=keywords_text

        )

        db.add(jd)

        db.commit()

        db.refresh(jd)


        # ====================================================
        # SUCCESS RESPONSE
        # ====================================================

        return {

            "success": True,

            "message": "JD saved successfully",

            "data": {

                "id": jd.id,

                "title": jd.title,

                "pdf_filename": jd.pdf_filename,

                "pdf_path": jd.pdf_path,

                "keywords": cleaned_keywords

            }

        }


    except Exception as e:

        # --------------------------------------------
        # Rollback DB
        # --------------------------------------------

        db.rollback()


        # --------------------------------------------
        # Remove PDF if DB save failed
        # --------------------------------------------

        if os.path.exists(file_path):

            try:

                os.remove(file_path)

            except Exception:
                pass


        print(
            f"❌ JD SAVE ERROR: {str(e)}"
        )


        raise HTTPException(
            status_code=500,
            detail=f"Unable to save JD: {str(e)}"
        )


# ============================================================
# GENERATE ASSESSMENT QUESTIONS FROM JD KEYWORDS
# ============================================================

@router.post("/generate-questions/{job_description_id}")
async def generate_assessment_questions(
    job_description_id: int,
    request: GenerateJdQuestionsRequest,
    db: Session = Depends(get_db)
):
    """
    Generate exactly 5 assessment questions based on the
    keywords stored for a Job Description.

    Each question contains:
        - question
        - expected_answer
        - time_limit (seconds)
    """

    # ========================================================
    # FETCH JOB DESCRIPTION
    # ========================================================

    jd = db.query(JobDescription).filter(
        JobDescription.id == job_description_id
    ).first()

    if not jd:
        raise HTTPException(
            status_code=404,
            detail="Job Description not found."
        )

    # ========================================================
    # VALIDATE KEYWORDS
    # ========================================================

    if not jd.keywords or not jd.keywords.strip():
        raise HTTPException(
            status_code=400,
            detail="No keywords found for this Job Description."
        )

    keywords = [
        keyword.strip()
        for keyword in jd.keywords.split(",")
        if keyword.strip()
    ]

    if not keywords:
        raise HTTPException(
            status_code=400,
            detail="No valid keywords found for this Job Description."
        )

    keywords_text = ", ".join(keywords)

    # ========================================================
    # GEMINI PROMPT
    # ========================================================

    prompt = f"""
You are an expert professional interviewer and assessment designer.

Create exactly 5 high-quality assessment questions for a
candidate based ONLY on the keywords provided below.

JOB TITLE:
{jd.title}

KEYWORDS:
{keywords_text}

IMPORTANT RULES:

1. Generate EXACTLY 5 questions.

2. Every question MUST be directly relevant to one or more
   of the provided keywords.

3. Do NOT invent technologies, tools, skills, concepts,
   qualifications, responsibilities, or domain knowledge that
   are not supported by the provided keywords.

4. Do NOT assume that the Job Description is an IT/software role.

5. The questions must be appropriate for the actual domain
   represented by the keywords.

6. Questions should test actual understanding, reasoning,
   practical knowledge, and problem solving.

7. Use a mixture of:
   - Conceptual questions
   - Practical/application-based questions
   - Scenario-based questions
   - Problem-solving questions

8. Avoid simple questions where the answer can be given with
   only a single word whenever a deeper assessment is possible.

9. Avoid duplicate or near-duplicate questions.

10. Vary the difficulty across the 5 questions.

11. For every question, provide an expected_answer.

12. The expected_answer should describe what a strong/correct
    candidate answer should contain.

13. The expected_answer is used by an AI evaluator to compare
    the candidate's response against the expected answer.

14. Do NOT write an unnecessarily long expected answer.
    Provide the important concepts, points, steps, or reasoning
    that should be present in a good answer.

15. Assign a reasonable time_limit to every question.

16. time_limit MUST be an integer representing seconds.

17. Use these general guidelines for time limits:

    - Simple conceptual question: 60-90 seconds
    - Intermediate technical/domain question: 90-120 seconds
    - Practical/problem-solving question: 120-180 seconds
    - Complex scenario-based question: 180-240 seconds

18. Choose the time limit based on the actual complexity of
    the individual question.

19. Do NOT give every question the same time limit unless
    their complexity genuinely requires it.

20. Do NOT provide the candidate's answer.

21. Return ONLY valid JSON.

Return the response in EXACTLY this format:

{{
    "questions": [
        {{
            "question": "Question 1",
            "expected_answer": "What a strong answer should contain...",
            "time_limit": 120
        }},
        {{
            "question": "Question 2",
            "expected_answer": "What a strong answer should contain...",
            "time_limit": 90
        }},
        {{
            "question": "Question 3",
            "expected_answer": "What a strong answer should contain...",
            "time_limit": 180
        }},
        {{
            "question": "Question 4",
            "expected_answer": "What a strong answer should contain...",
            "time_limit": 120
        }},
        {{
            "question": "Question 5",
            "expected_answer": "What a strong answer should contain...",
            "time_limit": 120
        }},
        {{
            "question": "Question 6",
            "expected_answer": "What a strong answer should contain...",
            "time_limit": 150
        }},
        {{
            "question": "Question 7",
            "expected_answer": "What a strong answer should contain...",
            "time_limit": 180
        }},
        {{
            "question": "Question 8",
            "expected_answer": "What a strong answer should contain...",
            "time_limit": 120
        }},
        {{
            "question": "Question 9",
            "expected_answer": "What a strong answer should contain...",
            "time_limit": 180
        }},
        {{
            "question": "Question 10",
            "expected_answer": "What a strong answer should contain...",
            "time_limit": 240
        }}
    ]
}}
"""

    # ========================================================
    # CALL GEMINI
    # ========================================================

    try:

        response = gemini_client.models.generate_content(
            model=PRIMARY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        if not response or not response.text:
            raise Exception(
                "Gemini returned an empty response."
            )

        # ====================================================
        # PARSE RESPONSE
        # ====================================================

        data = json.loads(response.text)

        questions = data.get("questions", [])

        if not isinstance(questions, list):
            raise Exception(
                "Gemini returned an invalid question format."
            )

        # ====================================================
        # CLEAN AND VALIDATE QUESTIONS
        # ====================================================

        cleaned_questions = []
        seen_questions = set()

        for item in questions:

            if not isinstance(item, dict):
                continue

            question = item.get("question")
            expected_answer = item.get("expected_answer")
            time_limit = item.get("time_limit")

            # ----------------------------------------------
            # Validate question
            # ----------------------------------------------

            if not question:
                continue

            question = str(question).strip()

            if not question:
                continue

            # ----------------------------------------------
            # Validate expected answer
            # ----------------------------------------------

            if not expected_answer:
                continue

            expected_answer = str(
                expected_answer
            ).strip()

            if not expected_answer:
                continue

            # ----------------------------------------------
            # Validate time limit
            # ----------------------------------------------

            try:
                time_limit = int(time_limit)
            except (TypeError, ValueError):
                continue

            if time_limit <= 0:
                continue

            # ----------------------------------------------
            # Prevent duplicate questions
            # ----------------------------------------------

            normalized_question = question.lower()

            if normalized_question in seen_questions:
                continue

            seen_questions.add(normalized_question)

            # ----------------------------------------------
            # Add question
            # ----------------------------------------------

            cleaned_questions.append({
                "question": question,
                "expected_answer": expected_answer,
                "time_limit": time_limit
            })

        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        if len(cleaned_questions) != 5:
            raise Exception(
                f"Gemini generated {len(cleaned_questions)} "
                f"valid questions instead of exactly 5."
            )

        # ========================================================
        # CREATE JD ASSESSMENT ATTEMPT
        # ========================================================

        # Find the next attempt number for this candidate + schedule
        last_attempt_no = (
            db.query(func.max(JdAssessmentAttempt.attempt_no))
            .filter(
                JdAssessmentAttempt.candidate_id == request.candidate_id,
                JdAssessmentAttempt.schedule_id == request.schedule_id
            )
            .scalar()
        )

        attempt_no = (last_attempt_no or 0) + 1


        # Generate a new attempt_id
        last_attempt_id = (
            db.query(func.max(JdAssessmentAttempt.attempt_id))
            .scalar()
        )

        attempt_id = (last_attempt_id or 0) + 1


        print(
            f"🆔 Creating JD attempt | "
            f"Attempt ID: {attempt_id} | "
            f"Attempt No: {attempt_no} | "
            f"Candidate: {request.candidate_id} | "
            f"Schedule: {request.schedule_id}"
        )


        # ========================================================
        # INSERT ALL 10 QUESTIONS
        # ========================================================

        for index, question in enumerate(cleaned_questions, start=1):

            attempt_question = JdAssessmentAttempt(
                attempt_id=attempt_id,
                candidate_id=request.candidate_id,
                schedule_id=request.schedule_id,
                attempt_no=attempt_no,
                question_no=index,
                question_text=question["question"],
                expected_answer=question["expected_answer"],
                time_limit=question["time_limit"],
                answer_text=None,
                ai_assessment=None,
                ai_score=None,
                started_at=None,
                answered_at=None
            )

            db.add(attempt_question)


        db.commit()


        print(
            f"✅ JD assessment attempt created successfully | "
            f"Attempt ID: {attempt_id} | "
            f"Questions: 5"
        )

        # ====================================================
        # SUCCESS RESPONSE
        # ====================================================

        return {
            "success": True,
            "message": "Assessment questions generated successfully.",
            "attempt_id": attempt_id,
            "attempt_no": attempt_no,
            "candidate_id": request.candidate_id,
            "schedule_id": request.schedule_id,
            "job_description": {
                "id": jd.id,
                "title": jd.title
            },
            "keywords": keywords,
            "questions": cleaned_questions
        }

    # ========================================================
    # INVALID JSON
    # ========================================================

    except json.JSONDecodeError as e:

        print(
            f"❌ Gemini returned invalid question JSON: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Gemini returned an invalid question response."
        )

    # ========================================================
    # OTHER ERRORS
    # ========================================================

    except Exception as e:

        print(
            f"❌ Gemini question generation failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate assessment questions: {str(e)}"
        )


@router.post("/assessment/start")
def start_jd_assessment(
    request: StartJdAssessmentRequest,
    db: Session = Depends(get_db)
):
    try:
        # ---------------------------------------------------------
        # 1. Find candidate using email
        # ---------------------------------------------------------
        candidate = (
            db.query(Candidate)
            .filter(Candidate.email == request.email)
            .first()
        )

        if not candidate:
            raise HTTPException(
                status_code=404,
                detail="Candidate not found for the provided email."
            )

        # ---------------------------------------------------------
        # 2. Find JD schedule
        # ---------------------------------------------------------
        schedule = (
            db.query(JDSchedule)
            .filter(JDSchedule.id == request.schedule_id)
            .first()
        )

        if not schedule:
            raise HTTPException(
                status_code=404,
                detail="JD schedule not found."
            )

        # ---------------------------------------------------------
        # 3. Find the JD associated with this schedule
        # ---------------------------------------------------------
        jd = (
            db.query(JobDescription)
            .filter(JobDescription.id == schedule.jd_id)
            .first()
        )

        if not jd:
            raise HTTPException(
                status_code=404,
                detail="Job Description not found for this schedule."
            )

        # ---------------------------------------------------------
        # 4. Check whether candidate already has a record
        #    for this JD schedule
        # ---------------------------------------------------------
        jd_candidate = (
            db.query(JdAssessmentCandidate)
            .filter(
                JdAssessmentCandidate.candidate_id == candidate.id,
                JdAssessmentCandidate.schedule_id == schedule.id
            )
            .first()
        )

        # ---------------------------------------------------------
        # 5. Create candidate-assessment record if it doesn't exist
        # ---------------------------------------------------------
        if not jd_candidate:

            jd_candidate = JdAssessmentCandidate(
                candidate_id=candidate.id,
                schedule_id=schedule.id,
                assessment_status="Continue",
                isapply=False
            )

            db.add(jd_candidate)

        else:

            # Already exists — mark assessment as in progress
            jd_candidate.assessment_status = "in_progress"

        db.commit()
        db.refresh(jd_candidate)

        # ---------------------------------------------------------
        # 6. Return information required by frontend
        # ---------------------------------------------------------
        return {
            "success": True,
            "message": "JD assessment started successfully.",
            "candidate_id": candidate.id,
            "candidate_name": candidate.name,
            "candidate_email": candidate.email,
            "schedule_id": schedule.id,
            "jd_id": jd.id,
            "jd_title": jd.title,
            "jd_candidate_id": jd_candidate.id,
            "assessment_status": jd_candidate.assessment_status
        }

    except HTTPException:
        raise

    except Exception as e:
        db.rollback()

        print("❌ Error starting JD assessment:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Failed to start JD assessment: {str(e)}"
        )

@router.put(
    "/assessment/{attempt_id}/question/{question_no}/answer"
)
async def save_jd_assessment_answer(
    attempt_id: int,
    question_no: int,
    request: SaveJdAnswerRequest,
    db: Session = Depends(get_db)
):
    try:
        # ---------------------------------------------------------
        # Find the existing question for this attempt
        # ---------------------------------------------------------
        attempt_question = (
            db.query(JdAssessmentAttempt)
            .filter(
                JdAssessmentAttempt.attempt_id == attempt_id,
                JdAssessmentAttempt.question_no == question_no
            )
            .first()
        )

        if not attempt_question:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Question {question_no} "
                    f"not found for attempt {attempt_id}."
                )
            )

        # ---------------------------------------------------------
        # Save candidate answer
        # ---------------------------------------------------------
        attempt_question.answer_text = request.answer_text.strip()
        attempt_question.answered_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(attempt_question)

        print(
            f"✅ JD Answer Saved | "
            f"Attempt: {attempt_id} | "
            f"Question: {question_no}"
        )

        # ---------------------------------------------------------
        # Grade the answer immediately after saving
        # ---------------------------------------------------------
        grading_result = grade_jd_answer(
            attempt_question=attempt_question,
            db=db
        )

        # ---------------------------------------------------------
        # Return save + grading result
        # ---------------------------------------------------------
        return {
            "success": True,
            "message": "Answer saved and graded successfully.",
            "attempt_id": attempt_id,
            "question_no": question_no,
            "attempt_question_id": (
                attempt_question.attempt_question_id
            ),
            "answered_at": attempt_question.answered_at,
            "ai_score": grading_result["score"],
            "ai_assessment": grading_result["feedback"]
        }

    except HTTPException:
        raise

    except Exception as e:
        db.rollback()

        print(
            f"❌ Failed to save/grade JD answer | "
            f"Attempt: {attempt_id} | "
            f"Question: {question_no} | "
            f"Error: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save and grade answer: {str(e)}"
        )

def grade_jd_answer(
    attempt_question: JdAssessmentAttempt,
    db: Session
):
    try:
        # ---------------------------------------------------------
        # Get question data
        # ---------------------------------------------------------
        question_text = attempt_question.question_text
        expected_answer = attempt_question.expected_answer
        candidate_answer = attempt_question.answer_text or ""

        print(
            f"🤖 Starting JD AI Grading | "
            f"Attempt: {attempt_question.attempt_id} | "
            f"Question: {attempt_question.question_no}"
        )

        # ---------------------------------------------------------
        # Gemini grading prompt
        # ---------------------------------------------------------
        prompt = f"""
You are a senior technical interviewer evaluating a candidate's
answer to an interview question.

QUESTION:
{question_text}

EXPECTED ANSWER:
{expected_answer}

CANDIDATE ANSWER:
{candidate_answer}

Evaluate the candidate's answer carefully.

Grading requirements:

1. Evaluate semantic and conceptual correctness.
2. Do NOT require the candidate to use the exact wording
   of the expected answer.
3. Give credit when the candidate demonstrates the correct
   underlying concept using different terminology.
4. Consider technical accuracy, completeness, relevance,
   and understanding.
5. Do not give credit for irrelevant content.
6. If the candidate's answer is empty or does not answer
   the question, give a score of 0.
7. Assign an integer score from 0 to 10.
8. Provide a short, professional explanation of the evaluation.

Return ONLY a JSON object in this format:

{{
    "score": 0,
    "feedback": "Short professional evaluation of the candidate's answer."
}}
"""

        # ---------------------------------------------------------
        # Call Gemini
        # ---------------------------------------------------------
        response = gemini_client.models.generate_content(
            model=PRIMARY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GradingSchema
            )
        )

        # ---------------------------------------------------------
        # Parse Gemini response
        # ---------------------------------------------------------
        result = json.loads(response.text)

        final_score = int(result.get("score", 0))
        final_feedback = result.get(
            "feedback",
            "No evaluation feedback was generated."
        )

        # ---------------------------------------------------------
        # Safety validation
        # ---------------------------------------------------------
        final_score = max(0, min(10, final_score))

        # ---------------------------------------------------------
        # Save grading result
        # ---------------------------------------------------------
        attempt_question.ai_score = final_score
        attempt_question.ai_assessment = final_feedback

        db.commit()
        db.refresh(attempt_question)

        print(
            f"✅ JD AI Grading Complete | "
            f"Attempt: {attempt_question.attempt_id} | "
            f"Question: {attempt_question.question_no} | "
            f"Score: {final_score}/10"
        )

        return {
            "score": final_score,
            "feedback": final_feedback
        }

    except Exception as e:
        db.rollback()

        print(
            f"❌ JD AI Grading Failed | "
            f"Attempt: {attempt_question.attempt_id} | "
            f"Question: {attempt_question.question_no} | "
            f"Error: {str(e)}"
        )

        raise

# ============================================================
# GENERATE FINAL JD ASSESSMENT RESULT
# ============================================================

@router.post("/assessment/generate-final-result/{attempt_id}")
async def generate_jd_final_result(
    attempt_id: int,
    db: Session = Depends(get_db)
):
    try:

        # ========================================================
        # 1. FETCH ALL QUESTIONS FOR THIS ATTEMPT
        # ========================================================

        attempt_questions = (
            db.query(JdAssessmentAttempt)
            .filter(
                JdAssessmentAttempt.attempt_id == attempt_id
            )
            .order_by(
                JdAssessmentAttempt.question_no
            )
            .all()
        )

        if not attempt_questions:
            raise HTTPException(
                status_code=404,
                detail=f"No questions found for attempt {attempt_id}."
            )

        # ========================================================
        # 2. VALIDATE THAT ALL QUESTIONS HAVE BEEN GRADED
        # ========================================================

        ungraded_questions = [
            question.question_no
            for question in attempt_questions
            if question.ai_score is None
        ]

        if ungraded_questions:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Questions {ungraded_questions} "
                    f"have not been graded yet."
                )
            )

        # ========================================================
        # 3. CALCULATE OVERALL SCORE
        # ========================================================

        total_questions = len(attempt_questions)

        total_score = sum(
            question.ai_score or 0
            for question in attempt_questions
        )

        maximum_score = total_questions * 10

        overall_score = total_score

        percentage = (
            (total_score / maximum_score) * 100
            if maximum_score > 0
            else 0
        )

        # Round percentage to 2 decimal places
        percentage = round(percentage, 2)

        # ========================================================
        # 4. PREPARE QUESTION EVALUATIONS FOR GEMINI
        # ========================================================

        evaluations = []

        for question in attempt_questions:

            evaluations.append({
                "question_no": question.question_no,
                "question": question.question_text,
                "expected_answer": question.expected_answer,
                "candidate_answer": question.answer_text or "",
                "score": question.ai_score,
                "evaluation": question.ai_assessment or ""
            })

        # ========================================================
        # 5. GEMINI OVERALL EVALUATION PROMPT
        # ========================================================

        prompt = f"""
You are a senior professional interviewer evaluating the
candidate's complete Job Description based interview.

The candidate answered {total_questions} questions.

Each question has already been individually evaluated by an AI
interviewer.

The individual evaluations are provided below:

{json.dumps(evaluations, indent=2)}

The calculated numerical result is:

TOTAL SCORE:
{total_score}/{maximum_score}

PERCENTAGE:
{percentage}%

Your task is to provide the candidate's OVERALL interview
evaluation.

Evaluate the candidate based on the complete interview.

Consider:

1. Overall technical/domain knowledge.
2. Understanding of the concepts.
3. Accuracy of responses.
4. Ability to explain concepts.
5. Problem-solving ability.
6. Consistency across the interview.
7. Strengths demonstrated by the candidate.
8. Areas where the candidate needs improvement.

IMPORTANT:

- Do NOT change the calculated overall score.
- Do NOT invent another score.
- The overall_score MUST be {overall_score}.
- The percentage MUST be {percentage}.
- Base your evaluation only on the provided question
  evaluations and candidate answers.
- Mention meaningful strengths rather than generic statements.
- Mention specific areas for improvement where possible.
- Keep the summary professional and concise.
- If a question was manually skipped, treat that as a
  non-attempted response and consider it when evaluating
  completeness.

Return ONLY valid JSON in exactly this format:

{{
    "overall_score": {overall_score},
    "percentage": {percentage},
    "strengths": [
        "Specific strength demonstrated by the candidate.",
        "Another specific strength demonstrated by the candidate."
    ],
    "improvements": [
        "Specific area that the candidate should improve.",
        "Another specific area that the candidate should improve."
    ],
    "summary": "A concise professional overall assessment of the candidate."
}}
"""

        # ========================================================
        # 6. CALL GEMINI
        # ========================================================

        response = gemini_client.models.generate_content(
            model=PRIMARY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=JdFinalResultSchema
            )
        )

        if not response or not response.text:
            raise Exception(
                "Gemini returned an empty final result."
            )

        # ========================================================
        # 7. PARSE GEMINI RESPONSE
        # ========================================================

        result = json.loads(response.text)

        # ========================================================
        # 8. RETURN FINAL RESULT
        # ========================================================

        print(
            f"✅ JD Final Result Generated | "
            f"Attempt: {attempt_id} | "
            f"Score: {overall_score}/{maximum_score} | "
            f"Percentage: {percentage}%"
        )

        return {
            "success": True,
            "message": "Final JD assessment result generated successfully.",

            "attempt_id": attempt_id,

            "total_questions": total_questions,

            "total_score": total_score,

            "maximum_score": maximum_score,

            "overall_score": overall_score,

            "percentage": percentage,

            "strengths": result.get(
                "strengths",
                []
            ),

            "improvements": result.get(
                "improvements",
                []
            ),

            "summary": result.get(
                "summary",
                ""
            )
        }

    except HTTPException:
        raise

    except json.JSONDecodeError as e:

        print(
            f"❌ Invalid Gemini final result JSON | "
            f"Attempt: {attempt_id} | "
            f"Error: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Gemini returned an invalid final result."
        )

    except Exception as e:

        db.rollback()

        print(
            f"❌ Failed to generate JD final result | "
            f"Attempt: {attempt_id} | "
            f"Error: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to generate final JD assessment result: "
                f"{str(e)}"
            )
        )

