import os
import io
import json
import time
from typing import Optional, List
import tempfile
import psycopg2
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from pdf2docx import Converter
from groq import Groq 
from dotenv import load_dotenv

load_dotenv() 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# --- DATABASE LOGIC ---
def init_db():
    if not DATABASE_URL: return
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                cv_preview TEXT
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e: print(f"DB Error: {e}")

def log_usage(ip: str, text: str):
    if not DATABASE_URL: return
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        cur = conn.cursor()
        cur.execute("INSERT INTO usage_logs (ip_address, cv_preview) VALUES (%s, %s)", (ip, text[:500]))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e: print(f"Log Error: {e}")

# --- APP SETUP ---
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CVPersonalInfo(BaseModel):
    name: str
    email: str
    phone: str
    linkedin: str
    location: str

class CVExperienceBullet(BaseModel):
    id: str
    text: str

class CVExperience(BaseModel):
    id: str
    title: str
    company: str
    location: str
    dates: str
    bullets: list[CVExperienceBullet]

class CVEducationDetail(BaseModel):
    id: str
    text: str

class CVEducation(BaseModel):
    id: str
    degree: str
    institution: str
    dates: str
    details: list[CVEducationDetail]

class CVProject(BaseModel):
    id: str
    name: str
    description: str
    technologies: list[str]

class CVStructured(BaseModel):
    personal_info: CVPersonalInfo
    summary: str
    experience: list[CVExperience]
    projects: list[CVProject]
    education: list[CVEducation]
    certifications: list[str]
    skills: list[str]

class JDData(BaseModel):
    role: str
    skills: list[str]
    responsibilities: list[str]
    qualifications: list[str]

class CVExtraction(BaseModel):
    cv_data: CVStructured
    jd_data: JDData

class Suggestion(BaseModel):
    id: str
    target_id: str
    type: str # summary, experience_bullet, project_description, education_detail, add_experience_bullet, add_project
    issue: str
    original_text: str
    replacement_text: str
    reason: str

class SkillGapCourse(BaseModel):
    topic: str
    description: str

class AnalysisResponse(BaseModel):
    is_cv: bool
    error_message: Optional[str] = None
    score: int
    match_status: str
    matched_skills: list[str]
    missing_skills: list[str]
    extraction: CVExtraction
    suggestions: list[Suggestion]
    skill_gap_courses: list[SkillGapCourse]

class AnalysisRequest(BaseModel):
    cv_text: str
    job_description: str

class CoverLetterRequest(BaseModel):
    cv_text: str
    job_description: str

@app.get("/")
async def root():
    return {"status": "online", "db": bool(DATABASE_URL)}

@app.post("/analyze")
async def analyze_cv(request: AnalysisRequest, client_request: Request):
    if not client:
        raise HTTPException(status_code=500, detail="AI Client not initialized.")
    
    client_ip = client_request.headers.get("x-forwarded-for") or client_request.client.host
    log_usage(client_ip, request.cv_text)    
    
    cv_text = request.cv_text[:6000]
    jd_text = request.job_description[:6000]

    prompt = f"""
    You are a "seen-it-all" expert career advisor and elite executive coach. 
    Analyze the following CV and Job Description with deep professional insight.
    
    Our selling point is: "Based on the data we have, we have improvements for your CV and also learning paths to strengthen you for the role."
    
    CRITICAL RULE: Your advice must be GROUNDED IN REALITY. Do not advise the candidate to lie or invent experiences. Instead, help them "appear better" by framing their existing expertise more effectively and identifying genuine skill gaps.
    
    STEP 1: Extract the CV data VERBATIM. 
    - Include Personal Info (Name, Email, Phone, LinkedIn, Location).
    - Include a Professional Summary.
    - Include ALL Experience records. Assign each record an ID like "exp_1" and each bullet point an ID like "b_1".
    - Include ALL Projects. Assign each record an ID like "proj_1".
    - Include ALL Education records. Assign each record an ID like "edu_1" and each detail bullet an ID like "ed_1".
    - Include ALL Certifications.
    - Include ALL Skills.
    
    STEP 2: Extract structured data from the JD (Role, Skills, Responsibilities, Qualifications).
    
    STEP 3: Generate a match score (0-100) and match status.
    
    STEP 4: Provide EXACTLY 5 high-impact, context-relevant improvement suggestions. 
    - DO NOT BE CARELESS. Your suggestions must reflect years of recruitment wisdom.
    - Each suggestion must reference a `target_id` from the extracted CV data.
    - `type` MUST be one of: "summary", "experience_bullet", "project_description", "education_detail", "add_experience_bullet".
    - Use "add_experience_bullet" to suggest a NEW bullet point for an experience record (target_id should be the exp_id).
    - Provide the `original_text` (empty for additions) and a `replacement_text` that demonstrates the candidate's value proposition specifically for this JD without fabricating achievements.
    - Ensure the `reason` explains the strategic advantage of this change.
    
    STEP 5: Suggest 3-4 professional learning paths or course topics to strengthen the candidate specifically for this role based on their actual missing skills.
    
    CV: {cv_text}
    JD: {jd_text}
    
    Return ONLY a JSON object with this exact structure:
    {{
      "is_cv": boolean,
      "error_message": "string",
      "score": number,
      "match_status": "string",
      "matched_skills": ["string"],
      "missing_skills": ["string"],
      "extraction": {{
        "cv_data": {{ 
          "personal_info": {{ "name": "", "email": "", "phone": "", "linkedin": "", "location": "" }},
          "summary": "",
          "experience": [
            {{ "id": "exp_1", "title": "", "company": "", "location": "", "dates": "", "bullets": [ {{ "id": "b_1", "text": "" }} ] }}
          ],
          "projects": [
            {{ "id": "proj_1", "name": "", "description": "", "technologies": [] }}
          ],
          "education": [
            {{ "id": "edu_1", "degree": "", "institution": "", "dates": "", "details": [ {{ "id": "ed_1", "text": "" }} ] }}
          ],
          "certifications": [],
          "skills": []
        }},
        "jd_data": {{ "role": "", "skills": [], "responsibilities": [], "qualifications": [] }}
      }},
      "suggestions": [
        {{ "id": "s1", "target_id": "b_1", "type": "experience_bullet", "issue": "", "original_text": "", "replacement_text": "", "reason": "" }}
      ],
      "skill_gap_courses": [{{ "topic": "string", "description": "string" }}]
    }}
    
    IMPORTANT RULES:
    1. EXTRACT DATA VERBATIM: Do not summarize or paraphrase original CV text during extraction.
    2. NO HALLUCINATIONS: If a piece of information is missing, leave the field empty.
    3. TARGETED SUGGESTIONS: Only suggest improvements for sections that exist.
    4. NON-CV CONTENT: If the document isn't a CV, set `is_cv` to false.
    5. PROFESSIONAL TONE: Suggestions should sound like they come from a top-tier career advisor.
    6. NO FABRICATION: Do not suggest adding experience or skills the candidate clearly does not have.
    """
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a career consultant specialized in high-end recruitment analysis. Your output is always strictly structured JSON. You must be extremely literal during extraction and never hallucinate data that isn't in the provided text."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=8000,
            reasoning_effort="low"
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result

    except Exception as e:
        print(f"ANALYSIS ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"AI Analysis failed: {str(e)}")

@app.post("/generate-cover-letter")
async def generate_cover_letter(request: CoverLetterRequest):
    if not client:
        raise HTTPException(status_code=500, detail="AI Client not initialized.")
    
    safe_cv = request.cv_text[:5000]
    safe_jd = request.job_description[:5000]

    prompt = f"""
    Write a high-impact, professional cover letter.
    CANDIDATE CV: {safe_cv}
    TARGET JOB: {safe_jd}
    
    INSTRUCTIONS:
    1. Focus on bridging the gap between technical expertise and business value.
    2. Mention specific tools or achievements found in the CV that match the JD.
    3. Keep it under 350 words.
    4. Use a modern, professional tone (no generic "To Whom It May Concern").
    
    Return ONLY the cover letter text.
    """
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are an executive career coach and expert cover letter writer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7 
        )
        return {"cover_letter": response.choices[0].message.content.strip()}
    except Exception as e:
        print(f"COVER LETTER ERROR: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate cover letter.")        

@app.post("/convert-pdf-to-docx")
async def convert_pdf_to_docx(file: UploadFile = File(...)):
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf_file:
            pdf_file.write(content)
            pdf_path = pdf_file.name
        docx_path = pdf_path.replace(".pdf", ".docx")
        cv = Converter(pdf_path)
        cv.convert(docx_path)
        cv.close()
        with open(docx_path, "rb") as docx_file:
            docx_data = docx_file.read()
        os.remove(pdf_path)
        if os.path.exists(docx_path): os.remove(docx_path)
        return Response(content=docx_data, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))