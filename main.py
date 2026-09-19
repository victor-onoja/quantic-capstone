import os
import io
import json
import re
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

# --- GROUNDING: stop suggestions and cover letters inventing skills ---
# The writing prompts alone do not stop the model adding tools, domains or metrics the CV
# never mentions, so every generated text goes through a fact-check pass and a code check.
# The check runs on the small model, which Groq rate-limits separately from the analysis model.
GROUNDING_MODEL = "openai/gpt-oss-20b"


def _grounding_prompt(cv_text: str, texts: List[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(texts))
    return f"""
    You are a strict fact-checker. The candidate will put each numbered text into their CV or sign it as a cover letter.

    For each text, find every claim the CV does not support: a skill, tool, technology, certification, domain,
    responsibility, achievement, number or metric that the CV does not state. Moving a metric onto a different
    achievement than the one the CV attaches it to is also unsupported.

    These are fine and must be kept: rewording, reordering and emphasis; combining facts the CV states; tying a real
    CV fact to the job ("this matches your need for..."); saying plainly that the candidate has not used something yet.

    Then return each text with the unsupported claims removed, changing as little as possible and keeping the tone,
    format and line breaks. If nothing supported is left, return an empty string for that text.

    CV:
    {cv_text}

    TEXTS:
    {numbered}

    Return ONLY a JSON object: {{"results": [{{"index": 0, "unsupported": ["each unsupported claim"], "text": "corrected text"}}]}}
    """


def remove_unsupported_claims(cv_text: str, texts: List[str]) -> List[str]:
    """Return the texts with claims the CV does not support removed. On any failure, return them unchanged."""
    try:
        response = client.chat.completions.create(
            model=GROUNDING_MODEL,
            messages=[{"role": "user", "content": _grounding_prompt(cv_text, texts)}],
            response_format={"type": "json_object"},
            max_tokens=8000,
            reasoning_effort="low"
        )
        results = json.loads(response.choices[0].message.content)["results"]
    except Exception as e:
        print(f"GROUNDING CHECK FAILED, texts returned unchecked: {e}")
        return texts

    corrected = list(texts)
    for item in results:
        index, text = item.get("index"), item.get("text")
        if isinstance(index, int) and 0 <= index < len(texts) and isinstance(text, str):
            corrected[index] = text.strip()
    return corrected


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _skill_phrases(skill: str) -> List[str]:
    """Split a missing-skill entry into phrases to look for.

    "GCP (GKE, Cloud Build)" gives ["gcp", "gke", "cloud build"]; "FinOps frameworks" also gives "finops",
    because product-style names (inner or repeated capitals) are specific enough to match on their own.
    """
    parts = [part.strip() for part in re.split(r"[(),/;]|\band\b|\bor\b", skill)]
    names = [word for part in parts for word in part.split() if re.search(r"[a-z][A-Z]|[A-Z].*[A-Z]", word)]
    return [phrase.lower() for phrase in parts + names if len(phrase) >= 2]


def _mentions(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def drop_ungrounded_suggestions(result: dict, cv_text: str) -> dict:
    """Backstop after the fact-check: drop suggestions that still name a missing skill or a number the CV lacks."""
    cv = cv_text.lower()
    absent = {
        phrase
        for skill in result.get("missing_skills") or []
        if isinstance(skill, str)
        for phrase in _skill_phrases(skill)
        if not _mentions(cv, phrase)
    }
    cv_numbers = set(_NUMBER.findall(cv_text))

    kept = []
    for suggestion in result.get("suggestions") or []:
        text = str(suggestion.get("replacement_text", "")).strip()
        if not text or text == str(suggestion.get("original_text", "")).strip():
            continue  # nothing left after grounding, or no change to suggest
        claimed = sorted(phrase for phrase in absent if _mentions(text.lower(), phrase))
        invented = sorted(set(_NUMBER.findall(text)) - cv_numbers)
        if claimed or invented:
            print(f"DROPPED UNGROUNDED SUGGESTION {suggestion.get('id')}: skills={claimed} numbers={invented}")
            continue
        kept.append(suggestion)
    result["suggestions"] = kept
    return result


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
    
    STEP 4: Provide UP TO 5 high-impact, context-relevant improvement suggestions. Fewer is right when the CV is already strong: never pad the list with a suggestion that needs an invented fact.
    - DO NOT BE CARELESS. Your suggestions must reflect years of recruitment wisdom.
    - Each suggestion must reference a `target_id` from the extracted CV data.
    - `type` MUST be one of: "summary", "experience_bullet", "project_description", "education_detail", "add_experience_bullet".
    - Use "add_experience_bullet" to suggest a NEW bullet point for an experience record (target_id should be the exp_id). Only use it to surface something the CV already shows elsewhere (for example a listed skill or project) that this role's bullets do not mention yet.
    - Provide the `original_text` (empty for additions) and a `replacement_text` that demonstrates the candidate's value proposition specifically for this JD.
    - Provide `evidence`: the exact CV text that every fact in `replacement_text` comes from.
    - Ensure the `reason` explains the strategic advantage of this change.

    GROUNDING RULES FOR `replacement_text` (the candidate pastes it straight into their CV, so an invented claim becomes a lie on their CV):
    - Every skill, tool, technology, certification, domain and responsibility in it must already appear in the CV. Better wording, ordering and emphasis are allowed; new facts are not.
    - Anything the JD asks for that the CV does not show belongs in `missing_skills` and `skill_gap_courses`, never in `replacement_text` — not as "exposure to", "familiar with", "applicable to", or "ready for" either.
    - Every number, percentage and metric must be copied exactly from the CV. Never add a metric; if a bullet has none, strengthen the wording without one.
    
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
        {{ "id": "s1", "target_id": "b_1", "type": "experience_bullet", "issue": "", "original_text": "", "replacement_text": "", "evidence": "", "reason": "" }}
      ],
      "skill_gap_courses": [{{ "topic": "string", "description": "string" }}]
    }}
    
    IMPORTANT RULES:
    1. EXTRACT DATA VERBATIM: Do not summarize or paraphrase original CV text during extraction.
    2. NO HALLUCINATIONS: If a piece of information is missing, leave the field empty.
    3. TARGETED SUGGESTIONS: Only suggest improvements for sections that exist.
    4. NON-CV CONTENT: If the document isn't a CV, set `is_cv` to false.
    5. PROFESSIONAL TONE: Suggestions should sound like they come from a top-tier career advisor.
    6. NO FABRICATION: Never put a skill, tool, metric or experience into a suggestion unless the CV states it.
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

        suggestions = [s for s in result.get("suggestions") or [] if isinstance(s, dict)]
        if suggestions:
            texts = [str(s.get("replacement_text", "")) for s in suggestions]
            for suggestion, text in zip(suggestions, remove_unsupported_claims(cv_text, texts)):
                suggestion["replacement_text"] = text
        result["suggestions"] = suggestions
        return drop_ungrounded_suggestions(result, request.cv_text)

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

    HONESTY RULES (the candidate signs this letter, so every claim must be true):
    5. Only claim skills, tools, certifications, domains and experience that appear in the CV. Never describe the candidate as proficient in, experienced with, or familiar with anything the CV does not show.
    6. Use only numbers and metrics that appear in the CV, copied exactly.
    7. For JD requirements the CV does not show, either leave them out or, at most once, say plainly that the candidate has not used it yet and name the closest experience the CV does show.
    
    Return ONLY the cover letter text.
    """
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are an executive career coach and expert cover letter writer. You never claim a skill, tool, metric or experience that the candidate's CV does not show."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=4000,
            reasoning_effort="low"
        )
        choice = response.choices[0]
        letter = (choice.message.content or "").strip()
        if not letter:
            raise RuntimeError(f"Model returned no cover letter (finish_reason={choice.finish_reason})")
        # Checked line by line (each paragraph or bullet): the checker misses claims buried in a long block.
        lines = letter.split("\n")
        filled = [i for i, line in enumerate(lines) if line.strip()]
        for i, text in zip(filled, remove_unsupported_claims(safe_cv, [lines[i] for i in filled])):
            lines[i] = text
        return {"cover_letter": re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() or letter}
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