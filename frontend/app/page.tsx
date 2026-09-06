"use client";

import { useState, useEffect, useRef } from "react";
import {
  Sparkles,
  Zap,
  AlertCircle,
  Search,
  PenLine,
  FileText,
  ArrowRight,
  RotateCcw,
  CheckCircle2,
} from "lucide-react";
import { UploadZone } from "@/components/UploadZone";
import { JobDescriptionInput } from "@/components/JobDescriptionInput";
import { ExtractionCard } from "@/components/ExtractionCard";
import { ResultsDashboard } from "@/components/ResultsDashboard";
import { OptimizedCVEditor } from "@/components/OptimizedCVEditor";
import { AppSidebar } from "@/components/AppSidebar";

const API_BASE =
  typeof window !== "undefined" && window.location.hostname === "localhost"
    ? "http://127.0.0.1:8000"
    : "https://quantic-capstone.onrender.com";

type Phase = "ANALYZE" | "REFINE" | "COVER_LETTER";

const STEPS = [
  { key: "ANALYZE" as Phase, label: "Analyze", icon: Search },
  { key: "REFINE" as Phase, label: "Refine", icon: PenLine },
  { key: "COVER_LETTER" as Phase, label: "Cover Letter", icon: FileText },
];

interface Experience {
  id: string;
  title: string;
  company: string;
  location: string;
  dates: string;
  bullets: { id: string; text: string }[];
}

interface Education {
  id?: string;
  degree: string;
  institution: string;
  dates: string;
  details: { id: string; text: string }[];
}

interface Project {
  id: string;
  name: string;
  description: string;
  technologies: string[];
}

interface CVData {
  personal_info: {
    name: string;
    email: string;
    phone: string;
    linkedin: string;
    location: string;
  };
  summary: string;
  experience: Experience[];
  projects: Project[];
  education: Education[];
  certifications: string[];
  skills: string[];
}

interface ExtractionData {
  skills?: string[];
  experience?: (Experience | { title: string; company: string })[];
  projects?: Project[];
  education?: (Education | { degree: string; institution: string })[];
  role?: string;
  summary?: string;
  responsibilities?: string[];
  qualifications?: string[];
}

interface Suggestion {
  id: string;
  target_id: string;
  type: string;
  issue: string;
  original_text: string;
  replacement_text: string;
  reason: string;
}

interface AnalysisResult {
  is_cv: boolean;
  error_message?: string;
  score: number;
  match_status: string;
  matched_skills: string[];
  missing_skills: string[];
  skill_gap_courses: { topic: string; description: string }[];
  extraction: {
    cv_data: CVData;
    jd_data: ExtractionData;
  };
  suggestions: Suggestion[];
}

export default function Home() {
  const [phase, setPhase] = useState<Phase>("ANALYZE");
  const [cvContent, setCvContent] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [resultData, setResultData] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTo(0, 0);
    }
  }, [phase]);

  const [coverLetter, setCoverLetter] = useState<string>("");
  const [isGeneratingLetter, setIsGeneratingLetter] = useState(false);

  const handleAnalyze = async () => {
    if (!cvContent || !jobDescription) return;

    setAnalyzing(true);
    setResultData(null);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cv_text: cvContent,
          job_description: jobDescription,
        }),
      });

      const data: AnalysisResult = await response.json();

      if (response.ok) {
        if (data.is_cv === false) {
          setError(data.error_message || "The uploaded document does not appear to be a CV. Please upload a valid resume.");
          setResultData(null);
        } else {
          setResultData(data);
        }
      } else {
        const errorData = data as unknown as { detail?: string };
        setError(errorData.detail || "Analysis failed. Please try again.");
      }
    } catch {
      setError("Cannot reach the server. Make sure the backend is running.");
    } finally {
      setAnalyzing(false);
    }
  };

  const handleGenerateCoverLetter = async () => {
    if (!cvContent || !jobDescription) return;
    setIsGeneratingLetter(true);
    setPhase("COVER_LETTER");

    try {
      const response = await fetch(`${API_BASE}/generate-cover-letter`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cv_text: cvContent,
          job_description: jobDescription,
        }),
      });

      if (!response.ok) throw new Error("Generation failed");

      const data = await response.json();
      setCoverLetter(data.cover_letter);
    } catch (err) {
      console.error(err);
      setError("Failed to generate cover letter. Please try again.");
    } finally {
      setIsGeneratingLetter(false);
    }
  };

  const canAnalyze = cvContent.length > 0 && jobDescription.length > 0;

  const handleReset = () => {
    setPhase("ANALYZE");
    setCoverLetter("");
    setResultData(null);
    setCvContent("");
    setJobDescription("");
    setError(null);
  };

  return (
    <div className="flex h-screen overflow-hidden surface-base">
      <AppSidebar
        currentPhase={phase}
        onPhaseChange={setPhase}
        hasResults={!!resultData}
      />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header
          className="shrink-0 border-b px-8 py-4 z-10"
          style={{
            borderColor: "var(--color-border)",
            backgroundColor: "color-mix(in srgb, var(--color-card) 60%, transparent)",
            backdropFilter: "blur(16px)",
          }}
        >
          <div className="flex items-center justify-between max-w-5xl mx-auto">
            <div>
              <h1 className="text-xl font-bold text-foreground">
                {phase === "ANALYZE" && "Analyze Your Fit"}
                {phase === "REFINE" && "Refine Your CV"}
                {phase === "COVER_LETTER" && "Write Cover Letter"}
              </h1>
              <p className="text-sm text-muted-foreground mt-0.5">
                {phase === "ANALYZE" &&
                  "Upload your CV and paste the job description to get started"}
                {phase === "REFINE" &&
                  "Review and apply AI-suggested improvements"}
                {phase === "COVER_LETTER" &&
                  "Generate a tailored cover letter for this role"}
              </p>
            </div>

            {/* Step Progress */}
            <div className="hidden md:flex items-center gap-1 bg-secondary p-1.5 rounded-xl">
              {STEPS.map((step, i) => {
                const isActive = phase === step.key;
                const isDone =
                  i < STEPS.findIndex((s) => s.key === phase);
                const StepIcon = step.icon;

                return (
                  <button
                    key={step.key}
                    onClick={() => {
                      if (isDone || isActive) setPhase(step.key);
                    }}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      isActive
                        ? "bg-card text-foreground shadow-sm"
                        : isDone
                          ? "text-primary cursor-pointer hover:bg-card/50"
                          : "text-muted-foreground cursor-default opacity-40"
                    }`}
                  >
                    {isDone ? (
                      <CheckCircle2 className="h-3.5 w-3.5" />
                    ) : (
                      <StepIcon className="h-3.5 w-3.5" />
                    )}
                    {step.label}
                  </button>
                );
              })}
            </div>
          </div>
        </header>

        {/* Content */}
        <div ref={scrollContainerRef} className="flex-1 overflow-y-auto">
          <div className="max-w-5xl mx-auto px-8 py-10 pb-24 space-y-10">
            {/* Error banner */}
            {error && (
              <div className="flex items-center gap-3 p-4 text-sm rounded-xl bg-destructive/5 border border-destructive/15 text-destructive animate-fade-in">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span className="font-medium">{error}</span>
                <button
                  onClick={() => setError(null)}
                  className="ml-auto text-xs font-semibold underline underline-offset-2 hover:no-underline"
                >
                  Dismiss
                </button>
              </div>
            )}

            {/* ===== PHASE 1: ANALYZE ===== */}
            {phase === "ANALYZE" && (
              <div className="space-y-10 animate-fade-in">
                {/* Input Cards */}
                <div className="grid gap-6 lg:grid-cols-2">
                  {/* Upload */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2.5">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-primary text-xs font-bold">
                        1
                      </span>
                      <h3 className="text-sm font-semibold text-muted-foreground">
                        Your CV
                      </h3>
                    </div>
                    <UploadZone
                      apiBase={API_BASE}
                      onFileContent={(content) => {
                        setCvContent(content);
                      }}
                    />
                  </div>

                  {/* Job Description */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2.5">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-primary text-xs font-bold">
                        2
                      </span>
                      <h3 className="text-sm font-semibold text-muted-foreground">
                        Job Description
                      </h3>
                    </div>
                    <JobDescriptionInput
                      value={jobDescription}
                      onChange={setJobDescription}
                    />
                  </div>
                </div>

                {/* Action */}
                <div className="flex flex-col items-center gap-3">
                  <button
                    id="analyze-button"
                    onClick={handleAnalyze}
                    disabled={!canAnalyze || analyzing}
                    className={`btn-primary px-10 py-3.5 text-base font-bold ${
                      analyzing ? "animate-pulse-soft" : ""
                    }`}
                  >
                    <Sparkles
                      className={`h-4 w-4 ${analyzing ? "animate-spin" : ""}`}
                    />
                    {analyzing
                      ? "Analyzing your profile..."
                      : "Analyze Match"}
                  </button>
                  <p className="text-xs text-muted-foreground/60">
                    Powered by GPT-OSS 120B — takes about 15 seconds
                  </p>
                </div>

                {/* Results Preview */}
                {resultData && (
                  <section className="space-y-8 animate-fade-in-up">
                    <ResultsDashboard
                      visible={!!resultData}
                      data={resultData}
                      showScore={true}
                      showSkills={true}
                      showLearningPath={false}
                    />

                    <div className="grid gap-6 md:grid-cols-2">
                      <ExtractionCard
                        title="Your CV Summary"
                        type="cv"
                        data={resultData.extraction.cv_data}
                      />
                      <ExtractionCard
                        title="What They Want"
                        type="jd"
                        data={resultData.extraction.jd_data}
                      />
                    </div>

                    <div className="flex justify-center pt-4">
                      <button
                        id="go-to-refine"
                        onClick={() => setPhase("REFINE")}
                        className="group btn-primary px-8 py-3.5 text-base"
                      >
                        See Detailed Analysis & Refine CV
                        <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
                      </button>
                    </div>
                  </section>
                )}
              </div>
            )}

            {/* ===== PHASE 2: REFINE ===== */}
            {phase === "REFINE" && resultData && (
              <div className="space-y-10 animate-fade-in">
                {/* Career Advisor Insight */}
                <div className="space-y-6">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                      <Sparkles className="h-5 w-5" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold">Career Advisor Insight</h2>
                      <p className="text-sm text-muted-foreground">
                        Based on the data we have, we have improvements for your CV and also learning paths to strengthen you for the role.
                      </p>
                    </div>
                  </div>
                  
                  <ResultsDashboard
                    visible={!!resultData}
                    data={resultData}
                    showScore={false}
                    showSkills={false}
                    showLearningPath={true}
                  />
                </div>

                <div className="border-t border-border pt-10">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                      <PenLine className="h-5 w-5" />
                    </div>
                    <h2 className="text-lg font-bold">CV Optimization</h2>
                  </div>
                  <OptimizedCVEditor
                    initialCvData={resultData.extraction.cv_data}
                    suggestions={resultData.suggestions}
                  />
                </div>

                <div className="flex items-center justify-center gap-4 pt-4">
                  <button
                    onClick={() => setPhase("ANALYZE")}
                    className="btn-ghost text-sm"
                  >
                    ← Back to Analysis
                  </button>
                  <button
                    id="go-to-cover-letter"
                    onClick={() => setPhase("COVER_LETTER")}
                    className="group btn-primary px-8 py-3"
                  >
                    Write Cover Letter
                    <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
                  </button>
                </div>
              </div>
            )}

            {/* ===== PHASE 3: COVER LETTER ===== */}
            {phase === "COVER_LETTER" && (
              <div className="max-w-3xl mx-auto space-y-8 animate-fade-in">
                <div className="surface-elevated p-8 text-center">
                  <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/8 mx-auto mb-5">
                    <Zap className="h-7 w-7 text-primary" />
                  </div>
                  <h3 className="text-lg font-bold mb-2">AI Cover Letter</h3>
                  <p className="text-sm text-muted-foreground mb-6 max-w-md mx-auto leading-relaxed">
                    Generate a professional cover letter that highlights your
                    strengths and addresses the role&apos;s specific
                    requirements.
                  </p>

                  {!coverLetter ? (
                    <button
                      id="generate-cover-letter"
                      onClick={handleGenerateCoverLetter}
                      disabled={isGeneratingLetter}
                      className={`btn-primary px-8 py-3 ${
                        isGeneratingLetter ? "animate-pulse-soft" : ""
                      }`}
                    >
                      {isGeneratingLetter ? (
                        <>
                          <RotateCcw className="h-4 w-4 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Sparkles className="h-4 w-4" />
                          Generate Cover Letter
                        </>
                      )}
                    </button>
                  ) : (
                    <div className="space-y-4 text-left animate-fade-in-up">
                      <div className="relative">
                        <textarea
                          id="cover-letter-editor"
                          className="textarea-field min-h-[400px] bg-card border border-border"
                          value={coverLetter}
                          onChange={(e) => setCoverLetter(e.target.value)}
                        />
                        <span className="absolute top-3 right-3 badge badge-primary text-[10px]">
                          Editable
                        </span>
                      </div>
                      <div className="flex gap-3">
                        <button
                          onClick={handleGenerateCoverLetter}
                          disabled={isGeneratingLetter}
                          className="btn-secondary flex-1"
                        >
                          <RotateCcw className="h-3.5 w-3.5" />
                          Regenerate
                        </button>
                        <button
                          id="copy-cover-letter"
                          onClick={() => {
                            navigator.clipboard.writeText(coverLetter);
                          }}
                          className="btn-primary flex-1"
                        >
                          Copy to Clipboard
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-center gap-4">
                  <button
                    onClick={() => setPhase("REFINE")}
                    className="btn-ghost text-sm"
                  >
                    ← Back to Refine
                  </button>
                  <button
                    onClick={handleReset}
                    className="btn-ghost text-sm text-muted-foreground"
                  >
                    <RotateCcw className="h-3 w-3" />
                    Start Over
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}