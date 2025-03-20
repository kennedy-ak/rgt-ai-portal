from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class StageConfidence(BaseModel):
    stage: str
    confidence: float


class ApplicantData(BaseModel):
    position: str = Field(..., description="Position applying for")
    seniority_level: str = Field(..., description="Seniority Level (Entry/Mid/Senior)")
    current_title: Optional[str] = Field(None, description="Current job title")
    current_company: Optional[str] = Field(None, description="Current company")
    total_years_in_tech: Optional[float] = Field(None, description="Total years in tech industry")
    highest_degree: Optional[str] = Field(None, description="Highest degree (e.g., BS, MS, PhD)")
    program: Optional[str] = Field(None, description="Degree Program/Major")
    school: Optional[str] = Field(None, description="School/University")
    graduation_year: Optional[int] = Field(None, description="Graduation year")
    technical_skills: Optional[str] = Field(None, description="Technical skills (comma-separated)")
    programming_languages: Optional[str] = Field(None, description="Programming languages (comma-separated)")
    tools_and_technologies: Optional[str] = Field(None, description="Tools & Technologies (comma-separated)")
    soft_skills: Optional[str] = Field(None, description="Soft skills (comma-separated)")
    industries: Optional[str] = Field(None, description="Industries experience (comma-separated)")
    certifications: Optional[str] = Field(None, description="Professional certifications")
    key_projects: Optional[str] = Field(None, description="Key projects (brief description)")
    recent_achievements: Optional[str] = Field(None, description="Recent achievements")

    # Fields extracted from CV
    name: Optional[str] = Field(None, description="Full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Location/Address")


class CVUploadRequest(BaseModel):
    position: str = Field(..., description="Position applying for")
    seniority_level: str = Field(..., description="Seniority Level (Entry/Mid/Senior)")
    additional_info: Optional[Dict[str, Any]] = Field(None, description="Additional applicant information")


class ApplicantPrediction(BaseModel):
    position: str
    score: float
    predicted_stage: str
    stage_confidences: List[StageConfidence]