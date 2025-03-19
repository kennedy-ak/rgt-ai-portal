from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum

class NSPData(BaseModel):
    """Model for NSP data in Excel format (base64 encoded)"""
    file_content: str = Field(
        ..., 
        description="Base64 encoded Excel file content",
        example="UEsDBBQAAAAIAGOJG1fM4mRaLwEAAD8EAAATAAAAeG..."
    )

class SubjectMetric(BaseModel):
    """Model for subject specialization metrics"""
    Subject: str = Field(
        ..., 
        description="Academic subject or program",
        example="Computer Science"
    )
    Total_Candidates: int = Field(
        ..., 
        description="Total number of NSPs with this subject",
        example=45,
        ge=0
    )
    Hired: int = Field(
        ..., 
        description="Number of hired candidates",
        example=30,
        ge=0
    )
    Not_Hired: int = Field(
        ..., 
        description="Number of candidates not hired",
        example=10,
        ge=0
    )
    Offered_Bootcamp: int = Field(
        ..., 
        description="Number of candidates offered bootcamp",
        example=5,
        ge=0
    )
    Hire_Rate: float = Field(
        ..., 
        description="Percentage of candidates hired",
        example=66.7,
        ge=0,
        le=100
    )

class OverallStats(BaseModel):
    """Model for overall statistics"""
    top_subject: str = Field(
        ..., 
        description="Subject with highest hire rate",
        example="Computer Science"
    )
    avg_rate: float = Field(
        ..., 
        description="Average hire rate across all subjects",
        example=58.4,
        ge=0,
        le=100
    )
    total_candidates: int = Field(
        ..., 
        description="Total number of candidates",
        example=150,
        ge=0
    )
 



class VisualizationResponse(BaseModel):
    """Model for visualization response"""
    success_rates_chart: str = Field(
        ..., 
        description="Base64 encoded image of success rates chart",
        example="iVBORw0KGgoAAAANSUhEUgAAA..."
    )
    retention_chart: str = Field(
        ..., 
        description="Base64 encoded image of retention comparison chart",
        example="iVBORw0KGgoAAAANSUhEUgAAA..."
    )

class RecommendationRequest(BaseModel):
    """Model for recommendation request"""
    subject_data: List[Dict[str, Any]] = Field(
        ..., 
        description="Subject specialization data",
        example=[
            {
                "Subject": "Computer Science",
                "Total Candidates": 50,
                "Hired": 35,
                "Not Hired": 10,
                "Offered Bootcamp": 5,
                "Hire Rate (%)": 70.0
            },
            {
                "Subject": "Information Technology",
                "Total Candidates": 40,
                "Hired": 25,
                "Not Hired": 10,
                "Offered Bootcamp": 5,
                "Hire Rate (%)": 62.5
            }
        ]
    )
    top_n: Optional[int] = Field(
        3, 
        description="Number of top subjects to consider",
        example=3,
        ge=1,
        le=10
    )

class RecommendationResponse(BaseModel):
    """Model for recommendation response"""
    recommendations: List[str] = Field(
        ..., 
        description="List of recommendations",
        example=[
            "Focus recruiting efforts on Computer Science graduates, who show a 12.3% higher retention rate than average.",
            "Consider expanding bootcamp offerings to Information Technology graduates to increase their hire rate.",
            "Implement mentorship programs for lower-performing subjects to improve their success rates."
        ]
    )

class ReportResponse(BaseModel):
    """Model for report response"""
    report_markdown: str = Field(
        ..., 
        description="Markdown formatted report",
        example="# NSP Hiring Success & Retention Analysis Report\n\n## Executive Summary..."
    )
    report_html: str = Field(
        ..., 
        description="HTML formatted report",
        example="<h1>NSP Hiring Success & Retention Analysis Report</h1><h2>Executive Summary...</h2>"
    )

class AnalysisResponse(BaseModel):
    """Model for complete analysis response"""
    subject_outcomes: List[Dict[str, Any]] = Field(
        ..., 
        description="Subject specialization outcomes",
        example=[
            {
                "Subject": "Computer Science",
                "Total Candidates": 50,
                "Hired": 35,
                "Not Hired": 10,
                "Offered Bootcamp": 5,
                "Hire Rate (%)": 70.0
            }
        ]
    )
    overall_stats: OverallStats = Field(
        ...,
        description="Overall statistics from the analysis"
    )
    success_rates_chart: str = Field(
        ..., 
        description="Base64 encoded image of success rates chart",
        example="iVBORw0KGgoAAAANSUhEUgAAA..."
    )
    retention_chart: str = Field(
        ..., 
        description="Base64 encoded image of retention comparison chart",
        example="iVBORw0KGgoAAAANSUhEUgAAA..."
    )
    recommendations: List[str] = Field(
        ..., 
        description="AI-generated recommendations",
        example=[
            "Focus recruiting efforts on Computer Science graduates",
            "Consider expanding bootcamp offerings for IT graduates"
        ]
    )
    report_markdown: str = Field(
        ..., 
        description="Markdown formatted report",
        example="# NSP Hiring Success & Retention Analysis Report..."
    )
    
class ErrorResponse(BaseModel):
    """Model for error response"""
    detail: str = Field(
        ..., 
        description="Error details",
        example="Invalid Excel file: No sheet named 'Sheet1'"
    )