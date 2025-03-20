from fastapi import FastAPI, HTTPException,UploadFile, File,Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import markdown
from pydantic import BaseModel
from attrition.predictor import EmployeeData, PredictionResponse, predict_attrition
from smart_match.predict import match_jobs_to_applicant, df
from datetime import datetime
from nsp_retention.nsp_analyzer import NSPAnalyzer, NSPVisualizer, generate_recommendations, generate_report
from nsp_retention.nsp_models import (    
    RecommendationRequest, RecommendationResponse, ReportResponse,
    AnalysisResponse, )
# Initialize FastAPI app
app = FastAPI(
    title="RGT API Project",
    description="API for all the protals",
    version="1.0.0"
)

# Set Groq API key - in production, use environment variables
GROQ_API_KEY = "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"

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobRequest(BaseModel):
    profile: str
    applied_position: str  # New field for the applied job position


# Root endpoint
@app.get("/")
def read_root():
    return {"message": "RGT API Project"}   
# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy",  "timestamp": datetime.now().isoformat()}
# Prediction endpoint

@app.post("/predict", response_model=PredictionResponse)
def predict(employee: EmployeeData):
    try:
        # Get prediction
        prediction = predict_attrition(employee)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# @app.post("/match-job")
# def match_job(request: JobRequest):
#     try:
#         best_job = match_jobs_to_applicant(
#             request.profile, request.applied_position, df)
#         return best_job
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/match-job")
def match_job(request: JobRequest):
    try:
        best_job = match_jobs_to_applicant(
            request.profile, request.applied_position, df)
        return best_job
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/analyze", response_model=AnalysisResponse)         
async def analyze_nsp_data(file: UploadFile = File(..., description="Excel file containing NSP data")):
    try:
        # Read the uploaded Excel file
        content = await file.read()
        
        # Load data into pandas DataFrame
        try:
            df = pd.read_excel(
                io.BytesIO(content),
                dtype={'Phone number': str}  # Ensure Phone number is read as string
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Excel file: {str(e)}")
        
        # Create analyzer
        analyzer = NSPAnalyzer(df)
        
        # Analyze data
        subject_outcomes = analyzer.analyze_hiring_success()
        
        # Get overall stats
        overall_stats = analyzer.get_overall_stats()
        
        # Create visualizer
        visualizer = NSPVisualizer(analyzer.df)
        
        # Generate visualizations
        success_rates_chart = visualizer.visualize_subject_success_rates()
        retention_chart = visualizer.visualize_retention_comparison()
        
        # Generate recommendations
        recommendations = generate_recommendations(subject_outcomes, GROQ_API_KEY)
        
        # Generate report
        report_markdown = generate_report(subject_outcomes, recommendations)
        
        # Convert subject_outcomes DataFrame to list of dicts
        subject_outcomes_list = subject_outcomes.to_dict(orient='records')
        
        return AnalysisResponse(
            subject_outcomes=subject_outcomes_list,
            overall_stats=overall_stats,
            success_rates_chart=success_rates_chart,
            retention_chart=retention_chart,
            recommendations=recommendations,
            report_markdown=report_markdown
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations",response_model=RecommendationResponse) 
async def get_recommendations(request: RecommendationRequest = Body(..., 
                                                                  description="Subject metrics data",
                                                                  example={
                                                                      "subject_data": [
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
                                                                      ],
                                                                      "top_n": 3
                                                                  })):
    try:
        # Convert to DataFrame
        subject_data = pd.DataFrame(request.subject_data)
        
        # Generate recommendations
        recommendations = generate_recommendations(subject_data, GROQ_API_KEY, request.top_n)
        
        return RecommendationResponse(recommendations=recommendations)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to generate full report
@app.post("/report",response_model=ReportResponse)
async def generate_full_report(file: UploadFile = File(..., description="Excel file containing NSP data")):
    try:
        # Read the uploaded Excel file
        content = await file.read()
        
        # Load data into pandas DataFrame
        df = pd.read_excel(
            io.BytesIO(content),
            dtype={'Phone number': str}
        )
        
        # Create analyzer
        analyzer = NSPAnalyzer(df)
        
        # Analyze data
        subject_outcomes = analyzer.analyze_hiring_success()
        
        # Generate recommendations
        recommendations = generate_recommendations(subject_outcomes, GROQ_API_KEY)
        
        # Generate report in markdown
        report_markdown = generate_report(subject_outcomes, recommendations)
        
        # Convert markdown to HTML
        report_html = markdown.markdown(report_markdown)
        
        return ReportResponse(
            report_markdown=report_markdown,
            report_html=report_html
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)