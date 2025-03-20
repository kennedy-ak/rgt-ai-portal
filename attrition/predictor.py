import pickle
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel
# Model for input data validation
class EmployeeData(BaseModel):
    age: int
    region: str
    work_mode: str
    skills: List[str]
    department: str
    duration: float  # Duration in months (decimal value * 10)
# Model for prediction response
class PredictionResponse(BaseModel):
    attrition_probability: float
    risk_level: str
    assessment: str
# Load the model
def load_model():
    try:
        with open('attrition/best_tuned_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
# Preprocess the input data
def preprocess_data(employee: EmployeeData):
    # Convert skills list to comma-separated string
    skills_str = ",".join(employee.skills) if employee.skills else "None"
    # Create DataFrame from input data
    input_data = pd.DataFrame({
        'age': [employee.age],
        'region': [employee.region],
        'work_mode': [employee.work_mode],
        'skills': [skills_str],
        'department': [employee.department],
        'duration': [employee.duration]
    })
    return input_data
# Make prediction
def predict_attrition(employee: EmployeeData) -> PredictionResponse:
    # Load model
    model = load_model()
    if not model:
        raise Exception("Model could not be loaded")
    # Preprocess data
    input_data = preprocess_data(employee)
    # Get prediction probability
    probability = int( float(model.predict_proba(input_data)[0][1]) * 100)
    # Determine risk level and assessment
    if probability >= 75:
        risk_level = "High Risk"
        assessment = "This employee has a high probability of leaving the organization soon."
    elif 50 <= probability < 75:
        risk_level = "Medium Risk"
        assessment = "This employee has a moderate risk of attrition and should be monitored."
    else:
        risk_level = "Low Risk"
        assessment = "This employee has a low probability of leaving in the near future."
    # Return prediction results
    return PredictionResponse(
        attrition_probability=probability,
        risk_level=risk_level,
        assessment=assessment
    )