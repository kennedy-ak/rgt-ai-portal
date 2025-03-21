import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from cv_screening.schemas import ApplicantData

# Global variables
tokenizer = None
bert_model = None
model = None
categorical_dim = None

# Stage mapping
stage_mapping = {
    0: "1st interview",
    1: "Technical interview",
    2: "Final interview",
    3: "Offer",
    4: "Hired"
}

def load_preprocessor():
    """Load the column transformer used during preprocessing"""
    try:
        # Fix path with proper os.path.join
        preprocessed_path = "cv_screening/preprocessed_data.npz"
        preprocessed = np.load(preprocessed_path, allow_pickle=True)
        # This approach assumes feature dimensions are available in the existing files
        categorical_dim = preprocessed['X_train_cat'].shape[1]
        return categorical_dim
    except Exception as e:
        print(f"Error loading preprocessor: {e}")
        # Return a default value if loading fails
        return 64  # Adjust this default value as needed


def get_bert_embeddings(texts):
    """Get BERT embeddings for a single text"""
    global tokenizer, bert_model
    
    # Tokenize
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Get embeddings
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Get the [CLS] token embeddings
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()
    
    return embeddings


def preprocess_applicant_data(applicant_data):
    """Preprocess a single applicant's data"""
    global categorical_dim

    # Convert pydantic model to dict if needed
    if not isinstance(applicant_data, dict):
        applicant_data = applicant_data.dict()

    # Create a DataFrame with the applicant data
    df = pd.DataFrame([applicant_data])

    # Rename columns to match the expected format
    column_mapping = {
        'current_title': 'Current Title',
        'current_company': 'Current Company',
        'highest_degree': 'Highest Degree',
        'program': 'Program',
        'school': 'School',
        'total_years_in_tech': 'Total Years in Tech',
        'graduation_year': 'Graduation Year',
        'seniority_level': 'Seniority Level',
        'technical_skills': 'Technical Skills',
        'programming_languages': 'Programming Languages',
        'tools_and_technologies': 'Tools & Technologies',
        'key_projects': 'Key Projects',
        'position': 'Position'
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Handle numerical features
    if 'Total Years in Tech' in df:
        df['Total Years in Tech'] = pd.to_numeric(df['Total Years in Tech'], errors='coerce')
    else:
        df['Total Years in Tech'] = np.nan

    if 'Graduation Year' in df:
        df['Graduation Year'] = pd.to_numeric(df['Graduation Year'], errors='coerce')
        df['Years Since Graduation'] = 2025 - df['Graduation Year']
    else:
        df['Years Since Graduation'] = np.nan

    # Fill missing values
    df['Total Years in Tech'].fillna(5, inplace=True)  # Use a reasonable default
    df['Years Since Graduation'].fillna(3, inplace=True)  # Use a reasonable default

    # Create experience level categories if not provided
    if 'Experience_Level' not in df:
        df['Experience_Level'] = pd.cut(
            df['Total Years in Tech'],
            bins=[0, 2, 5, 10, float('inf')],
            labels=['Entry', 'Junior', 'Mid', 'Senior']
        )

    # Create a feature vector with the correct dimensions
    feature_vector = np.zeros(categorical_dim, dtype=np.float32)

    return feature_vector


def predict_applicant_score(applicant_data, cv_text):
    """Predict applicant score and stage"""
    global model
    
    # Get BERT embeddings
    bert_embedding = get_bert_embeddings([cv_text])

    # Preprocess categorical features
    cat_features = preprocess_applicant_data(applicant_data)

    # Convert to tensors
    cat_tensor = torch.FloatTensor(cat_features).unsqueeze(0)  # Add batch dimension
    text_tensor = torch.FloatTensor(bert_embedding)

    # Make prediction
    with torch.no_grad():
        score_pred, stage_pred = model(cat_tensor, text_tensor)

        score = score_pred.item()
        _, predicted_stage = torch.max(stage_pred, 1)
        stage = stage_mapping[predicted_stage.item()]

        # Get confidence scores for each stage
        stage_probs = torch.nn.functional.softmax(stage_pred, dim=1)[0]
        stage_confidences = [
            {"stage": stage_mapping[i], "confidence": float(prob)}
            for i, prob in enumerate(stage_probs)
        ]

    return score, stage, stage_confidences


def initialize_models():
    """Initialize BERT and transformer models"""
    global tokenizer, bert_model, model, categorical_dim
    
    # Check if model file exists - use os.path.join for platform-independent paths
    model_path =  'cv_screening\cv_scoring_model.pt'
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file '{model_path}' not found. Please train the model first.")

    # Load preprocessor information
    categorical_dim = load_preprocessor()
    print(f"Loaded preprocessor information. Categorical dimension: {categorical_dim}")

    # Initialize BERT
    print("Initializing BERT model...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    bert_dim = 768  # Base BERT embedding dimension

    # Load the model - now from transformer_model instead of transformer
    from cv_screening.transformer import ApplicantTransformer
    print("Loading trained model...")
    model = ApplicantTransformer(categorical_dim=categorical_dim, bert_dim=bert_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Models initialized successfully!")