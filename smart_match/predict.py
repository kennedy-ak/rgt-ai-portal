import pandas as pd
import ast
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz  # Faster alternative to fuzzywuzzy

# Load Sentence Transformer Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load precomputed job embeddings
df = pd.read_csv("embeddings.csv")
df["embedding"] = df["embedding"].apply(ast.literal_eval)


def generate_embedding(text: str) -> list:
    """Generate a semantic embedding vector for the input text using the Sentence Transformer model.

    Args:
        text (str): The input text to be embedded (e.g., job description or applicant profile)

    Returns:
        list: A list of floats representing the text embedding vector
    """
    return embedder.encode(text).tolist()


def is_similar(job_title: str, applied_position: str, threshold: int = 80) -> bool:
    """Check if two job titles are similar using fuzzy string matching.

    Compares the partial ratio between lowercased versions of the strings.

    Args:
        job_title (str): The job title from the database
        applied_position (str): The position the applicant has applied for
        threshold (int, optional): Minimum similarity score (0-100) to consider titles similar. 
                                  Defaults to 80.

    Returns:
        bool: True if the similarity score meets or exceeds the threshold, False otherwise
    """
    return fuzz.partial_ratio(job_title.lower(), applied_position.lower()) >= threshold


def match_jobs_to_applicant(profile: str, applied_position: str, jobs_df: pd.DataFrame = df) -> dict:
    """Match an applicant's profile to the most suitable job, excluding their applied position.

    The matching is based on cosine similarity between semantic embeddings of:
    1. The applicant's profile text
    2. Precomputed job description embeddings

    Args:
        profile (str): The applicant's profile or resume text
        applied_position (str): The job title the applicant has applied for (to be excluded)
        jobs_df (pd.DataFrame, optional): DataFrame containing job embeddings with columns:
                                          - 'filename': Job title (with .pdf extension)
                                          - 'embedding': Precomputed embedding vectors
                                          Defaults to the preloaded DataFrame.

    Returns:
        dict: A dictionary containing:
              - 'Job Title': The best matching job title (without .pdf extension)
              - 'Match Percentage': The similarity percentage (0-100)
              If no matches found, returns a default "No suitable job found" message with 0% match.
    """
    # Generate embedding for the applicant's profile
    query_embedding = generate_embedding(profile)

    # Calculate similarity scores for all jobs
    job_matches = []
    for _, row in jobs_df.iterrows():
        similarity = 1 - cosine(query_embedding, row["embedding"])
        job_matches.append((row["filename"], similarity))

    # Filter out jobs similar to the applied position and remove .pdf extension
    filtered_matches = [
        (row[0].replace(".pdf", ""), row[1])
        for row in job_matches
        if not is_similar(row[0].replace(".pdf", ""), applied_position)
    ]

    # Handle case where all jobs are filtered out
    if not filtered_matches:
        return {"Job Title": "No suitable job found", "Match Percentage": 0}

    # Sort matches by similarity and get the best match
    best_match = sorted(filtered_matches, key=lambda x: x[1], reverse=True)[0]

    # Format the result with rounded percentage
    return {
        "Job Title": best_match[0],
        "Match Percentage": round(best_match[1] * 100, 2)
    }
