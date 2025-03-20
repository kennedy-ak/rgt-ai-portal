

import pandas as pd
import ast
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz  # Faster alternative to fuzzywuzzy

# Load Sentence Transformer Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load precomputed job embeddings
df = pd.read_csv("smart_match/embeddings.csv")
df["embedding"] = df["embedding"].apply(ast.literal_eval)

# Function to generate embedding


def generate_embedding(text: str):
    return embedder.encode(text).tolist()

# Function to check if the job title is similar to the applied position


def is_similar(job_title: str, applied_position: str, threshold=80):
    return fuzz.partial_ratio(job_title.lower(), applied_position.lower()) >= threshold

# Function to match jobs (excluding applied position from results)


def match_jobs_to_applicant(profile: str, applied_position: str, jobs_df: pd.DataFrame):
    query_embedding = generate_embedding(profile)
    job_matches = []

    for _, row in jobs_df.iterrows():
        similarity = 1 - cosine(query_embedding, row["embedding"])
        job_matches.append((row["filename"], similarity))

    # Remove ".pdf" and filter out the applied position from job matches
    filtered_matches = [
        (row[0].replace(".pdf", ""), row[1])
        for row in job_matches
        if not is_similar(row[0].replace(".pdf", ""), applied_position)
    ]

    # If all jobs are filtered out, return a default message
    if not filtered_matches:
        return {"Job Title": "No suitable job found", "Match Percentage": 0}

    # Sort matches by similarity and get the best match
    best_match = sorted(filtered_matches, key=lambda x: x[1], reverse=True)[0]

    return {"Job Title": best_match[0], "Match Percentage": round(best_match[1] * 100, 2)}