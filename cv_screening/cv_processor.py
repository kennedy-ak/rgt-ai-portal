import os
import re
import json
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_file(file_path):
    """Extract text from PDF or DOCX using Langchain loaders and text splitter"""
    try:
        # Get appropriate loader
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(('.doc', '.docx')):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file format")

        # Load documents
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)

        # Combine all text
        return " ".join([doc.page_content for doc in docs])

    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        raise


def extract_cv_info(text):
    # Initialize Groq LLM
    llm = ChatGroq(
        api_key="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",
        model_name="deepseek-r1-distill-qwen-32b"
    )

    # Create a more comprehensive prompt template
    template = """
    You are an expert CV analyzer. Extract the following information from the CV below in a structured format.
    If any field is not found, indicate with "Not specified".

    CV TEXT:
    {cv_text}

    EXTRACT THE FOLLOWING INFORMATION:
    1. Full Name
    2. Email Address
    3. Phone Number
    4. Location/Address
    5. Current Title
    6. Current Company
    7. Total Years in Tech (estimate if not explicitly stated)
    8. Highest Degree
    9. Program/Major
    10. School/University
    11. Graduation Year
    12. Technical Skills (comma-separated)
    13. Programming Languages (comma-separated)
    14. Tools & Technologies (comma-separated)
    15. Soft Skills (comma-separated)
    16. Industries Experience (comma-separated)
    17. Certifications
    18. Key Projects
    19. Recent Achievements

    IMPORTANT: Your response must be a valid, parseable JSON object with the following format:
    {{
        "full_name": "String value",
        "email": "String value",
        "phone": "String value",
        "location": "String value",
        "current_title": "String value",
        "current_company": "String value",
        "total_years_in_tech": number or "Not specified",
        "highest_degree": "String value",
        "program": "String value",
        "school": "String value",
        "graduation_year": number or "Not specified",
        "technical_skills": "String value",
        "programming_languages": "String value",
        "tools_and_technologies": "String value",
        "soft_skills": "String value",
        "industries": "String value",
        "certifications": "String value",
        "key_projects": "String value",
        "recent_achievements": "String value"
    }}
    DO NOT include ANY explanatory text before or after the JSON object.
    Your entire response must be ONLY valid, parseable JSON, nothing else.
    """

    prompt = PromptTemplate(
        input_variables=["cv_text"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    result = None  # Initialize result variable

    try:
        # Get response from Groq
        result = chain.run(cv_text=text)

        # Try to extract JSON from the result if there's extra text
        json_match = re.search(r'({.*})', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            cv_data = json.loads(json_str)
        else:
            # If no JSON pattern found, try to parse the whole result
            cv_data = json.loads(result)

        # Return extracted information in the format expected by the rest of the application
        return {
            'name': cv_data.get('full_name', 'Not specified'),
            'email': cv_data.get('email', 'Not specified'),
            'phone': cv_data.get('phone', 'Not specified'),
            'location': cv_data.get('location', 'Not specified'),
            'current_title': cv_data.get('current_title', 'Not specified'),
            'current_company': cv_data.get('current_company', 'Not specified'),
            'total_years_in_tech': cv_data.get('total_years_in_tech', 'Not specified'),
            'highest_degree': cv_data.get('highest_degree', 'Not specified'),
            'program': cv_data.get('program', 'Not specified'),
            'school': cv_data.get('school', 'Not specified'),
            'graduation_year': cv_data.get('graduation_year', 'Not specified'),
            'technical_skills': cv_data.get('technical_skills', 'Not specified'),
            'programming_languages': cv_data.get('programming_languages', 'Not specified'),
            'tools_and_technologies': cv_data.get('tools_and_technologies', 'Not specified'),
            'soft_skills': cv_data.get('soft_skills', 'Not specified'),
            'industries': cv_data.get('industries', 'Not specified'),
            'certifications': cv_data.get('certifications', 'Not specified'),
            'key_projects': cv_data.get('key_projects', 'Not specified'),
            'recent_achievements': cv_data.get('recent_achievements', 'Not specified')
        }

    except Exception as e:
        print(f"Error extracting information: {str(e)}")
        if result is not None:
            print(f"Raw response: {result}")
        return {
            'name': None,
            'email': None,
            'phone': None,
            'location': None
        }


def process_cv(file_path):
    """Process the CV file and return extracted information"""
    try:
        text = extract_text_from_file(file_path)
        cv_info = extract_cv_info(text)
        return cv_info
    except Exception as e:
        print(f"Error processing CV: {str(e)}")
        return {"error": str(e)}


def create_cv_text(applicant_data):
    """Create CV text from applicant data"""
    # Convert pydantic model to dict if needed
    if not isinstance(applicant_data, dict):
        applicant_data = applicant_data.dict()

    text_columns = [
        'current_title', 'current_company', 'highest_degree', 'program', 'school',
        'industries', 'technical_skills', 'soft_skills', 'tools_and_technologies',
        'programming_languages', 'certifications', 'key_projects', 'recent_achievements'
    ]

    cv_text = ' '.join(str(applicant_data.get(col, '')) for col in text_columns
                       if col in applicant_data and applicant_data.get(col) not in ['', 'nan', None, 'Not specified'])
    return cv_text