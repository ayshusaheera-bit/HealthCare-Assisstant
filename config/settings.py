import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY is not set in environment or .env file.")
