
import google.generativeai as genai
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import GEMINI_API_KEY

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found.")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

with open("models_list.txt", "w") as f:
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                f.write(f"{m.name}\n")
    except Exception as e:
        f.write(f"Error listing models: {str(e)}\n")
print("Models written to models_list.txt")
