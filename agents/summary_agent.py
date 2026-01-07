"""
Summary Agent
Uses Google's Gemini API to generate concise medical summaries from transcribed text.
"""

import google.generativeai as genai
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import GEMINI_API_KEY

class SummaryAgent:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing. Please set it in .env or environment variables.")
        
        genai.configure(api_key=GEMINI_API_KEY)
        # Trying gemini-flash-latest as a last resort for quota issues
        self.model = genai.GenerativeModel('gemini-flash-latest')

    def generate_summary(self, text):
        """
        Generates a concise clinical summary from the provided text.
        """
        if not text:
            return "No text provided for summary."

        prompt = f"""
        You are a healthcare assistant.
        
        Create a clear and structured medical summary from the transcription below.
        
        Include:
        - Symptoms
        - Duration
        - Severity
        - Advice / Medicines (if mentioned)
        
        Use simple English.
        Do not add new information.
        
        Text: "{text}"
        
        Clinical Note:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {e}"

if __name__ == "__main__":
    # Test block
    agent = SummaryAgent()
    sample_text = "My name is Aysha Sahira and I have had a severe headache since yesterday. I also feel a bit feverish."
    print("Generating summary...")
    print(agent.generate_summary(sample_text))
