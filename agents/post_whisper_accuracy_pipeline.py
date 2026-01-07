"""
PostWhisperAccuracyPipeline (Gemini Edition)
Cleans, translates, and normalizes Whisper transcriptions using Gemini 2.0 Flash.
"""

import google.generativeai as genai
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import GEMINI_API_KEY

class PostWhisperAccuracyPipeline:
    def __init__(self, known_entities=None):
        """
        known_entities example:
        {
            "name": "Aysha Saheera",
            "college": "KMCT Institute of Emerging Technology and Management"
        }
        """
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing from configuration.")
        
        genai.configure(api_key=GEMINI_API_KEY)
        # Using gemini-flash-latest for best quota compatibility
        self.model = genai.GenerativeModel('gemini-flash-latest')
        self.known_entities = known_entities or {}

    def process(self, whisper_text, detected_language):
        """
        Processes raw transcription text through Gemini for:
        - Translation to English (if not already)
        - Grammar and spelling correction
        - Medical term normalization
        - Proper noun protection (using known_entities)
        """
        if not whisper_text:
            return ""

        entities_context = "\n".join([f"- {k}: {v}" for k, v in self.known_entities.items()])
        
        prompt = f"""
        You are an expert medical transcriptionist and language expert.
        
        TASK:
        Clean and normalize the following medical transcription.
        
        RULES:
        1. Translate to English if the input is in another language.
        2. Fix all spelling and grammar errors.
        3. Normalize medical shorthand (e.g., "bp high" -> "blood pressure is high").
        4. PROTECT the following known entities:
        {entities_context}
         Ensure these specific names and details are spelled exactly as provided above.
        5. Return ONLY the cleaned English text. No explanations.

        INPUT TEXT:
        "{whisper_text}"

        CLEANED TEXT:
        """

        try:
            response = self.model.generate_content(prompt)
            # Basic cleaning of response in case models adds quotes
            cleaned_text = response.text.strip().replace('"', '')
            return cleaned_text
        except Exception as e:
            print(f"Warning: Gemini-based accuracy pipeline failed: {e}")
            # Fallback to raw text if API fails
            return whisper_text

if __name__ == "__main__":
    # Test
    pipeline = PostWhisperAccuracyPipeline(known_entities={"name": "Aysha Saheera"})
    test_text = "My name is aisha sahara and i have chest pain undu"
    print(f"Original: {test_text}")
    print(f"Processed: {pipeline.process(test_text, 'en')}")
