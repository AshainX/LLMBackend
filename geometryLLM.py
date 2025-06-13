import time  # <-- Add this line
import requests
import re
#from huggingface_hub.hf_api import HfFolder

from transformers import pipeline
from huggingface_hub import login
HF_TOKEN = "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"
login(token=HF_TOKEN)
# Save your HF token only once (if not already)
# HfFolder.save_token("*your_token_here*")  # Uncomment if not done before

# API config
# API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-128k-instruct"
# headers = {"Authorization": f"Bearer {HF_TOKEN}"}
# import requests
# import re

class GeometrySolver:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-128k-instruct"
        self.headers = {"Authorization": "Bearer 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"}

    def clean_question(self, text):
        """
        Remove OCR artifacts and keep only the question
        """


        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 2]
        return ' '.join(' '.join(lines).split())


    def extract_answer(self, response):
        """
        Simple answer extraction
        """

        numbers = re.findall(r'\d+', response)
        return numbers[0] if numbers else "No answer found"

    def solve(self, question_text):
        try:
            question = self.clean_question(question_text)
            if not question:
                return "Could not extract question"
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": f"Question: {question}\nAnswer:"},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()[0]['generated_text'].split("Answer:")[-1].strip()
            return "API unavailable"
        except Exception:
            return "Could not solve problem"

solver = GeometrySolver()