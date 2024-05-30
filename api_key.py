import os
import logging

def get_api_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logging.error("OpenAI API key not found. Please set the environment variable 'OPENAI_API_KEY'.")
        return None
    return api_key
