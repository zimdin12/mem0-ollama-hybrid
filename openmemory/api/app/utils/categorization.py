import json
import logging
import os
from typing import List

from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Use existing environment variables from .env
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.100.10:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b-instruct")

openai_client = OpenAI(
    base_url=f"{OLLAMA_BASE_URL}/v1",
    api_key="ollama"  # Ollama doesn't need a real API key
)


class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    try:
        # Modified prompt to ensure JSON output
        messages = [
            {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT + "\n\nRespond ONLY with valid JSON in this exact format: {\"categories\": [\"category1\", \"category2\"]}"},
            {"role": "user", "content": memory}
        ]

        # Use regular completion (Ollama doesn't support structured output)
        completion = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=1000
        )

        response_content = completion.choices[0].message.content
        
        # Parse JSON manually
        # Try to extract JSON if it's wrapped in markdown code blocks
        if "```json" in response_content:
            response_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            response_content = response_content.split("```")[1].split("```")[0].strip()
        
        # Parse the JSON
        parsed_data = json.loads(response_content)
        
        # Extract categories
        if isinstance(parsed_data, dict) and "categories" in parsed_data:
            categories = parsed_data["categories"]
        elif isinstance(parsed_data, list):
            categories = parsed_data
        else:
            logging.warning(f"Unexpected response format: {parsed_data}")
            return []
        
        return [cat.strip().lower() for cat in categories if cat]

    except json.JSONDecodeError as e:
        logging.error(f"[ERROR] Failed to parse JSON response: {e}")
        logging.debug(f"[DEBUG] Raw response: {response_content}")
        return []
    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories: {e}")
        try:
            logging.debug(f"[DEBUG] Raw response: {completion.choices[0].message.content}")
        except Exception as debug_e:
            logging.debug(f"[DEBUG] Could not extract raw response: {debug_e}")
        return []