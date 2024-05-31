import logging
from openai import OpenAI, APIConnectionError, RateLimitError, APIError
from time import sleep
from retrying import retry

def retry_on_exception(exception):
    return isinstance(exception, (APIConnectionError, APIError))

@retry(retry_on_exception=retry_on_exception, wait_fixed=2000, stop_max_attempt_number=3)
def create_chat_completion(client, messages, model):
    try:
        response = client.chat.completions.create(model=model, messages=messages)
        token_usage = response.usage
        logging.info(f"Tokens used: Prompt - {token_usage.prompt_tokens}, Completion - {token_usage.completion_tokens}, Total - {token_usage.total_tokens}")
        return response.choices[0].message.content, token_usage.total_tokens
    except RateLimitError:
        logging.warning("Rate limit exceeded. Retrying after a short delay.")
        sleep(60)
        raise
    except APIConnectionError as e:
        logging.error(f"Connection error: {e}")
        raise
    except APIError as e:
        logging.error(f"API error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return f"ERROR: An unexpected error occurred: {e}", 0
