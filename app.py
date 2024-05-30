from flask import Flask, request, render_template, jsonify, session
import os
import logging
import json
import faiss
import numpy as np
from openai import OpenAI, APIConnectionError, RateLimitError, APIError
from time import sleep
from retrying import retry
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
CONFIG_FILE = 'config.json'

# --- Data and Model ---
DATA_FILE = 'data.txt'
FAISS_INDEX_FILE = 'faiss_index'
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Global Variables ---
index = None  # FAISS index
model = None  # SentenceTransformer model
data = None  # Loaded data from file


def load_config(config_file=CONFIG_FILE):
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
        required_keys = ['file_path', 'model']
        for key in required_keys:
            if key not in config:
                logging.error(f"Missing required configuration key: {key}")
                return None
        return config
    except FileNotFoundError:
        logging.error("Configuration file not found.")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON configuration file: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while loading configuration: {e}")
        return None


def get_api_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logging.error("""
        OpenAI API key not found.

        Please set the environment variable 'OPENAI_API_KEY' with your key.
        """)
        return None
    return api_key


def retry_on_exception(exception):
    return isinstance(exception, (APIConnectionError, APIError))


@retry(retry_on_exception=retry_on_exception,
       wait_fixed=2000,
       stop_max_attempt_number=3)
def create_chat_completion(client, messages, model):
    try:
        response = client.chat.completions.create(model=model,
                                                  messages=messages)
        token_usage = response.usage
        logging.info(
            f"Tokens used: Prompt - {token_usage.prompt_tokens}, Completion - {token_usage.completion_tokens}, Total - {token_usage.total_tokens}"
        )
        return response.choices[0].message.content
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
        return f"ERROR: An unexpected error occurred: {e}"


def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logging.error("The file was not found. Please check the file path.")
        return "ERROR: The file was not found. Please check the file path."
    except IOError as e:
        logging.error(f"An error occurred while reading the file: {e}")
        return f"ERROR: An error occurred while reading the file: {e}"


def initialize_model_and_index():
    global index, model, data
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = f.read().splitlines()
        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(data)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_FILE)
        logging.info(f"Data loaded: {data}")
    except Exception as e:
        logging.error(
            f"Error initializing FAISS index or SentenceTransformer model: {e}"
        )
        raise


def semantic_search(query, top_k=5):
    try:
        logging.info(f"Performing semantic search for query: {query}")
        query_embedding = model.encode([query])
        distances, indices = index.search(query_embedding, top_k)
        search_results = [data[idx] for idx in indices[0]]
        logging.info(f"Search results indices: {indices[0]}")
        logging.info(f"Search results data: {search_results}")
        return search_results
    except Exception as e:
        logging.error(f"Error during semantic search: {e}")
        return []


def handle_user_question(question):
    config = load_config()
    if not config:
        return {"error": "Configuration error."}

    api_key = get_api_key()
    if not api_key:
        return {"error": "API key error."}

    client = OpenAI(api_key=api_key)

    if 'conversation' not in session:
        session['conversation'] = []

    conversation = session['conversation']
    conversation.append({"role": "user", "content": question})

    relevant_context = semantic_search(question)
    context_str = "\n".join(relevant_context)
    if not context_str:
        context_str = "I couldn't find relevant information in the salon's data."

    logging.info(f"Context passed to the assistant: {context_str}")

    messages = [{
        "role": "system",
        "content": f"You are a customer support assistant for Prisma Salon. Here is the information about the salon:\n{context_str}\nPlease provide help based on customer questions."
    }] + conversation

    prompt_limit = config.get("prompt_limit", 100)  # Default to maximum allowed by model
    answer_limit = config.get("answer_limit", 200)

    total_tokens = sum([len(msg['content']) for msg in messages]) + answer_limit
    logging.info(f"Total estimated tokens: {total_tokens}")

    if total_tokens > prompt_limit:
        logging.warning("Prompt too long, truncating context.")
        context_str = context_str[:prompt_limit - len(question) - answer_limit - 100]  # 100 is a buffer
        messages[0]['content'] = f"You are a customer support assistant for Prisma Salon. Here is the information about the salon:\n{context_str}\nPlease provide help based on customer questions."

    try:
        assistant_response = create_chat_completion(client, messages, config.get('model'))
    except InvalidRequestError as e:
        if "maximum context length" in str(e):
            logging.error("Prompt still too long after truncation. Please shorten the context.")
            assistant_response = "ERROR: Your request was too long. Please try rephrasing or simplifying your question."
        else:
            logging.error(f"InvalidRequestError: {e}")
            assistant_response = f"ERROR: An unexpected error occurred: {e}"

    conversation.append({"role": "assistant", "content": assistant_response})
    session['conversation'] = conversation
    return {"answer": assistant_response, "tokens_used": total_tokens}



@app.route('/')
def index():
    # Initialize the conversation history
    if 'conversation' not in session:
        session['conversation'] = []

    return render_template('index.html', conversation=session['conversation'])


@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form['question']
    response = handle_user_question(user_question)
    return jsonify(response)


if __name__ == '__main__':
    config = load_config()
    if not config:
        exit(1)

    api_key = get_api_key()
    if not api_key:
        exit(1)

    initialize_model_and_index()  # Initialize index and model
    app.run(debug=True)
