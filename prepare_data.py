import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import sys
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path):
    """Load data from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().splitlines()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    except IOError as e:
        logging.error(f"Error reading file {file_path}: {e}")
        sys.exit(1)


def create_embeddings(data, model_name='all-MiniLM-L6-v2'):
    """Create embeddings from data using SentenceTransformer."""
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(data, convert_to_tensor=True)
        return embeddings.cpu().numpy()
    except ImportError as e:
        logging.error(f"Error importing libraries: {e}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error creating embeddings: {e}")
        traceback.print_exc()
        sys.exit(1)


def save_faiss_index(embeddings, index_path='faiss_index'):
    """Save embeddings to a FAISS index."""
    try:
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings are empty or None.")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, index_path)
        logging.info("FAISS index created and saved.")
    except ValueError as ve:
        logging.error(f"Value error: {ve}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error saving FAISS index: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    data_file_path = 'data.txt'
    index_file_path = 'faiss_index'

    logging.info("Loading data...")
    data = load_data(data_file_path)

    if not data:
        logging.error("No data loaded. Exiting.")
        sys.exit(1)

    logging.info("Creating embeddings...")
    embeddings = create_embeddings(data)

    if embeddings is None or len(embeddings) == 0:
        logging.error("Failed to create embeddings. Exiting.")
        sys.exit(1)

    logging.info("Saving FAISS index...")
    save_faiss_index(embeddings, index_file_path)
