import traceback

def check_torch_installation():
    try:
        import torch
        print(f"Torch version: {torch.__version__}")
        # Perform a simple operation to check if Torch is working
        tensor = torch.tensor([1.0, 2.0, 3.0])
        print(f"Tensor: {tensor}")
    except ImportError as e:
        print("Torch is not installed.")
        traceback.print_exc()
    except Exception as e:
        print("An error occurred while checking Torch installation.")
        traceback.print_exc()

def check_sentence_transformers_installation():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer and model loaded successfully.")
    except ImportError as e:
        print("SentenceTransformers is not installed.")
        traceback.print_exc()
    except Exception as e:
        print("An error occurred while checking SentenceTransformers installation.")
        traceback.print_exc()

def check_faiss_installation():
    try:
        import faiss
        print("FAISS imported successfully.")
        # Perform a simple operation to check if FAISS is working
        index = faiss.IndexFlatL2(10)  # 10-dimensional vectors
        print(f"FAISS index: {index}")
    except ImportError as e:
        print("FAISS is not installed.")
        traceback.print_exc()
    except Exception as e:
        print("An error occurred while checking FAISS installation.")
        traceback.print_exc()

def main():
    print("Checking Torch installation...")
    check_torch_installation()

    print("\nChecking SentenceTransformers installation...")
    check_sentence_transformers_installation()

    print("\nChecking FAISS installation...")
    check_faiss_installation()

if __name__ == "__main__":
    main()
