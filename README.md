
# Prisma AI Support

Prisma AI Support is a chatbot application built with Flask and integrated with OpenAI's GPT-4. It uses FAISS for semantic search to provide quick and relevant responses to user queries based on a `data.txt` file. This project demonstrates the integration of various technologies to create an intelligent assistant for customer support.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Features
- Interactive chatbot interface using Flask and vanilla JavaScript
- Semantic search using FAISS and SentenceTransformer model
- Integration with OpenAI's GPT-4 for generating contextual responses
- Easy configuration through `config.json`
- Efficiently processes and indexes text data from `data.txt`

## Prerequisites
- Python 3.11 or higher
- An OpenAI API key [Get it here](https://beta.openai.com/signup/)

## Installation

### Clone the Repository
```bash
git clone https://github.com/your-username/prisma-ai-support.git
cd prisma-ai-support
```

### Install Dependencies
Make sure you have Poetry installed, then run:
```bash
poetry install
```

### Set Up the Environment Variables
Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Preparing the Data
Before running the application, ensure that the FAISS index is up-to-date with the current data:
```bash
poetry run python prepare_data.py
```

### Running the Application
Start the Flask application:
```bash
poetry run python app.py
```
Access the application at `http://127.0.0.1:5000`.

### Chatting with the Bot
Use the web interface to interact with the chatbot. Enter your questions in the input field and receive AI-generated responses based on the data and model context.

## Scripts

- **prepare_data.py**: Prepares the data by creating embeddings and building the FAISS index.
- **check_installation.py**: Checks the installation and version of PyTorch, SentenceTransformers, and FAISS.

## Configuration
Edit the `config.json` file to configure the file paths and model settings:
```json
{
    "file_path": "data.txt",
    "model": "gpt-4"
}
```

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

### Steps to Contribute
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
