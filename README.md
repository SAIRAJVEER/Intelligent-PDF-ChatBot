# 🤖 Intelligent PDF ChatBot

An AI-powered PDF Question Answering application built using **Retrieval-Augmented Generation (RAG)**. Upload your PDF documents, ask questions, and receive intelligent, context-aware answers based on the uploaded content.

The application integrates multiple Large Language Models (LLMs), vector databases, and evaluation frameworks to provide a flexible and reliable document-based AI assistant.

## 🚀 Features

- **PDF Upload:** Upload PDF documents and extract their content.
- **RAG Pipeline:** Retrieve relevant document chunks to generate context-aware answers.
- **Multiple LLM Support:** Integration with OpenAI (ChatGPT), Google Gemini, Grok, Mistral, and Hugging Face models.
- **Semantic Search:** Retrieve relevant information using vector embeddings.
- **Vector Database:** Store and search document embeddings using ChromaDB.
- **LLM Evaluation:** Evaluate answer quality using RAGAS.
- **LLM Observability:** Track and monitor LLM workflows using LangSmith.
- **Interactive UI:** User-friendly interface built with Streamlit.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | User interface |
| LangChain | LLM and RAG orchestration |
| ChromaDB | Vector database |
| Embeddings | Semantic document retrieval |
| OpenAI / Gemini / Grok / Mistral | LLM integrations |
| Hugging Face | Open-source model integration |
| RAGAS | RAG evaluation |
| LangSmith | LLM tracing and monitoring |

## 🧠 How It Works

The application follows a Retrieval-Augmented Generation workflow:

1. **Document Upload:** Upload a PDF document.
2. **Text Extraction:** Extract text from the uploaded PDF.
3. **Text Chunking:** Split extracted text into smaller chunks.
4. **Embedding Generation:** Convert text chunks into vector embeddings.
5. **Vector Storage:** Store embeddings in ChromaDB.
6. **Retrieval:** Retrieve relevant document chunks based on the user's question.
7. **Answer Generation:** Pass retrieved context to the selected LLM.
8. **Evaluation:** Evaluate generated responses using RAGAS and monitor execution through LangSmith.

## 📂 Project Structure

```text
Intelligent-PDF-ChatBot/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
│
└── vectorstore/
    └── ChromaDB data
```

*Note: Adjust the structure above to match your actual project files.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SAIRAJVEER/Intelligent-PDF-ChatBot.git
```

### 2. Navigate to the Project Directory

```bash
cd Intelligent-PDF-ChatBot
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the root directory and add the API keys for the providers you have configured.

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
MISTRAL_API_KEY=your_mistral_api_key
XAI_API_KEY=your_grok_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token

LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
```

Only configure the keys required by your selected integrations. Never upload your actual API keys or `.env` file to GitHub.

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Evaluation and Monitoring

This project incorporates two important components:

**RAGAS**
- Faithfulness
- Answer Relevancy
- Retrieval and generation quality evaluation

**LangSmith**
- LLM execution tracing
- Monitoring application workflows
- Debugging and performance analysis

## 🎯 Project Objectives

- Understand and implement Retrieval-Augmented Generation.
- Build practical applications using multiple LLM providers.
- Implement semantic search and vector databases.
- Evaluate RAG performance and answer quality.
- Develop and deploy interactive GenAI applications.
## 📸 Application Preview

![PDF RAG Assistant](screenshots/image1.png)
(screenshots/image2.png)
(screenshots/image3.png)
## 👨‍💻 Author

**Sai Pratap Rajveer**

- GitHub: [SAIRAJVEER](https://github.com/SAIRAJVEER)

---

⭐ If you find this project interesting, consider giving the repository a star!
