# MultiAgentRAG

MultiAgentRAG is a **cutting-edge, modular Retrieval-Augmented Generation (RAG) framework** for local LLMs. It enables you to build, visualize, and run advanced RAG pipelines with features like dynamic routing, document grading, hallucination detection, and web search fallback. This project is inspired by and extends the ideas from [LangGraph's Adaptive RAG tutorial](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_adaptive_rag_local.ipynb).

---

## Why MultiAgentRAG?

- **Showcases advanced RAG agent design**: Demonstrates how to combine retrieval, generation, grading, and web search in a single, adaptive workflow.
- **Plug-and-play for your own knowledge base**: Just drop your `.txt` files into `data/raw/` and the agent adapts—no code changes needed.
- **Dynamic and robust**: The agent intelligently routes queries, grades context, and avoids hallucinations, making it suitable for real-world, production-grade applications.
- **Clear, extensible codebase**: Designed for easy understanding and rapid prototyping in research.

---

## Features

- **Local RAG with LLMs:** Use your own local models for both retrieval and generation.
- **Dynamic Routing:** Automatically decides whether to answer from your document corpus or use web search, based on the actual content of your vectorstore.
- **Document Grading:** Grades the relevance of retrieved documents to the user query.
- **Hallucination Detection:** Checks if the generated answer is grounded in the retrieved documents.
- **Web Search Fallback:** Uses web search when local documents are insufficient.
- **Graph Visualization:** Visualize your agent workflow as a graph.
- **Easy Document Ingestion:** Just drop your `.txt` files into `data/raw/` to update your knowledge base.
- **Modular Codebase:** Easily extend or swap out components.

---

## Project Structure

```
MultiAgentRAG/
│
├── configs.yaml           # Configuration file
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── data/
│   └── raw/               # Place your .txt documents here
│
└── src/
    ├── app.py             # Main entry point
    ├── config.py          # Config loader
    ├── generate.py        # Generation logic
    ├── grade_doc.py       # Document grading
    ├── hall_detector.py   # Hallucination detection
    ├── llm.py             # LLM loading/utilities
    ├── prompts.py         # Prompt templates
    ├── router.py          # Routing logic
    ├── vectorstore.py     # Vectorstore and ingestion
    ├── web_search.py      # Web search integration
    └── AgentGraph.png     # Workflow graph visualization (auto-generated)
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/JJbuc/Rag_Agent
cd MultiAgentRAG
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Add Your Documents

Place your `.txt` files in the `data/raw/` directory.  
These will be automatically loaded and indexed for retrieval.

### 4. Configure Your Settings

Edit `configs.yaml` to set parameters such as:
- LLM model name and path
- Retriever settings (`k_retriever`, etc.)
- Runtime options (max retries, etc.)

### 5. Run the App

```bash
cd src
python app.py
```

You will be prompted to enter your question.  
The agent will decide whether to answer from your documents or use web search, and will display the final answer.

---

## How It Works

1. **Document Ingestion:** All `.txt` files in `data/raw/` are loaded and split into chunks for embedding and retrieval.
2. **Dynamic Routing:** When you ask a question, the agent uses an LLM-powered router to decide if your vectorstore likely contains the answer, based on the actual topics present in your documents.
3. **Retrieval & Grading:** If routed to the vectorstore, relevant documents are retrieved and graded for relevance.
4. **Generation:** The LLM generates an answer using the retrieved context.
5. **Hallucination Detection:** The answer is checked for grounding in the retrieved documents.
6. **Web Search Fallback:** If the answer is not grounded or documents are not relevant, the agent can use web search and retry.
7. **Visualization:** The workflow graph is saved as `src/AgentGraph.png`.

---

## Customization

- **Add More Document Types:** Extend `vectorstore.py` to support PDFs, Markdown, etc.
- **Change LLMs:** Edit `llm.py` and `configs.yaml` to use different local or remote models.
- **Modify Prompts:** Tweak prompt templates in `prompts.py` for your use case.
- **Tune Routing:** The router dynamically adapts to your corpus, but you can further customize its logic in `router.py`.

---

## Example: Why This Pipeline Outperforms Normal RAG

Suppose you have the same `.txt` documents in `data/raw/` and use the same local LLM for both a **normal RAG** pipeline and this **MultiAgentRAG** pipeline.

Below are some illustrative examples showing how MultiAgentRAG delivers more accurate, relevant, and robust answers.

---

### **Question 1:**  
**"What are the main types of agent memory in AI systems?"**

**Normal RAG Output:**  
> The main types of agent memory are short-term and long-term memory.

**MultiAgentRAG Output:**  
> The main types of agent memory in AI systems include episodic memory (for storing specific experiences), semantic memory (for general knowledge), and working memory (for temporary information processing).  
> *(Source: agent_article.txt)*

**Why is MultiAgentRAG better?**  
- It retrieves more relevant context, grades document relevance, and generates a more complete, nuanced answer grounded in your documents.

---

### **Question 2:**  
**"What are the latest models released for Llama 3.2?"**

**Normal RAG Output:**  
> Sorry, I could not find information about Llama 3.2 models in the documents.

**MultiAgentRAG Output:**  
> The latest models released for Llama 3.2 include Llama 3.2 11B Vision Instruct and Llama 3.2 90B Vision Instruct, available on Azure AI Model Catalog.  
> *(Sourced via web search fallback)*

**Why is MultiAgentRAG better?**  
- When the answer is not in your documents, it automatically falls back to web search, ensuring up-to-date and complete answers.

---

### **Question 3:**  
**"Explain adversarial attacks on LLMs."**

**Normal RAG Output:**  
> Adversarial attacks are attacks on AI models.

**MultiAgentRAG Output:**  
> Adversarial attacks on LLMs involve crafting inputs that intentionally cause the model to make mistakes or reveal sensitive information. These attacks can exploit weaknesses in the model's training data or prompt structure.  
> *(Source: agent_article3.txt)*

**Why is MultiAgentRAG better?**  
- The pipeline grades and filters context, ensuring the answer is specific, accurate, and grounded in your documents—avoiding vague or generic responses.

---

### **Question 4:**  
**"What is prompt engineering?"**

**Normal RAG Output:**  
> Prompt engineering is designing prompts for AI models.

**MultiAgentRAG Output:**  
> Prompt engineering is the process of designing and refining input prompts to guide LLMs toward producing desired outputs. Techniques include prompt templates, chaining, and context injection.  
> *(Source: agent_article2.txt)*

**Why is MultiAgentRAG better?**  
- The answer is more detailed and contextually rich, thanks to document grading and hallucination detection.

---
## Visualization

## Visualization

Below is a visualization of the agent workflow graph, illustrating how MultiAgentRAG routes queries, retrieves documents, grades context, detects hallucinations, and falls back to web search when needed:

![Agent Workflow Graph](C:/Users/"Jay Jani"/Downloads/langgraph.drawio.png)

This diagram demonstrates the adaptive and modular flow that sets MultiAgentRAG apart from standard RAG pipelines.

---

## Credits

This project is inspired by and builds upon the excellent [LangGraph Adaptive RAG tutorial](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_adaptive_rag_local.ipynb) by the LangChain team.

---
