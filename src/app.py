"""
app.py — Top-down entrypoint for the Adaptive RAG project.

Run:
    python src/app.py --q "What is the refund policy?"
    python src/app.py --q "latest NVIDIA earnings?" --config configs.yaml
    python src/app.py --q "..." --rebuild    # force re-ingest/index

What this file does:
1) Parse CLI arguments
2) Load config
3) Ensure a local vector store exists (build if missing / --rebuild)
4) Prepare dependencies: retriever, LLMs, web search tool, prompts
5) Build the LangGraph workflow
6) Execute one question end-to-end and print the final answer + route
"""

from __future__ import annotations

# --- Standard library imports ---
import argparse
import sys
from typing import Any, Dict

# --- Project imports (you will implement these modules) ---
from config import load_config  # -> dict
from ingest import load_local_docs, split_docs, build_vectorstore
from vectorstore import get_retriever, load_vectorstore, save_vectorstore
from llm import get_llm, get_llm_json  # -> BaseChatModel (normal + JSON/structured)
import prompts  # -> prompt strings (router/doc grader/hallucination/answer)
from web_search import get_web_search_tool  # -> Callable web search tool
from build_graph import build_workflow  # -> Compiled LangGraph graph

