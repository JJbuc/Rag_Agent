import os
import yaml

_DEFAULTS = {
    "paths": {
        "raw_dir": "data/raw",
        "index_dir": "data/index",
    },
    "chunking": {
        "chunk_size": 1000,
        "overlap": 200,
    },
    "retrieval": {
        "k_retriever": 6,
    },
    "models": {
        "embedding": "nomic-embed-text-v1.5",
        "llm": "llama3.1:8b-instruct-q8_0",
        "temperature": 0.0,
    },
    "web": {
        "web_search_k": 3,
    },
    "runtime": {
        "max_retries": 5,
        "force_rebuild": False,
    },
}


def load_config(path: str = "configs.yaml") -> dict:
    """
    Load config from YAML, merge it over defaults, ensure data dirs exist,
    and return a plain dict.
    """
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    def _merge(base: dict, override: dict) -> dict:
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                _merge(base[k], v)
            else:
                base[k] = v
        return base

    cfg = _merge(_DEFAULTS.copy(), data)

    raw_dir = os.path.abspath(os.path.expanduser(cfg["paths"]["raw_dir"]))
    index_dir = os.path.abspath(os.path.expanduser(cfg["paths"]["index_dir"]))
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    cfg["paths"]["raw_dir"] = raw_dir
    cfg["paths"]["index_dir"] = index_dir

    return cfg
