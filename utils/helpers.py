"""
Helper utility functions for the IDS system.
"""
import os
import json
import time
import logging
from functools import wraps
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def ensure_directories(dirs: list):
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def timer(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def save_json(data: Dict, path: str):
    """Save dictionary as JSON file."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Dict:
    """Load JSON file as dictionary."""
    with open(path, 'r') as f:
        return json.load(f)
