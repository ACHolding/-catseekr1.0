#!/usr/bin/env python3
"""
Cat R1 — single-file local assistant (stdlib + tkinter).

- files = off (no external checkpoints, no network APIs, no virtual file store)
- Cat R1 dialogue engine (local 1-bit ternary bootstrap stack)
- chat, code interpreter, canvas, document editor, terminal, memory
"""

from __future__ import annotations

import faulthandler
import io
import json
import math
import os
import random
import re
import statistics
import sys
import textwrap
import threading
import time
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime

faulthandler.enable()
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

BRAND = "Cat R1"
APP_NAME = BRAND
WINDOW_TITLE = BRAND
BOT_NAME = BRAND
MODEL_NAME = BRAND
FILES_ENABLED = False  # no external model checkpoints
VIRTUAL_FILES_ENABLED = False  # files = off (no in-memory attach store)
PYTHON_TARGET = "3.14"

CAT_R1_SYSTEM = (
    f"You are {BRAND}, a helpful local assistant. "
    "You run entirely on-device with a 1-bit ternary bootstrap stack. "
    "Be clear, structured, and conversational. Use short markdown when it helps. "
    "Files are off — no uploads, no external APIs. Offer code in fenced blocks when relevant."
)

# Tool modes
MODE_CHAT = "chat"
MODE_CODE = "code_interpreter"
MODE_CANVAS = "canvas"
MODE_ANALYSIS = "analysis"


def _text_insert_safe(s: str, *, code_fence: bool = False) -> str:
    if not isinstance(s, str):
#
