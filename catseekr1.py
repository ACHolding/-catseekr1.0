#!/usr/bin/env python3
"""
Single-file CAT R1 BitNet GUI - Edition R1.1.1A

Changes:
- Re-monikered to CAT R1.1.1A Gemini Flash architecture blend profile.
- files = off (no external model files, no runpy handoff, no network APIs)
- Python 3.14-friendly stdlib-only build
- GUI: ChatGPT-style layout (sidebar, tools, modes)
- Code interpreter (auto-run Python, charts), Canvas draw + Canvas doc editor
- Virtual file attach, user memory, multi-chat threads, regenerate/stop
- Conversational heuristics for natural replies (offline, no cloud API)
- Ternary BitLinear layers inside a causal transformer block structure.
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

WINDOW_TITLE = "AC HOLDINGS [C] 1999-2026 R1.1.1A Catseek"
BOT_NAME = "CAT R1.1.1A GEMINI-FLASH-BIT"
MODEL_NAME = "CAT R1.1.1A BitNet (Flash Config)"
FILES_ENABLED = False  # no external model checkpoints
VIRTUAL_FILES_ENABLED = True  # in-memory paste/upload text (ChatGPT attach)
PYTHON_TARGET = "3.14"

# ChatGPT-style tool modes
MODE_CHAT = "chat"
MODE_CODE = "code_interpreter"
MODE_CANVAS = "canvas"
MODE_ANALYSIS = "analysis"


def _text_insert_safe(s: str, *, code_fence: bool = False) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\x00", "").replace("&&", "; ")
    if code_fence:
        return s
    out: list[str] = []
    for ch in s:
        if ch == "[":
            out.append("\uFF3B")
        elif ch == "]":
            out.append("\uFF3D")
        elif ch == "$":
            out.append("\uFF04")
        elif ch == "{":
            out.append("(")
        elif ch == "}":
            out.append(")")
        elif ch == "\\":
            out.append("\uFF3C")
        else:
            out.append(ch)
    return "".join(out)


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    acc = 2166136261
    for ch in text.encode("utf-8", "replace"):
        acc ^= ch
        acc = (acc * 16777619) & 0xFFFFFFFF
    return acc


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    m = max(values)
    exps: list[float] = []
    total = 0.0
    for v in values:
        z = (v - m)
        if z < -60.0:
            e = 0.0
        elif z > 60.0:
            e = math.exp(60.0)
        else:
            e = math.exp(z)
        exps.append(e)
        total += e
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [e / total for e in exps]


def _silu(x: float) -> float:
    if x >= 40.0:
        return x
    if x <= -40.0:
        return 0.0
    return x / (1.0 + math.exp(-x))


def _dot(a: list[float], b: list[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def _count_repeats(s: str) -> int:
    best = 1
    cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def _clean_generated(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch in "\n\r\t" or (" " <= ch <= "~") or ch.isprintable():
            cleaned.append(ch)
    s = "".join(cleaned).replace("\r\n", "\n").replace("\r", "\n")
    for marker in ("\nUser:", "\nYOU:", "\n[SYSTEM]", "\n[YOU]", "\n[AHA]"):
        if marker in s:
            s = s.split(marker, 1)[0]
    s = s.strip()
    if "\n\n\n" in s:
        while "\n\n\n" in s:
            s = s.replace("\n\n\n", "\n\n")
    return s


def _is_low_quality(text: str) -> bool:
    s = text.strip()
    if len(s) < 16:
        return True
    if _count_repeats(s) >= 7:
        return True
    printable = sum(1 for ch in s if ch.isprintable() or ch in "\n\t")
    if printable / max(1, len(s)) < 0.95:
        return True
    ascii_like = sum(1 for ch in s if ch == "\n" or ch == "\t" or (32 <= ord(ch) < 127))
    if ascii_like / max(1, len(s)) < 0.90:
        return True
    if len(s) > 50 and s.count(" ") < 6:
        return True
    letters = sum(1 for ch in s if ch.isalpha())
    if len(s) > 24 and letters / max(1, len(s)) < 0.45:
        return True
    words = [w for w in s.split() if w]
    if len(s) > 20 and len(words) < 3:
        return True
    if s.count("\\") >= 2 or s.count("`") >= 2:
        return True
    noisy = sum(1 for ch in s if ch in "`\\^=<>|~")
    if noisy / max(1, len(s)) > 0.08:
        return True
    return False


class ByteTokenizer:
    bos_id = 256
    eos_id = 257
    vocab_size = 258

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False, limit: int | None = None) -> list[int]:
        data = list(text.encode("utf-8", "replace"))
        out: list[int] = []
        if add_bos:
            out.append(self.bos_id)
        out.extend(data)
        if add_eos:
            out.append(self.eos_id)
        if limit is not None and len(out) > limit:
            # Keep the start (system/instruction tokens) when context is trimmed.
            out = out[:limit]
        return out

    def decode(self, token_ids: list[int]) -> str:
        data = bytearray()
        for tok in token_ids:
            if 0 <= tok < 256:
                data.append(tok)
        return data.decode("utf-8", "replace")


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 258
    context_size: int = 64
    d_model: int = 20
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 40
    ternary_threshold: float = 0.28

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class BitLinear:
    def __init__(self, in_features: int, out_features: int, *, seed: int, threshold: float = 0.28, bias: bool = True) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.master: list[list[float]] = []
        self.pos_index: list[list[int]] = []
        self.neg_index: list[list[int]] = []
        self.row_scale: list[float] = []
        self.bias: list[float] = []
        rnd = random.Random(seed)
        for _ in range(out_features):
            row = [(rnd.random() * 2.0 - 1.0) for _ in range(in_features)]
            self.master.append(row)
            pos: list[int] = []
            neg: list[int] = []
            for idx, val in enumerate(row):
                if val > threshold:
                    pos.append(idx)
                elif val < -threshold:
                    neg.append(idx)
            nonzero = len(pos) + len(neg)
            self.pos_index.append(pos)
            self.neg_index.append(neg)
            self.row_scale.append(1.0 / math.sqrt(max(1, nonzero)))
            self.bias.append((rnd.random() - 0.5) * 0.02 if bias else 0.0)

    def nonzero_ratio(self) -> float:
        total = self.in_features * self.out_features
        nz = sum(len(p) + len(n) for p, n in zip(self.pos_index, self.neg_index))
        return nz / max(1, total)

    def forward_vec(self, x: list[float]) -> list[float]:
        out = [0.0] * self.out_features
        for row_idx in range(self.out_features):
            acc = self.bias[row_idx]
            for col_idx in self.pos_index[row_idx]:
                acc += x[col_idx]
            for col_idx in self.neg_index[row_idx]:
                acc -= x[col_idx]
            out[row_idx] = acc * self.row_scale[row_idx]
        return out

    def forward_seq(self, seq: list[list[float]]) -> list[list[float]]:
        return [self.forward_vec(x) for x in seq]


class RMSNorm:
    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        self.dim = dim
        self.eps = eps
        self.weight = [1.0] * dim

    def forward_vec(self, x: list[float]) -> list[float]:
        sq = 0.0
        for v in x:
            sq += v * v
        rms = math.sqrt((sq / max(1, self.dim)) + self.eps)
        inv = 1.0 / rms
        return [x[i] * inv * self.weight[i] for i in range(self.dim)]

    def forward_seq(self, seq: list[list[float]]) -> list[list[float]]:
        return [self.forward_vec(x) for x in seq]


class BitSelfAttention:
    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        dim = cfg.d_model
        thr = cfg.ternary_threshold
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.score_scale = 1.0 / math.sqrt(max(1, self.head_dim))
        self.q_proj = BitLinear(dim, dim, seed=seed + 11, threshold=thr, bias=False)
        self.k_proj = BitLinear(dim, dim, seed=seed + 23, threshold=thr, bias=False)
        self.v_proj = BitLinear(dim, dim, seed=seed + 37, threshold=thr, bias=False)
        self.o_proj = BitLinear(dim, dim, seed=seed + 53, threshold=thr, bias=False)

    def forward(self, seq: list[list[float]]) -> list[list[float]]:
        q_all = self.q_proj.forward_seq(seq)
        k_all = self.k_proj.forward_seq(seq)
        v_all = self.v_proj.forward_seq(seq)

        q_heads: list[list[list[float]]] = []
        k_heads: list[list[list[float]]] = []
        v_heads: list[list[list[float]]] = []
        for q, k, v in zip(q_all, k_all, v_all):
            q_heads.append([q[h * self.head_dim:(h + 1) * self.head_dim] for h in range(self.num_heads)])
            k_heads.append([k[h * self.head_dim:(h + 1) * self.head_dim] for h in range(self.num_heads)])
            v_heads.append([v[h * self.head_dim:(h + 1) * self.head_dim] for h in range(self.num_heads)])

        out_seq: list[list[float]] = []
        for t in range(len(seq)):
            merged: list[float] = []
            for h in range(self.num_heads):
                qh = q_heads[t][h]
                scores: list[float] = []
                for j in range(t + 1):
                    score = _dot(qh, k_heads[j][h]) * self.score_scale
                    scores.append(score)
                probs = _softmax(scores)
                acc = [0.0] * self.head_dim
                for j, p in enumerate(probs):
                    vh = v_heads[j][h]
                    for i in range(self.head_dim):
                        acc[i] += p * vh[i]
                merged.extend(acc)
            out_seq.append(self.o_proj.forward_vec(merged))
        return out_seq


class BitFeedForward:
    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        dim = cfg.d_model
        hidden = cfg.ffn_dim
        thr = cfg.ternary_threshold
        self.up_proj = BitLinear(dim, hidden, seed=seed + 101, threshold=thr)
        self.gate_proj = BitLinear(dim, hidden, seed=seed + 211, threshold=thr)
        self.down_proj = BitLinear(hidden, dim, seed=seed + 307, threshold=thr)

    def forward_vec(self, x: list[float]) -> list[float]:
        up = self.up_proj.forward_vec(x)
        gate = self.gate_proj.forward_vec(x)
        hidden = [_silu(g) * u for g, u in zip(gate, up)]
        return self.down_proj.forward_vec(hidden)

    def forward_seq(self, seq: list[list[float]]) -> list[list[float]]:
        return [self.forward_vec(x) for x in seq]


class BitNetBlock:
    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = BitSelfAttention(cfg, seed=seed + 1000)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = BitFeedForward(cfg, seed=seed + 2000)

    def forward(self, seq: list[list[float]]) -> list[list[float]]:
        n1 = self.norm1.forward_seq(seq)
        attn_out = self.attn.forward(n1)
        mid = []
        for x, y in zip(seq, attn_out):
            mid.append([a + b for a, b in zip(x, y)])
        n2 = self.norm2.forward_seq(mid)
        mlp_out = self.mlp.forward_seq(n2)
        out = []
        for x, y in zip(mid, mlp_out):
            out.append([a + b for a, b in zip(x, y)])
        return out


class BitNetLM:
    def __init__(self, cfg: ModelConfig, *, seed: int = 1337) -> None:
        self.cfg = cfg
        rnd = random.Random(seed)
        self.token_embedding: list[list[float]] = []
        for _ in range(cfg.vocab_size):
            self.token_embedding.append([(rnd.random() * 2.0 - 1.0) * 0.18 for _ in range(cfg.d_model)])
        self.positional = self._build_positional(cfg.context_size, cfg.d_model)
        self.blocks = [BitNetBlock(cfg, seed=seed + 5000 * i) for i in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = BitLinear(cfg.d_model, cfg.vocab_size, seed=seed + 9090, threshold=cfg.ternary_threshold, bias=False)

    @staticmethod
    def _build_positional(length: int, dim: int) -> list[list[float]]:
        rows: list[list[float]] = []
        for pos in range(length):
            row = [0.0] * dim
            for i in range(0, dim, 2):
                div = math.exp(-(math.log(10000.0) * i) / max(1, dim))
                row[i] = math.sin(pos * div) * 0.10
                if i + 1 < dim:
                    row[i + 1] = math.cos(pos * div) * 0.10
            rows.append(row)
        return rows

    def forward_last(self, token_ids: list[int]) -> list[float]:
        if not token_ids:
            token_ids = [0]
        token_ids = token_ids[-self.cfg.context_size:]
        seq: list[list[float]] = []
        for pos, tok in enumerate(token_ids):
            emb = self.token_embedding[tok]
            posv = self.positional[pos]
            seq.append([emb[i] + posv[i] for i in range(self.cfg.d_model)])
        for block in self.blocks:
            seq = block.forward(seq)
        last = self.final_norm.forward_vec(seq[-1])
        return self.lm_head.forward_vec(last)

    def total_ternary_params(self) -> int:
        count = 0
        for block in self.blocks:
            for layer in (
                block.attn.q_proj,
                block.attn.k_proj,
                block.attn.v_proj,
                block.attn.o_proj,
                block.mlp.up_proj,
                block.mlp.gate_proj,
                block.mlp.down_proj,
            ):
                count += layer.in_features * layer.out_features
        count += self.lm_head.in_features * self.lm_head.out_features
        return count

    def average_nonzero_ratio(self) -> float:
        ratios: list[float] = []
        for block in self.blocks:
            for layer in (
                block.attn.q_proj,
                block.attn.k_proj,
                block.attn.v_proj,
                block.attn.o_proj,
                block.mlp.up_proj,
                block.mlp.gate_proj,
                block.mlp.down_proj,
            ):
                ratios.append(layer.nonzero_ratio())
        ratios.append(self.lm_head.nonzero_ratio())
        return sum(ratios) / max(1, len(ratios))


class BigramPrior:
    def __init__(self, tokenizer: ByteTokenizer, texts: list[str]) -> None:
        size = tokenizer.vocab_size
        counts = [[1 for _ in range(size)] for _ in range(size)]
        for text in texts:
            toks = tokenizer.encode(text, add_bos=True, add_eos=True)
            for prev, cur in zip(toks, toks[1:]):
                counts[prev][cur] += 1

        self.log_probs: list[list[float]] = []
        for row in counts:
            total = float(sum(row))
            self.log_probs.append([math.log(c / total) for c in row])

    def logits(self, prev_token: int) -> list[float]:
        return self.log_probs[prev_token]


STYLE_CORPUS = [
    "Hi. The GUI is online. The local CAT R1.1.1A engine is fully ready.",
    "Files are off. Everything runs in one Python file with tkinter and stdlib only.",
    "Ask for /profile or /model to inspect the architecture configuration.",
    "Give me the exact error line, the expected result, and the actual result.",
    "Here is a clean way to do it: keep the GUI simple, keep the model tiny, and keep the code readable.",
    "The transformer stack uses ternary BitLinear layers mimicking distilled architectures.",
    "The attention path is causal, so each token only sees earlier tokens.",
    "The feed-forward path uses a gated nonlinear block and projects back to model width.",
    "Use small prompts for better local results.",
    "When you ask for Python code, I return direct code blocks.",
    "A tiny local model is best for compact tasks, UI demos, and structured experiments.",
    "The bootstrap weights are embedded in memory. No external checkpoint is required.",
    "Try commands like /profile, /model, /reset, or ask for a Python snippet.",
    "For debugging, share the traceback and I will narrow it down.",
    "For architecture work, I can describe the tokenizer, the context size, and the ternary layers.",
]


@dataclass(slots=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


_FORBIDDEN_CODE_RE = re.compile(
    r"\b(import\s+(os|sys|subprocess|shutil|socket|pathlib|ctypes|multiprocessing)"
    r"|__import__|open\s*\(|exec\s*\(|eval\s*\(|compile\s*\(|globals\s*\(|locals\s*\()"
)


def _extract_python_code(text: str) -> str | None:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if text.strip().startswith("```"):
        return text.strip().strip("`").replace("python", "", 1).strip()
    for prefix in ("/python ", "/run ", "/exec ", "run python:", "execute:"):
        if text.lower().startswith(prefix):
            return text[len(prefix) :].strip()
    return None


class PythonSandbox:
    """Restricted in-process Python interpreter (ChatGPT/Gemini-style code runner)."""

    MAX_CODE_CHARS = 4000
    TIMEOUT_SEC = 4.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._allowed_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }

    def run(self, code: str) -> SandboxResult:
        code = (code or "").strip()
        if not code:
            return SandboxResult("", "No code provided.", 1)
        if len(code) > self.MAX_CODE_CHARS:
            return SandboxResult("", f"Code too long (max {self.MAX_CODE_CHARS} chars).", 1)
        if _FORBIDDEN_CODE_RE.search(code):
            return SandboxResult("", "Blocked: imports and file/process access are disabled in the sandbox.", 1)

        out_buf = io.StringIO()
        err_buf = io.StringIO()
        result_box: list[SandboxResult] = []

        def target() -> None:
            globs = {"__builtins__": dict(self._allowed_builtins), "__name__": "__sandbox__"}
            locs: dict[str, object] = {}
            try:
                with redirect_stdout(out_buf), redirect_stderr(err_buf):
                    exec(compile(code, "<sandbox>", "exec"), globs, locs)
                result_box.append(SandboxResult(out_buf.getvalue(), err_buf.getvalue(), 0))
            except Exception:
                err_buf.write(traceback.format_exc())
                result_box.append(SandboxResult(out_buf.getvalue(), err_buf.getvalue(), 1))

        thread = threading.Thread(target=target, daemon=True)
        with self._lock:
            thread.start()
            thread.join(self.TIMEOUT_SEC)
        if thread.is_alive():
            return SandboxResult(out_buf.getvalue(), "Execution timed out.\n", 124, timed_out=True)
        return result_box[0] if result_box else SandboxResult("", "Sandbox failed to start.", 1)

    def format_result(self, result: SandboxResult) -> str:
        parts = ["**Python sandbox**"]
        if result.stdout.strip():
            parts.append("```\n" + result.stdout.rstrip() + "\n```")
        if result.stderr.strip():
            parts.append("```\n" + result.stderr.rstrip() + "\n```")
        if not result.stdout.strip() and not result.stderr.strip():
            parts.append("_(no output)_")
        parts.append(f"exit={result.exit_code}" + (" (timeout)" if result.timed_out else ""))
        return "\n".join(parts)


class TerminalSandbox:
    """Mini terminal: shell-like commands without leaving the app."""

    def __init__(self, python: PythonSandbox) -> None:
        self.python = python
        self.history: list[str] = []

    def help_text(self) -> str:
        return textwrap.dedent(
            """
            Terminal sandbox commands:
              help              show this help
              clear             clear terminal scrollback (GUI)
              history           last commands
              python <code>     run one line of Python
              run               multiline Python (end with .end on its own line)
              canvas demo       draw sample shapes on the canvas tab
              canvas clear      clear canvas
              canvas line x1 y1 x2 y2 [#color]
              canvas rect x1 y1 x2 y2 [#color]
              canvas text x y "message" [#color]
              date | time       local clock
            """
        ).strip()

    def run(self, line: str, *, canvas: "CanvasWorkspace") -> tuple[str, list[dict]]:
        raw = (line or "").rstrip("\n")
        if not raw.strip():
            return "", []
        self.history.append(raw)
        pl = raw.strip().lower()
        if pl in ("help", "?"):
            return self.help_text(), []
        if pl == "history":
            return "\n".join(self.history[-12:]) or "(empty)", []
        if pl in ("date", "time"):
            now = datetime.now()
            return now.strftime("%Y-%m-%d %H:%M:%S"), []
        if pl.startswith("python "):
            res = self.python.run(raw[7:])
            return self.python.format_result(res), []
        if pl == "run":
            return "Paste code, then a line with only `.end` to execute.", []
        if pl.startswith("canvas "):
            return canvas.parse_command(raw[7:].strip())
        if pl == "clear":
            return "__CLEAR__", []
        return f"Unknown command: {raw.split()[0]!r}. Type `help`.", []


class CanvasWorkspace:
    """Gemini/ChatGPT-style drawable canvas (tkinter renders ops)."""

    WIDTH = 420
    HEIGHT = 280
    BG = "#0a0a12"

    def __init__(self) -> None:
        self.ops: list[dict] = []

    def clear(self) -> None:
        self.ops.clear()

    def parse_command(self, cmd: str) -> tuple[str, list[dict]]:
        pl = cmd.lower().strip()
        if pl == "clear":
            self.clear()
            return "Canvas cleared.", []
        if pl == "demo":
            self.clear()
            self.ops.extend(
                [
                    {"op": "rect", "coords": [20, 20, 400, 260], "outline": "#333355", "width": 2},
                    {"op": "line", "coords": [40, 200, 380, 60], "fill": "#00d9ff", "width": 3},
                    {"op": "oval", "coords": [120, 80, 220, 180], "outline": "#ff6b9d", "width": 2},
                    {"op": "text", "coords": [50, 40], "text": "CAT R1 Canvas", "fill": "#00ffaa"},
                ]
            )
            return "Canvas demo drawn. Open the Canvas tab to view.", list(self.ops)
        parts = cmd.split()
        if not parts:
            return "Canvas: try `demo`, `clear`, `line`, `rect`, or `text`.", []
        op = parts[0].lower()
        try:
            if op == "line" and len(parts) >= 5:
                color = parts[5] if len(parts) > 5 else "#00d9ff"
                self.ops.append({"op": "line", "coords": list(map(int, parts[1:5])), "fill": color, "width": 2})
                return "Line added.", list(self.ops)
            if op == "rect" and len(parts) >= 5:
                color = parts[5] if len(parts) > 5 else "#00aaff"
                self.ops.append({"op": "rect", "coords": list(map(int, parts[1:5])), "outline": color, "width": 2})
                return "Rectangle added.", list(self.ops)
            if op == "text" and len(parts) >= 4:
                x, y = int(parts[1]), int(parts[2])
                msg = " ".join(parts[3:]).strip('"').strip("'")
                color = "#ffffff"
                self.ops.append({"op": "text", "coords": [x, y], "text": msg, "fill": color})
                return "Text added.", list(self.ops)
        except ValueError:
            return "Canvas parse error: check numeric coordinates.", []
        return f"Unknown canvas command: {op}", []


def _render_canvas(canvas: object, ops: list[dict]) -> None:
    import tkinter as tk

    c = canvas  # tk.Canvas
    c.delete("all")
    c.configure(bg=CanvasWorkspace.BG)
    for item in ops:
        op = item.get("op")
        coords = item.get("coords", [])
        if op == "line":
            c.create_line(*coords, fill=item.get("fill", "#00d9ff"), width=item.get("width", 2))
        elif op == "rect":
            c.create_rectangle(*coords, outline=item.get("outline", "#00aaff"), width=item.get("width", 2))
        elif op == "oval":
            c.create_oval(*coords, outline=item.get("outline", "#ff6b9d"), width=item.get("width", 2))
        elif op == "text":
            c.create_text(coords[0], coords[1], text=item.get("text", ""), fill=item.get("fill", "#fff"), anchor="nw")


class ConversationalHeuristics:
    """Natural-language replies (Gemini/ChatGPT-style) without the tiny LM."""

    JOKES = [
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "I told my computer I needed a break. It said: no problem, I will go to sleep.",
    ]

    @classmethod
    def try_reply(cls, prompt: str, history: list[tuple[str, str]]) -> str | None:
        p = prompt.strip()
        if not p:
            return None
        pl = p.lower()
        words = pl.split()

        if pl in ("thanks", "thank you", "thx", "ty"):
            return "You are welcome. Happy to help."
        if pl in ("bye", "goodbye", "see you", "see ya", "later"):
            return "Goodbye. The sandbox and canvas stay here when you return."
        if pl in ("how are you", "how are you?", "how r u", "how's it going", "how are things"):
            return "I am running locally and ready. How can I help you today?"
        if any(q in pl for q in ("who are you", "what are you", "your name")):
            return (
                f"I am {BOT_NAME}, a single-file local assistant with a BitNet core, "
                "a Python sandbox, a terminal tab, and a drawable canvas. No cloud API required."
            )
        if any(q in pl for q in ("what can you do", "capabilities", "features", "help me")):
            return (
                "I can chat in plain language, run Python in a sandbox, use the terminal tab "
                "(`help`, `python print(1)`, `canvas demo`), draw on the canvas, and answer "
                "simple math. Try `/help` or open the Terminal / Python / Canvas tabs."
            )
        if "joke" in pl:
            return random.choice(cls.JOKES)
        if any(q in pl for q in ("time", "what time", "date", "what day", "today")):
            now = datetime.now()
            return f"Local time: {now.strftime('%A, %B %d, %Y — %H:%M:%S')}."
        if pl.startswith("translate ") and len(words) >= 3:
            phrase = " ".join(words[2:])
            return f'I cannot call external APIs, but here is your phrase echoed: "{phrase}".'
        if pl.startswith("define ") or pl.startswith("what is ") and "2+2" not in pl and not re.search(r"\d\s*[+\-*/]", pl):
            topic = p.split(maxsplit=2)[-1].strip("?.")
            if topic:
                return (
                    f"In plain terms, {topic} is whatever you are building toward in your project. "
                    "If you want a precise definition, tell me the field (programming, math, hardware)."
                )
        if any(w in pl for w in ("sad", "stressed", "anxious", "overwhelmed")):
            return (
                "That sounds tough. Break the problem into one small step, test it, then move on. "
                "If you share the exact blocker, I can help debug in the Python sandbox."
            )
        if any(w in pl for w in ("awesome", "great job", "nice", "cool", "amazing")):
            return "Glad that helped. Want to try the canvas or run some code next?"
        if len(words) <= 2 and pl in ("yes", "no", "ok", "okay", "sure", "yep", "nope"):
            if history and history[-1][0] == "assistant":
                return "Got it. Tell me the next step you want to take."
            return "Understood. What would you like to do next?"
        if pl.startswith("repeat ") or pl == "again":
            for role, text in reversed(history):
                if role == "assistant":
                    return f"Last reply was: {text[:400]}"
            return "No prior reply to repeat yet."
        if "explain" in pl or pl.startswith("how do i ") or pl.startswith("how to "):
            return (
                "Here is a simple approach: (1) state the goal in one sentence, "
                "(2) list inputs and outputs, (3) test the smallest case in the Python tab, "
                "(4) expand once that works."
            )
        if any(w in pl for w in ("opinion", "think about", "should i")):
            return (
                "I would weigh tradeoffs: speed vs clarity, and whether a 10-line prototype "
                "in the sandbox answers the question before you commit to a bigger design."
            )
        if re.match(r"^(hi|hello|hey)\b", pl) and len(words) <= 4:
            return "Hi there. Ask me anything, or type `/terminal help` to see sandbox commands."
        return None


@dataclass(slots=True)
class ConversationThread:
    id: str
    title: str
    history: list[tuple[str, str]] = field(default_factory=list)
    created: float = field(default_factory=time.time)


class ConversationStore:
    """ChatGPT-style multi-chat sidebar."""

    def __init__(self) -> None:
        self.threads: dict[str, ConversationThread] = {}
        self.active_id: str = ""
        self.new_chat("Welcome chat")

    def new_chat(self, title: str = "New chat") -> str:
        cid = uuid.uuid4().hex[:10]
        self.threads[cid] = ConversationThread(id=cid, title=title[:48])
        self.active_id = cid
        return cid

    def active(self) -> ConversationThread:
        if self.active_id not in self.threads:
            self.new_chat()
        return self.threads[self.active_id]

    def titles(self) -> list[tuple[str, str]]:
        items = sorted(self.threads.values(), key=lambda t: t.created, reverse=True)
        return [(t.id, t.title) for t in items]


class CanvasDocument:
    """ChatGPT Canvas: collaborative document / code draft pane."""

    def __init__(self) -> None:
        self.text = (
            "# Canvas document\n\n"
            "Use this pane like ChatGPT Canvas: draft essays, code, or notes.\n"
            "Say **open canvas** or `/doc show` to focus here.\n"
        )

    def append(self, block: str) -> None:
        self.text = (self.text.rstrip() + "\n\n" + block.strip() + "\n").lstrip()

    def replace(self, text: str) -> None:
        self.text = text

    def summarize_request(self, prompt: str) -> str:
        pl = prompt.lower()
        if "outline" in pl:
            return self.text + "\n\n## Outline\n- Introduction\n- Main points\n- Conclusion\n"
        if any(k in pl for k in ("essay", "article", "write", "draft")):
            topic = prompt.strip().strip("?.")
            return self.text + f"\n\n## Draft\n**Topic:** {topic}\n\nOpening paragraph goes here.\n"
        return self.text + f"\n\n> Added from chat: {prompt[:200]}\n"


class VirtualFileStore:
    """In-memory file attach (ChatGPT upload without disk)."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def add(self, name: str, content: str) -> str:
        safe = re.sub(r"[^\w.\-]+", "_", name.strip())[:64] or "paste.txt"
        self.files[safe] = content[:50000]
        return safe

    def list_names(self) -> list[str]:
        return list(self.files.keys())

    def get(self, name: str) -> str | None:
        return self.files.get(name)

    def analyze_prompt(self, name: str) -> str:
        body = self.files.get(name)
        if not body:
            return f"File {name!r} not found."
        lines = body.splitlines()
        preview = "\n".join(lines[:12])
        return (
            f"**File `{name}`** ({len(body)} chars, {len(lines)} lines)\n"
            f"```\n{preview}\n```\n"
            f"Ask me to summarize, find bugs, or extract data from this file."
        )


class UserMemory:
    """ChatGPT memory: lightweight facts from chat."""

    def __init__(self) -> None:
        self.facts: dict[str, str] = {}

    def ingest(self, prompt: str) -> None:
        m = re.search(r"(?:my name is|call me|i am)\s+([A-Za-z][A-Za-z0-9 _\-]{1,30})", prompt, re.I)
        if m:
            self.facts["name"] = m.group(1).strip()
        if "prefer" in prompt.lower():
            self.facts["preference"] = prompt.strip()[:200]

    def summary(self) -> str:
        if not self.facts:
            return "No saved memory yet. Say `my name is ...` or set facts in the Memory tab."
        return "\n".join(f"- **{k}**: {v}" for k, v in self.facts.items())

    def context_line(self) -> str:
        if not self.facts:
            return ""
        return "User memory: " + "; ".join(f"{k}={v}" for k, v in self.facts.items())


class CodeInterpreterTool:
    """Auto-runs Python like ChatGPT Advanced Data Analysis."""

    TRIGGER_WORDS = (
        "run", "execute", "calculate", "compute", "plot", "chart", "analyze",
        "python", "code", "interpreter", "sandbox", "fibonacci", "prime",
    )

    def __init__(self, sandbox: PythonSandbox) -> None:
        self.sandbox = sandbox

    def should_auto_run(self, prompt: str, mode: str) -> bool:
        pl = prompt.lower()
        if mode == MODE_CODE:
            return True
        if "```" in prompt:
            return True
        if any(w in pl for w in self.TRIGGER_WORDS) and (
            _extract_python_code(prompt) or "print(" in pl or "chart_" in pl
        ):
            return True
        return False

    def run_prompt(self, prompt: str) -> tuple[str, list[dict]]:
        code = _extract_python_code(prompt)
        if not code:
            code = self._synthesize_code(prompt)
        if not code:
            return "No runnable Python detected. Use ```python blocks or /run <code>.", []
        self.sandbox.ctx.canvas_ops.clear()
        result = self.sandbox.run(code)
        self.sandbox.ctx.canvas_ops  # keep ops
        msg = self.sandbox.format_result(result)
        return msg, list(self.sandbox.ctx.canvas_ops)

    def _synthesize_code(self, prompt: str) -> str | None:
        pl = prompt.lower()
        if "fibonacci" in pl:
            m = re.search(r"(\d+)", prompt)
            n = int(m.group(1)) if m else 10
            n = min(n, 30)
            return (
                f"n = {n}\n"
                "a, b = 0, 1\n"
                "seq = []\n"
                "for _ in range(n):\n"
                "    seq.append(a)\n"
                "    a, b = b, a + b\n"
                "print(seq)\n"
                "chart_bar([str(i) for i in range(len(seq))], seq, title='Fibonacci')\n"
            )
        if "plot" in pl or "chart" in pl or "graph" in pl:
            return (
                "xs = list(range(10))\n"
                "ys = [x * x for x in xs]\n"
                "print(list(zip(xs, ys)))\n"
                "chart_line(xs, ys, title='y = x^2')\n"
            )
        if "prime" in pl:
            return (
                "def primes(n):\n"
                "    out = []\n"
                "    for p in range(2, n + 1):\n"
                "        if all(p % d for d in range(2, int(p**0.5) + 1)):\n"
                "            out.append(p)\n"
                "    return out\n"
                "ps = primes(50)\n"
                "print(ps)\n"
                "chart_bar([str(p) for p in ps[:12]], ps[:12], title='Primes')\n"
            )
        m = re.search(r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)", prompt)
        if m:
            a, op, b = m.group(1), m.group(2), m.group(3)
            return f"print({a} {op} {b})\n"
        return None


class BitNetEngine:
    def __init__(self) -> None:
        self.history: list[tuple[str, str]] = []
        self.last_aha = ""
        self.last_tool_output: str | None = None
        self.last_canvas_ops: list[dict] = []
        self._multiline_buffer: list[str] = []
        self.python_sandbox = PythonSandbox()
        self.terminal = TerminalSandbox(self.python_sandbox)
        self.canvas = CanvasWorkspace()
        self.tokenizer = ByteTokenizer()
        self.cfg = ModelConfig()
        self.model = BitNetLM(self.cfg, seed=1337)
        self.prior = BigramPrior(self.tokenizer, STYLE_CORPUS)
        self.allowed_tokens = [10] + list(range(32, 127)) + [self.tokenizer.eos_id]

    def profile_text(self) -> str:
        nz = self.model.average_nonzero_ratio() * 100.0
        return (
            f"# {MODEL_NAME}\n\n"
            f"- files = {'off' if not FILES_ENABLED else 'on'}\n"
            f"- target runtime = Python {PYTHON_TARGET}\n"
            f"- GUI = tkinter\n"
            f"- tokenizer = byte-level UTF-8\n"
            f"- context = {self.cfg.context_size} tokens\n"
            f"- d_model = {self.cfg.d_model}\n"
            f"- layers = {self.cfg.n_layers}\n"
            f"- heads = {self.cfg.n_heads}\n"
            f"- feed-forward = {self.cfg.ffn_dim}\n"
            f"- ternary weights = -1, 0, 1 BitLinear\n"
            f"- ternary params = {self.model.total_ternary_params():,}\n"
            f"- average nonzero ratio = {nz:.1f}%\n"
            f"- external files = none\n"
            f"- network/API = off\n"
            f"- python sandbox = on (restricted)\n"
            f"- terminal tab = on\n"
            f"- canvas = on\n"
        )

    def model_text(self) -> str:
        return (
            "BitNet R1.1.1A Stack Profile\n"
            "────────────────────────────\n"
            f"1. Byte tokenizer -> embeddings ({self.cfg.vocab_size} vocab)\n"
            f"2. {self.cfg.n_layers} causal transformer block(s) [Flash emulation config]\n"
            "3. Each block = RMSNorm -> ternary self-attention -> residual\n"
            "4. Then RMSNorm -> ternary gated MLP -> residual\n"
            "5. Final RMSNorm -> ternary LM head\n"
            "\n"
            "This is an explicit ternary BitNet-style structure running on local bootstrap weights."
        )

    def help_text(self) -> str:
        return (
            "Chat commands:\n"
            "- /profile or /pr\n"
            "- /model\n"
            "- /reset\n"
            "- /help\n"
            "- /terminal help\n"
            "- /python <code>  or  /run <code>\n"
            "- /canvas demo | clear\n"
            "\n"
            "GUI tabs: Chat | Terminal | Python | Canvas\n"
            "\n"
            "Try:\n"
            "- hello\n"
            "- what can you do?\n"
            "- run python: print(2 ** 10)\n"
            "- write python code for a timer\n"
            "- draw on canvas (or /canvas demo)\n"
        )

    def _fallback_reply(self, prompt: str) -> str:
        p = prompt.strip()
        pl = p.lower()
        if not p:
            return "Send a prompt. The GUI and R1.1.1A core are ready."
        if any(k in pl for k in ("build", "make", "create", "design")) and any(k in pl for k in ("gui", "model", "bitnet", "transformer")):
            return (
                "Keep the GUI on the main thread, run inference in a worker thread, "
                "use a byte tokenizer, 2 causal BitNet blocks, RMSNorm, ternary attention, "
                "a gated MLP, and a ternary LM head."
            )
        if "?" in p:
            return "I can help. Give me a concrete target, a constraint, or an error line and I will tighten the answer."
        return "R1.1.1A Core is live. Give me a concrete task and I will keep the answer compact."

    def _seed_prefix(self, prompt: str) -> str:
        pl = prompt.lower()
        if any(k in pl for k in ("make", "build", "create")):
            return "A clean build for that is: "
        if any(k in pl for k in ("explain", "how", "why", "?")):
            return "Here is the clean way to frame it: "
        return "My take: "

    def _sample_token(self, logits: list[float], rnd: random.Random, *, top_k: int = 12, temperature: float = 0.82) -> int:
        idx = sorted(self.allowed_tokens, key=lambda i: logits[i], reverse=True)[:top_k]
        top_vals = [logits[i] / max(0.05, temperature) for i in idx]
        probs = _softmax(top_vals)
        r = rnd.random()
        c = 0.0
        for i, p in zip(idx, probs):
            c += p
            if r <= c:
                return i
        return idx[-1]

    def _model_reply(self, prompt: str) -> str:
        prefix = self._seed_prefix(prompt)
        context = (
            "System: You are a compact local assistant running in a tkinter GUI. "
            "Files are off. Reply clearly.\n"
            f"User: {prompt}\n"
            f"Assistant: {prefix}"
        )
        token_ids = self.tokenizer.encode(context, add_bos=True, add_eos=False)
        if len(token_ids) > self.cfg.context_size:
            token_ids = token_ids[: self.cfg.context_size]
        generated: list[int] = []
        rnd = random.Random(_stable_seed(prompt, len(self.history)))
        recent_window = 24

        for _ in range(64):
            bit_logits = self.model.forward_last(token_ids)
            prior_logits = self.prior.logits(token_ids[-1])
            merged = [-1e9] * self.cfg.vocab_size
            recent = token_ids[-recent_window:]
            counts: dict[int, int] = {}
            for tok in recent:
                counts[tok] = counts.get(tok, 0) + 1

            for i in self.allowed_tokens:
                merged[i] = (bit_logits[i] * 0.32) + (prior_logits[i] * 0.68)
                if i in counts:
                    merged[i] -= counts[i] * 0.12

            next_tok = self._sample_token(merged, rnd)
            if next_tok == self.tokenizer.eos_id:
                break
            token_ids.append(next_tok)
            token_ids = token_ids[-self.cfg.context_size:]
            generated.append(next_tok)

            tail = self.tokenizer.decode(generated)
            if tail.endswith("\n\n"):
                break
            if len(tail) > 160 and tail[-1] in ".!?":
                break

        body = _clean_generated(self.tokenizer.decode(generated))
        if _is_low_quality(body):
            return self._fallback_reply(prompt)
        return _clean_generated(prefix + body)

    def _record_exchange(self, prompt: str, reply: str) -> str:
        self.history.append(("user", prompt))
        self.history.append(("assistant", reply))
        if len(self.history) > 40:
            self.history = self.history[-40:]
        return reply

    def generate(self, prompt: str) -> str:
        self.last_aha = ""
        self.last_tool_output = None
        self.last_canvas_ops = []
        raw = (prompt or "").strip()
        pl = raw.lower()

        if pl in ("/pr", "/profile"):
            return self.profile_text()
        if pl in ("/model", "/about"):
            return self.model_text()
        if pl in ("/help", "/?", "help"):
            return self.help_text()
        if pl in ("/reset", "/clear"):
            self.history.clear()
            self.last_aha = ""
            self._multiline_buffer.clear()
            self.canvas.clear()
            return "Conversation history cleared."
        if pl in ("/terminal", "/term"):
            return self.terminal.help_text()
        if pl.startswith("/terminal "):
            msg, ops = self.terminal.run(raw[10:], canvas=self.canvas)
            self.last_canvas_ops = ops
            if msg == "__CLEAR__":
                return "__TERMINAL_CLEAR__"
            return msg
        if pl.startswith("/python ") or pl.startswith("/run ") or pl.startswith("/exec "):
            code = raw.split(" ", 1)[1] if " " in raw else ""
            result = self.python_sandbox.run(code)
            self.last_tool_output = result.stdout + result.stderr
            return self._record_exchange(prompt, self.python_sandbox.format_result(result))
        if pl in ("/canvas", "/canvas help"):
            return "Canvas: /canvas demo | /canvas clear | or use Terminal: `canvas demo`"
        if pl.startswith("/canvas "):
            msg, ops = self.canvas.parse_command(raw[8:])
            self.last_canvas_ops = ops
            return self._record_exchange(prompt, msg)

        code = _extract_python_code(raw)
        if code and any(k in pl for k in ("run", "execute", "eval", "```", "/python", "/run")):
            result = self.python_sandbox.run(code)
            self.last_tool_output = result.stdout + result.stderr
            return self._record_exchange(prompt, self.python_sandbox.format_result(result))

        if pl == ".end" and self._multiline_buffer:
            code = "\n".join(self._multiline_buffer)
            self._multiline_buffer.clear()
            result = self.python_sandbox.run(code)
            self.last_tool_output = result.stdout + result.stderr
            return self._record_exchange(prompt, self.python_sandbox.format_result(result))

        conv = ConversationalHeuristics.try_reply(raw, self.history)
        if conv is not None:
            return self._record_exchange(prompt, conv)

        if pl in ("hi", "hello", "hey", "hi!", "hello!", "hey!"):
            return self._record_exchange(
                prompt,
                "Hi. Chat, Python sandbox, terminal, and canvas are all online.",
            )

        if any(k in pl for k in ("draw", "canvas", "sketch", "plot")) and not pl.startswith("/"):
            if "clear" in pl:
                self.canvas.clear()
                return self._record_exchange(prompt, "Canvas cleared.")
            if "demo" in pl or "circle" in pl or "line" in pl:
                msg, ops = self.canvas.parse_command("demo")
                self.last_canvas_ops = ops
                return self._record_exchange(prompt, msg)
            return self._record_exchange(
                prompt,
                "Open the Canvas tab, or try `/canvas demo` or Terminal: `canvas demo`.",
            )

        if pl.startswith("run python:") or pl.startswith("execute:"):
            code = raw.split(":", 1)[1].strip()
            result = self.python_sandbox.run(code)
            self.last_tool_output = result.stdout + result.stderr
            return self._record_exchange(prompt, self.python_sandbox.format_result(result))

        if "what is" in pl or "what's" in pl:
            m = re.search(r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)", pl)
            if m:
                a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
                if op == "+":
                    val = a + b
                elif op == "-":
                    val = a - b
                elif op == "*":
                    val = a * b
                elif op == "/" and b != 0:
                    val = a / b
                else:
                    val = None
                if val is not None:
                    out = int(val) if val == int(val) else val
                    return self._record_exchange(prompt, f"{a:g} {op} {b:g} = {out}")
        if any(k in pl for k in ("bug", "traceback", "error", "exception")) and "why" in pl:
            self.last_aha = "isolate one concrete failure, then test the smallest input that still breaks."
            return self._record_exchange(
                prompt,
                "Give me the exact error line, the expected result, and the actual result.",
            )
        if "python" in pl and any(k in pl for k in ("write", "code", "snippet", "script")):
            snippet = (
                "```python\n"
                "import time\n"
                "\n"
                "def main() -> None:\n"
                "    print(\"Hello from CAT R1.1.1A\")\n"
                "    for i in range(3):\n"
                "        print(f\"tick {i}\")\n"
                "        time.sleep(0.2)\n"
                "\n"
                "\n"
                "if __name__ == \"__main__\":\n"
                "    main()\n"
                "```\n"
                "Paste into the **Python** tab and click Run, or send `/run` with the code."
            )
            return self._record_exchange(prompt, snippet)
        if any(k in pl for k in ("build", "make", "create", "design")) and any(k in pl for k in ("gui", "model", "bitnet", "transformer")):
            return self._record_exchange(
                prompt,
                "Use a single-file build, keep tkinter as the front end, run inference in a background thread, "
                "and structure the model as tokenizer -> embeddings -> causal BitNet blocks -> ternary LM head.",
            )
        reply = self._model_reply(prompt)
        body = reply
        for pfx in ("My take: ", "Here is the clean way to frame it: ", "A clean build for that is: "):
            if body.startswith(pfx):
                body = body[len(pfx) :]
                break
        if _is_low_quality(body):
            reply = ConversationalHeuristics.try_reply(prompt, self.history) or self._fallback_reply(prompt)
        return self._record_exchange(prompt, reply)


def run_cli() -> None:
    engine = BitNetEngine()
    print(f"{MODEL_NAME} CLI. Type 'exit' to quit.\n")
    while True:
        try:
            msg = input(">>> ")
            if msg.strip().lower() == "exit":
                break
            started = time.perf_counter()
            out = engine.generate(msg)
            elapsed = (time.perf_counter() - started) * 1000.0
            print(out)
            if engine.last_aha:
                print("Aha:", engine.last_aha)
            print(f"[{elapsed:.1f} ms]\n")
        except (EOFError, KeyboardInterrupt):
            break


def run_gui() -> None:
    import tkinter as tk
    from tkinter import font, messagebox, scrolledtext, ttk

    engine = BitNetEngine()

    root = tk.Tk()
    root.title(WINDOW_TITLE)
    root.geometry("1180x720")
    root.configure(bg="#050505")
    root.minsize(900, 560)

    fonts = {
        "mono": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=11),
        "bold": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=11, weight="bold"),
        "italic": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=10, slant="italic"),
        "small": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=9),
    }

    shell = tk.PanedWindow(root, orient="horizontal", bg="#050505", sashwidth=6, sashrelief="flat")
    shell.pack(fill="both", expand=True)

    chat_frame = tk.Frame(shell, bg="#050505")
    tools_frame = tk.Frame(shell, bg="#050505")
    shell.add(chat_frame, minsize=420)
    shell.add(tools_frame, minsize=340)

    chat = scrolledtext.ScrolledText(
        chat_frame,
        bg="#050505",
        fg="#00d9ff",
        font=fonts["mono"],
        insertbackground="cyan",
        relief="flat",
        padx=12,
        pady=12,
        state="disabled",
        wrap="word",
    )
    chat.pack(expand=True, fill="both")

    for tag_name, color, fnt in [
        ("user", "#ffffff", fonts["bold"]),
        ("think", "#4a4a4a", fonts["italic"]),
        ("bot", "#00aaff", fonts["bold"]),
        ("code", "#00ffaa", fonts["small"]),
        ("aha", "#ffd54f", fonts["bold"]),
        ("system", "#8a8a8a", fonts["small"]),
    ]:
        chat.tag_config(tag_name, foreground=color, font=fnt)

    notebook = ttk.Notebook(tools_frame)
    notebook.pack(fill="both", expand=True, padx=4, pady=4)

    tab_terminal = tk.Frame(notebook, bg="#050505")
    tab_python = tk.Frame(notebook, bg="#050505")
    tab_canvas = tk.Frame(notebook, bg="#050505")
    notebook.add(tab_terminal, text="Terminal")
    notebook.add(tab_python, text="Python")
    notebook.add(tab_canvas, text="Canvas")

    term_out = scrolledtext.ScrolledText(
        tab_terminal,
        bg="#0a0a0a",
        fg="#b8ffb8",
        font=fonts["mono"],
        height=12,
        state="disabled",
        wrap="word",
    )
    term_out.pack(fill="both", expand=True, padx=4, pady=4)

    term_in = tk.Entry(tab_terminal, bg="#111", fg="#b8ffb8", font=fonts["mono"], insertbackground="#b8ffb8")
    term_in.pack(fill="x", padx=4, pady=(0, 4))

    py_code = scrolledtext.ScrolledText(
        tab_python,
        bg="#0a0a0a",
        fg="#00ffaa",
        font=fonts["mono"],
        height=10,
        wrap="none",
    )
    py_code.pack(fill="both", expand=True, padx=4, pady=4)
    py_code.insert("1.0", "print('CAT R1 sandbox ready')\nprint(2 ** 10)")

    py_out = scrolledtext.ScrolledText(
        tab_python,
        bg="#050505",
        fg="#888",
        font=fonts["small"],
        height=5,
        state="disabled",
        wrap="word",
    )
    py_out.pack(fill="x", padx=4, pady=(0, 4))

    canvas_widget = tk.Canvas(
        tab_canvas,
        width=CanvasWorkspace.WIDTH,
        height=CanvasWorkspace.HEIGHT,
        bg=CanvasWorkspace.BG,
        highlightthickness=1,
        highlightbackground="#333",
    )
    canvas_widget.pack(fill="both", expand=True, padx=8, pady=8)
    _render_canvas(canvas_widget, engine.canvas.ops)

    canvas_bar = tk.Frame(tab_canvas, bg="#050505")
    canvas_bar.pack(fill="x", padx=8, pady=4)
    for label, cmd in [("Demo", "demo"), ("Clear", "clear")]:
        tk.Button(
            canvas_bar,
            text=label,
            command=lambda c=cmd: _canvas_btn(c),
            bg="#222",
            fg="#00d9ff",
            font=fonts["small"],
            relief="flat",
        ).pack(side="left", padx=4)

    def _canvas_btn(sub: str) -> None:
        msg, ops = engine.canvas.parse_command(sub)
        if ops:
            engine.canvas.ops[:] = ops
        _render_canvas(canvas_widget, engine.canvas.ops)
        term_log(f"[canvas] {msg}\n")

    def term_log(text: str) -> None:
        term_out.config(state="normal")
        term_out.insert("end", text if text.endswith("\n") else text + "\n")
        term_out.config(state="disabled")
        term_out.see("end")

    term_multiline: list[str] = []

    def run_term_line() -> None:
        line = term_in.get().strip()
        if not line:
            return
        term_in.delete(0, "end")
        term_log(f"$ {line}")
        pl = line.lower()
        if pl == "run":
            term_multiline.clear()
            term_log("Multiline Python: enter lines, then `.end`")
            return
        if term_multiline and pl != ".end":
            term_multiline.append(line)
            term_log(f"  + line {len(term_multiline)}")
            return
        if pl == ".end" and term_multiline:
            code = "\n".join(term_multiline)
            term_multiline.clear()
            res = engine.python_sandbox.run(code)
            term_log(engine.python_sandbox.format_result(res))
            return
        msg, ops = engine.terminal.run(line, canvas=engine.canvas)
        if msg == "__CLEAR__":
            term_out.config(state="normal")
            term_out.delete("1.0", "end")
            term_out.config(state="disabled")
            return
        term_log(msg)
        if ops:
            engine.canvas.ops[:] = ops
            _render_canvas(canvas_widget, engine.canvas.ops)

    def run_python_tab() -> None:
        code = py_code.get("1.0", "end")
        res = engine.python_sandbox.run(code)
        py_out.config(state="normal")
        py_out.delete("1.0", "end")
        py_out.insert("end", res.stdout)
        if res.stderr:
            py_out.insert("end", "\n" + res.stderr)
        py_out.insert("end", f"\n[exit {res.exit_code}]")
        py_out.config(state="disabled")

    term_in.bind("<Return>", lambda _e: run_term_line())
    tk.Button(
        tab_terminal,
        text="Run line",
        command=run_term_line,
        bg="#222",
        fg="#00d9ff",
        font=fonts["small"],
        relief="flat",
    ).pack(pady=(0, 6))
    tk.Button(
        tab_python,
        text="Run Python",
        command=run_python_tab,
        bg="#222",
        fg="#00ffaa",
        font=fonts["small"],
        relief="flat",
    ).pack(pady=4)

    term_log(engine.terminal.help_text() + "\n")

    inp = tk.Frame(chat_frame, bg="#050505")
    inp.pack(fill="x", padx=10, pady=5)

    entry = tk.Entry(
        inp,
        bg="#111",
        fg="#00d9ff",
        font=fonts["mono"],
        insertbackground="cyan",
        relief="flat",
        bd=2,
    )
    entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

    btns = tk.Frame(inp, bg="#050505")
    btns.pack(side="right")
    for t, c in [
        ("Help", "/help"),
        ("Term", "/terminal help"),
        ("Canvas", "/canvas demo"),
        ("Profile", "/profile"),
        ("Model", "/model"),
        ("Py", "write python code "),
        ("Reset", "/reset"),
    ]:
        tk.Button(
            btns,
            text=t,
            command=lambda c=c: entry.insert("end", c),
            bg="#222",
            fg="#00d9ff",
            font=fonts["small"],
            relief="flat",
        ).pack(side="left", padx=2)

    status = tk.Label(
        root,
        text=f"Ready | files=off | py3.14 | {BOT_NAME}=online",
        bg="#050505",
        fg="#666",
        font=fonts["small"],
        anchor="w",
    )
    status.pack(fill="x", padx=10, pady=2)

    def log_line(sender: str, text: str, tag: str | None = None) -> None:
        body = _text_insert_safe(text if isinstance(text, str) else str(text), code_fence=(tag == "code"))
        head_tag = "bot" if sender == BOT_NAME else (tag if tag is not None else "think")
        if sender == "SYSTEM":
            head_tag = "system"
        body_tag = tag if tag is not None else ("bot" if sender == BOT_NAME else "think")
        if sender == "SYSTEM":
            body_tag = "system"
        try:
            chat.config(state="normal")
            chat.insert("end", f"[{sender}]: ", head_tag)
            chat.insert("end", f"{body}\n\n", body_tag)
            chat.config(state="disabled")
            chat.see("end")
        except tk.TclError:
            esc = (f"[{sender}]: " + body).encode("unicode_escape", errors="replace").decode("ascii", errors="replace")[:12000]
            chat.config(state="normal")
            chat.insert("end", esc + "\n\n", "think")
            chat.config(state="disabled")
            chat.see("end")

    log_line("SYSTEM", f"{BOT_NAME} ONLINE")
    log_line(
        "SYSTEM",
        "Chat + Terminal sandbox + Python interpreter + Canvas | files=off | /help",
    )

    def send() -> None:
        msg = entry.get().strip()
        if not msg:
            return
        entry.delete(0, "end")
        log_line("YOU", msg, "user")
        status.config(text="Running local BitNet forward pass...")

        def worker() -> None:
            started = time.perf_counter()
            try:
                resp = engine.generate(msg)
            except Exception as e:  # pragma: no cover - GUI safety path
                resp = f"(error) {type(e).__name__}: {e}"
                engine.last_aha = ""
            aha = engine.last_aha
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            def show() -> None:
                if resp == "__TERMINAL_CLEAR__":
                    term_out.config(state="normal")
                    term_out.delete("1.0", "end")
                    term_out.config(state="disabled")
                    status.config(text=f"Ready | {elapsed_ms:.1f} ms | terminal cleared")
                    return
                if engine.last_tool_output:
                    term_log(engine.last_tool_output)
                if engine.last_canvas_ops:
                    engine.canvas.ops[:] = engine.last_canvas_ops
                    _render_canvas(canvas_widget, engine.canvas.ops)
                    notebook.select(tab_canvas)
                if "```" in resp:
                    parts = resp.split("```")
                    for i, part in enumerate(parts):
                        if not part:
                            continue
                        body = part
                        if i % 2 == 1:
                            body = body.lstrip()
                            if body.lower().startswith("python"):
                                body = body[6:].lstrip("\n\r")
                        log_line(BOT_NAME, body, "code" if i % 2 == 1 else None)
                else:
                    log_line(BOT_NAME, resp, None)
                if aha:
                    log_line("AHA", f"Aha: {aha}", "aha")
                status.config(
                    text=f"Ready | {elapsed_ms:.1f} ms | sandbox+canvas | bitnet=online",
                )

            root.after(0, show)

        threading.Thread(target=worker, daemon=True).start()

    entry.bind("<Return>", lambda _e: send())
    entry.focus_set()
    def on_close() -> None:
        if messagebox.askokcancel("Quit", f"Exit {WINDOW_TITLE}?"):
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def main(argv: list[str]) -> int:
    args = set(argv[1:])
    if "--cli" in args or "--headless" in args:
        run_cli()
        return 0
    try:
        run_gui()
        return 0
    except Exception as exc:
        print("GUI failed, switching to CLI.", file=sys.stderr)
        print("Reason:", exc, file=sys.stderr)
        run_cli()
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

