#!/usr/bin/env python3
"""Fail on provenance, history, and process wording in comments and docstrings."""

import ast
import io
import re
import subprocess
import sys
import tokenize

BANNED = [
    (
        r"\b[\w./-]+\.(py|rs|c|cc|cpp|h|hpp|m|jl|js|ts|f90)\s*:\s*\d+",
        "file:line reference",
    ),
    (
        r"\b(reference|original|upstream)\s+(implementation|code|version)s?\b",
        "names another implementation",
    ),
    (r"\b(ported|adapted|translated|copied|derived)\s+from\b", "provenance"),
    (r"\bverbatim\b", "provenance"),
    (
        r"\b(follows|matches|mirrors|replicates)\s+(the\s+)?(exact\s+)?"
        r"(algorithm|implementation|code|computation|version)\b",
        "provenance",
    ),
    (r"\bbm3d[-_]streak[-_]removal\b|\b_internal\.py\b", "names another codebase"),
    (
        r"\bno longer (used|needed|supported|required|called)\b|\b(was changed|has been changed|legacy)\b",
        "history",
    ),
    (r"\b(note that|let's|let us|key insight|for reference|as discussed)\b", "process"),
    (r"\b(TODO|FIXME|XXX)\b", "open item"),
    (r"(?<![\w&])#\d+\b|\b(PR|issue)\s*#?\d+\b", "PR or issue reference"),
]
BANNED = [(re.compile(p, 0 if why == "open item" else re.I), why) for p, why in BANNED]


def check(path, line, text, out):
    for rx, why in BANNED:
        if rx.search(text):
            out.append(f"{path}:{line}: {why}: {text.strip()[:100]}")


def rust_comments(src):
    i, n, line = 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            yield line, src[i:j]
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            yield line, src[i:j]
            line += src.count("\n", i, j)
            i = j
        elif (
            c == "r"
            and re.match(r'r#*"', src[i : i + 8])
            and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_"))
        ):
            hashes = len(re.match(r"r(#*)", src[i:]).group(1))
            end = src.find('"' + "#" * hashes, i + 2 + hashes)
            end = n if end < 0 else end + 1 + hashes
            line += src.count("\n", i, end)
            i = end
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            line += src.count("\n", i, j)
            i = j + 1
        elif c == "'" and (src.startswith("\\", i + 1) or src[i + 2 : i + 3] == "'"):
            i = src.find("'", i + 2) + 1
        else:
            i += 1


def python_comments(src):
    for node in ast.walk(ast.parse(src)):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            b = node.body
            if (
                b
                and isinstance(b[0], ast.Expr)
                and isinstance(b[0].value, ast.Constant)
                and isinstance(b[0].value.value, str)
            ):
                for k, text in enumerate(b[0].value.value.splitlines()):
                    yield b[0].lineno + k, text
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            yield tok.start[0], tok.string


def main(argv):
    paths = (
        argv
        or subprocess.run(
            ["git", "ls-files", "*.rs", "*.py"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    )
    out = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        comments = (
            rust_comments(src)
            if path.endswith(".rs")
            else python_comments(src)
            if path.endswith(".py")
            else ()
        )
        for line, text in comments:
            check(path, line, text, out)
    for line in out:
        print(line)
    return 1 if out else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
