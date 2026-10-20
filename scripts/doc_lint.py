#!/usr/bin/env python3
"""Check the shape and wording of doc comments, docstrings, and comments."""

import ast
import inspect
import io
import re
import subprocess
import sys
import tokenize

RUST_SECTIONS = {
    "Arguments",
    "Returns",
    "Errors",
    "Panics",
    "Safety",
    "Example",
    "Examples",
    "References",
}
PY_SECTIONS = {
    "Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Attributes",
    "Example",
    "Examples",
    "References",
}
SUMMARY_MAX_LINES = 2
PARAGRAPH_MAX_WORDS = 40
RUN_MIN_LINES = 3
RUN_MAX_WORDS = 25

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
        r"\b(follows|matches|mirrors|replicates)\s+(the\s+)?(exact\s+)?(algorithm|implementation|code|computation|version)\b",
        "provenance",
    ),
    (r"\bbm3d[-_]streak[-_]removal\b|\b_internal\.py\b", "names another codebase"),
    (r"\b(previously|no longer|was changed|has been changed|legacy)\b", "history"),
    (r"\b(note that|let'?s|key insight|for reference|as discussed)\b", "process"),
    (r"\b(TODO|FIXME|XXX)\b", "open item"),
    (r"(?<![\w&])#\d+\b|\b(PR|issue)\s*#?\d+\b", "PR or issue reference"),
    (r"\b(carefully|high-performance|state-of-the-art|blazing)\b", "emphasis"),
]
BANNED = [(re.compile(p, 0 if why == "open item" else re.I), why) for p, why in BANNED]
DIRECTIVE = re.compile(
    r"^(!|-\*-|noqa\b|type:\s*ignore|pragma\b|fmt:\s*(on|off)|pylint:|SPDX-)", re.I
)


def words(text):
    return len(text.split())


def banned(path, line, text, out):
    for rx, why in BANNED:
        if rx.search(text):
            out.append(f"{path}:{line}: {why}: {text.strip()[:100]}")


def paragraphs(lines):
    para = []
    for n, text in lines:
        if text.strip():
            para.append((n, text))
        elif para:
            yield para
            para = []
    if para:
        yield para


def check_summary(path, body, out):
    summary = []
    for n, text in body:
        if not text.strip():
            break
        summary.append((n, text))
    if not summary:
        out.append(f"{path}:{body[0][0]}: doc has no summary line")
        return len(body)
    if len(summary) > SUMMARY_MAX_LINES:
        out.append(
            f"{path}:{summary[0][0]}: summary is longer than {SUMMARY_MAX_LINES} lines"
        )
    return len(summary)


def check_section_body(path, name, lines, out, example_marker, item_start):
    for para in paragraphs(lines):
        if name in ("Example", "Examples"):
            if not example_marker(para[0][1]):
                out.append(
                    f"{path}:{para[0][0]}: prose in {name}; only code is allowed"
                )
            continue
        items, cur = [], []
        for n, text in para:
            if item_start(text) and cur:
                items.append(cur)
                cur = []
            cur.append((n, text))
        items.append(cur)
        for item in items:
            if sum(words(t) for _, t in item) > PARAGRAPH_MAX_WORDS:
                out.append(
                    f"{path}:{item[0][0]}: {name} entry is longer than {PARAGRAPH_MAX_WORDS} words"
                )


def check_description(path, pre, out):
    for k, para in enumerate(paragraphs(pre)):
        if k:
            out.append(
                f"{path}:{para[0][0]}: only one description paragraph is allowed"
            )
        elif sum(words(t) for _, t in para) > PARAGRAPH_MAX_WORDS:
            out.append(
                f"{path}:{para[0][0]}: description is longer than {PARAGRAPH_MAX_WORDS} words"
            )


def check_rust_doc(path, block, out):
    body = [(n, t[1:] if t.startswith(" ") else t) for n, t in block]
    if (
        body
        and re.match(r"^#\s", body[0][1])
        and body[0][1][2:].strip() not in RUST_SECTIONS
    ):
        out.append(f"{path}:{body[0][0]}: start with a summary sentence, not a heading")
    i = check_summary(path, body, out)
    fence = False
    section, lines, pre = None, [], []

    def flush():
        if section:
            check_section_body(
                path,
                section,
                lines,
                out,
                lambda t: t.strip().startswith("```"),
                lambda t: re.match(r"^[*-]\s", t) is not None,
            )

    for n, text in body[i:]:
        if text.strip().startswith("```"):
            if not fence and section not in ("Example", "Examples"):
                out.append(f"{path}:{n}: code block outside an Examples section")
            fence = not fence
            if section:
                lines.append((n, text))
            continue
        if fence:
            if section:
                lines.append((n, text))
            continue
        m = re.match(r"^#\s+(.+?)\s*$", text)
        if m:
            flush()
            section, lines = m.group(1), []
            if section not in RUST_SECTIONS:
                out.append(f"{path}:{n}: section '{section}' is not allowed")
            continue
        if section is None:
            pre.append((n, text))
        else:
            lines.append((n, text))
    flush()
    check_description(path, pre, out)


def check_comment_run(path, run, out):
    texts = [t.strip() for _, t in run]
    if texts[0].startswith("SAFETY:"):
        return
    if len(run) >= RUN_MIN_LINES and sum(words(t) for t in texts) > RUN_MAX_WORDS:
        out.append(
            f"{path}:{run[0][0]}: comment of {len(run)} lines; keep comments to two lines"
        )


def rust_comments(src):
    toks, i, n, line = [], 0, len(src), 1
    line_start_code = {}
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            toks.append((line, src[i:j], line_start_code.get(line, False)))
            i = j
            continue
        if src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            toks.append((line, src[i:j], True))
            line += src.count("\n", i, j)
            i = j
            continue
        if (
            c == "r"
            and re.match(r'r#*"', src[i : i + 8])
            and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_"))
        ):
            hashes = len(re.match(r"r(#*)", src[i:]).group(1))
            end = src.find('"' + "#" * hashes, i + 2 + hashes)
            end = n if end < 0 else end + 1 + hashes
            line += src.count("\n", i, end)
            line_start_code[line] = True
            i = end
            continue
        if c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            line += src.count("\n", i, j)
            line_start_code[line] = True
            i = j + 1
            continue
        if c == "'" and (src.startswith("\\", i + 1) or src[i + 2 : i + 3] == "'"):
            j = src.find("'", i + 2)
            line_start_code[line] = True
            i = j + 1
            continue
        if not c.isspace():
            line_start_code[line] = True
        i += 1
    return toks


def lint_rust(path, src, out):
    blocks = {"///": [], "//!": [], "//": []}
    prev = {"///": None, "//!": None, "//": None}
    for line, text, after_code in rust_comments(src):
        if text.startswith("/*"):
            banned(path, line, text, out)
            continue
        kind = (
            "///"
            if text.startswith("///") and not text.startswith("////")
            else "//!"
            if text.startswith("//!")
            else "//"
        )
        body = text[len(kind) :]
        banned(path, line, body, out)
        if after_code:
            continue
        if DIRECTIVE.match(body.strip()):
            continue
        if prev[kind] is not None and prev[kind] == line - 1:
            blocks[kind][-1].append((line, body))
        else:
            blocks[kind].append([(line, body)])
        for k in prev:
            prev[k] = line if k == kind else None
    for kind in ("///", "//!"):
        for block in blocks[kind]:
            check_rust_doc(path, block, out)
    for run in blocks["//"]:
        check_comment_run(path, run, out)


def check_py_docstring(path, lineno, doc, out):
    raw = inspect.cleandoc(doc).splitlines()
    body = [(lineno + k, t) for k, t in enumerate(raw)]
    for n, t in body:
        banned(path, n, t, out)
    if not body:
        return
    i = check_summary(path, body, out)
    section, lines, pre, k = None, [], [], i

    def flush():
        if section:
            check_section_body(
                path,
                section,
                lines,
                out,
                lambda t: t.strip().startswith((">>>", "...")),
                lambda t: t[:1].strip() != "",
            )

    while k < len(body):
        n, text = body[k]
        nxt = body[k + 1][1] if k + 1 < len(body) else ""
        if text.strip() and re.match(r"^\s*-{3,}\s*$", nxt):
            flush()
            section, lines = text.strip(), []
            if section not in PY_SECTIONS:
                out.append(f"{path}:{n}: section '{section}' is not allowed")
            k += 2
            continue
        if section is None:
            pre.append((n, text))
        else:
            lines.append((n, text))
        k += 1
    flush()
    check_description(path, pre, out)


def lint_python(path, src, out):
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        out.append(f"{path}:{e.lineno}: syntax error")
        return
    for node in ast.walk(tree):
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
                check_py_docstring(path, b[0].lineno, b[0].value.value, out)
    run, last = [], None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.COMMENT:
            continue
        line = tok.start[0]
        body = tok.string[1:]
        banned(path, line, body, out)
        whole = tok.line.strip().startswith("#")
        if not whole or DIRECTIVE.match(body.strip()):
            continue
        if run and last == line - 1:
            run.append((line, body))
        else:
            if run:
                check_comment_run(path, run, out)
            run = [(line, body)]
        last = line
    if run:
        check_comment_run(path, run, out)


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
        if path.endswith(".rs"):
            lint_rust(path, src, out)
        elif path.endswith(".py"):
            lint_python(path, src, out)
    for line in out:
        print(line)
    return 1 if out else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
