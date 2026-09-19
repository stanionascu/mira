"""Language-agnostic symbol extractor.

Prefers tree-sitter parsing when the package and a grammar for the language
are installed (precise spans, including symbols nested inside other symbols'
bodies such as class methods); falls back to heuristic regex/brace scanning
otherwise. The module imports cleanly without tree-sitter installed.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Any

# Languages that use indentation-based scoping
_INDENTATION_LANGUAGES = {"python", "py"}

# Languages that use brace-based scoping
_BRACE_LANGUAGES = {
    "javascript",
    "js",
    "typescript",
    "ts",
    "tsx",
    "jsx",
    "go",
    "rust",
    "rs",
    "java",
    "c",
    "cpp",
    "cs",
    "swift",
    "kotlin",
    "kt",
    "scala",
    "php",
}

# Python patterns
_PY_DEF = re.compile(r"^(\s*)(async\s+)?def\s+(\w+)")
_PY_CLASS = re.compile(r"^(\s*)class\s+(\w+)")
_PY_DECORATOR = re.compile(r"^\s*@")

# JS/TS patterns
_JS_FUNCTION = re.compile(r"^(\s*)(?:export\s+)?(?:async\s+)?function\s+(\w+)")
_JS_ARROW = re.compile(
    r"^(\s*)(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>"
)
_JS_CLASS = re.compile(r"^(\s*)(?:export\s+)?(?:default\s+)?class\s+(\w+)")
_JS_METHOD = re.compile(r"^(\s*)(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{")

# Go patterns
_GO_FUNC = re.compile(r"^func\s+(?:\(\s*\w+\s+\*?(\w+)(?:\[[^\]]*\])?\s*\)\s+)?(\w+)\s*\(")
_GO_TYPE = re.compile(r"^type\s+(\w+)\s+(struct|interface)\s*\{")

# Rust patterns
_RUST_FN = re.compile(r"^(\s*)(?:pub\s+)?(?:async\s+)?fn\s+(\w+)")
_RUST_STRUCT = re.compile(r"^(\s*)(?:pub\s+)?struct\s+(\w+)")
_RUST_IMPL = re.compile(r"^(\s*)impl(?:<[^>]*>)?\s+(\w+)")

# Java patterns
_JAVA_METHOD = re.compile(
    r"^(\s*)(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?"
    r"(?:\w+(?:<[^>]*>)?(?:\[\])*\s+)(\w+)\s*\("
)
_JAVA_CLASS = re.compile(
    r"^(\s*)(?:(?:public|private|protected|static|abstract|final|sealed|non-sealed|strictfp)\s+)*"
    r"(?:class|interface|enum|record)\s+(\w+)"
)
_JAVA_CTOR = re.compile(r"^(\s*)(?:public|private|protected)\s+(\w+)\s*\(")


@dataclass
class SymbolSpan:
    """A located symbol extracted from source code."""

    name: str
    kind: str  # "function", "class", "method", "struct", "impl"
    start_line: int  # 1-based
    end_line: int  # 1-based, inclusive
    source: str
    # Container-scoped name where one exists, e.g. "ClassName.method" or
    # "Receiver.Method". Empty for top-level symbols.
    qualified_name: str = ""


def extract_symbols(source: str, language: str) -> list[SymbolSpan]:
    """Extract all top-level symbols from source code.

    Args:
        source: The source code text.
        language: Language identifier (e.g. "python", "js", "go").

    Returns:
        List of extracted symbols with their source code spans.
    """
    if not source.strip():
        return []

    ts_symbols = _extract_treesitter(source, language)
    if ts_symbols is not None:
        return ts_symbols

    style = _detect_style(source, language)
    if style == "indentation":
        return _extract_indentation_based(source)
    return _extract_brace_based(source, language)


def find_symbol_by_name(source: str, language: str, name: str) -> SymbolSpan | None:
    """Find a specific symbol by plain or container-qualified name."""
    for sym in extract_symbols(source, language):
        if name in (sym.name, sym.qualified_name):
            return sym
    return None


def _detect_style(source: str, language: str) -> str:
    """Pick indentation- or brace-based extraction; falls back to a
    def/class keyword scan so unknown extensions on Python files don't
    get misclassified as brace-style by dict literals."""
    lang = language.lower().strip()

    if lang in _INDENTATION_LANGUAGES:
        return "indentation"
    if lang in _BRACE_LANGUAGES:
        return "brace"

    # Unknown language — heuristic: check for Python-style def/class keywords
    lines = source.splitlines()
    has_def_class = any(_PY_DEF.match(line) or _PY_CLASS.match(line) for line in lines)
    if has_def_class:
        return "indentation"

    return "brace"


def _extract_indentation_based(source: str) -> list[SymbolSpan]:
    """Extract symbols from indentation-based languages (Python)."""
    lines = source.splitlines()
    symbols: list[SymbolSpan] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        def_match = _PY_DEF.match(line)
        cls_match = _PY_CLASS.match(line)
        match = def_match or cls_match

        if match:
            indent = len(match.group(1))
            if def_match:
                name = def_match.group(3)
                kind = "function" if indent == 0 else "method"
            else:
                assert cls_match is not None
                name = cls_match.group(2)
                kind = "class"

            start = i
            while start > 0 and _PY_DECORATOR.match(lines[start - 1]):
                start -= 1

            end = i + 1
            while end < len(lines):
                next_line = lines[end]
                if not next_line.strip():
                    end += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= indent:
                    break
                end += 1

            body = "\n".join(lines[start:end])
            symbols.append(
                SymbolSpan(
                    name=name,
                    kind=kind,
                    start_line=start + 1,
                    end_line=end,
                    source=body,
                )
            )

            # Stay on the class line so inner methods get found on the next iteration.
            if kind == "class":
                i += 1
            else:
                i = end
        else:
            i += 1

    return symbols


def _extract_brace_based(source: str, language: str) -> list[SymbolSpan]:
    """Extract symbols from brace-based languages."""
    lang = language.lower().strip()
    lines = source.splitlines()

    if lang in ("go",):
        return _extract_go(lines)
    if lang in ("rust", "rs"):
        return _extract_rust(lines)
    if lang in ("java",):
        return _extract_java(lines)
    # Default: JS/TS/C-like
    return _extract_js_ts(lines)


def _extract_js_ts(lines: list[str]) -> list[SymbolSpan]:
    """Extract symbols from JavaScript/TypeScript."""
    symbols: list[SymbolSpan] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        cls_match = _JS_CLASS.match(line)
        if cls_match:
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=cls_match.group(2),
                    kind="class",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                )
            )
            i = end + 1
            continue

        fn_match = _JS_FUNCTION.match(line)
        if fn_match:
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=fn_match.group(2),
                    kind="function",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                )
            )
            i = end + 1
            continue

        arrow_match = _JS_ARROW.match(line)
        if arrow_match:
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=arrow_match.group(2),
                    kind="function",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                )
            )
            i = end + 1
            continue

        i += 1

    return symbols


def _extract_go(lines: list[str]) -> list[SymbolSpan]:
    """Extract symbols from Go."""
    symbols: list[SymbolSpan] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        fn_match = _GO_FUNC.match(line)
        if fn_match:
            receiver, name = fn_match.group(1), fn_match.group(2)
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=name,
                    kind="method" if receiver else "function",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                    qualified_name=f"{receiver}.{name}" if receiver else "",
                )
            )
            i = end + 1
            continue

        type_match = _GO_TYPE.match(line)
        if type_match:
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=type_match.group(1),
                    kind=type_match.group(2),
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                )
            )
            i = end + 1
            continue

        i += 1

    return symbols


def _extract_rust(lines: list[str]) -> list[SymbolSpan]:
    """Extract symbols from Rust."""
    symbols: list[SymbolSpan] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        impl_match = _RUST_IMPL.match(line)
        if impl_match:
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=impl_match.group(2),
                    kind="impl",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                )
            )
            i = end + 1
            continue

        struct_match = _RUST_STRUCT.match(line)
        if struct_match:
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=struct_match.group(2),
                    kind="struct",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                )
            )
            i = end + 1
            continue

        fn_match = _RUST_FN.match(line)
        if fn_match:
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=fn_match.group(2),
                    kind="function",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                )
            )
            i = end + 1
            continue

        i += 1

    return symbols


def _extract_java(lines: list[str]) -> list[SymbolSpan]:
    """Extract symbols from Java.

    Descends into class/interface/enum/record bodies (every Java method
    lives inside one) and qualifies methods with their enclosing type,
    mirroring the Python extractor. Method bodies are skipped over so
    statements like `return foo(...)` can't false-match.
    """
    symbols: list[SymbolSpan] = []
    enclosing: list[tuple[str, int]] = []  # (type name, body end index)

    i = 0
    while i < len(lines):
        while enclosing and enclosing[-1][1] < i:
            enclosing.pop()
        line = lines[i]

        cls_match = _JAVA_CLASS.match(line)
        if cls_match:
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=cls_match.group(2),
                    kind="class",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                )
            )
            enclosing.append((cls_match.group(2), end))
            i += 1
            continue

        method_match = _JAVA_METHOD.match(line)
        if not method_match and enclosing:
            ctor = _JAVA_CTOR.match(line)
            if ctor and ctor.group(2) == enclosing[-1][0]:
                method_match = ctor
        if method_match and enclosing:
            # Abstract/interface declarations have no body — span is the line.
            braceless = ";" in line and "{" not in line
            end = i if braceless else _find_brace_end(lines, i)
            name = method_match.group(2)
            symbols.append(
                SymbolSpan(
                    name=name,
                    kind="method",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                    qualified_name=f"{enclosing[-1][0]}.{name}",
                )
            )
            i = end + 1
            continue

        i += 1

    return symbols


def _find_brace_end(lines: list[str], start: int) -> int:
    """Find the line where braces balance out, starting from a given line.

    Scans from ``start`` counting ``{`` and ``}`` until the count returns to zero.
    Returns the line index of the closing brace. If braces never balance,
    returns the last line index.
    """
    depth = 0
    for i in range(start, len(lines)):
        line = lines[i]
        for ch in line:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
    return len(lines) - 1


# ── tree-sitter backend ───────────────────────────────────────────────
# Grammar package per language key: module name, or (module, language-fn)
# for packages exposing several grammars (tree-sitter-typescript).
_TS_GRAMMARS: dict[str, str | tuple[str, str]] = {
    "python": "tree_sitter_python",
    "py": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "js": "tree_sitter_javascript",
    "jsx": "tree_sitter_javascript",
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "ts": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "rs": "tree_sitter_rust",
    "java": "tree_sitter_java",
}

_TS_LANGUAGE_CACHE: dict[str, Any] = {}
_TS_UNAVAILABLE: set[str] = set()

# Node types whose value, when assigned to a variable, defines a function.
_TS_FUNCTION_VALUES = {"arrow_function", "function_expression", "function"}

# Identifier-ish node types for name resolution.
_TS_NAME_TYPES = (
    "identifier",
    "field_identifier",
    "type_identifier",
    "property_identifier",
)


def _ts_language(grammar_key: str) -> Any | None:
    """Load (and cache) the tree-sitter Language for a grammar key.

    Returns None when tree-sitter or the grammar package is not installed.
    """
    if grammar_key in _TS_LANGUAGE_CACHE:
        return _TS_LANGUAGE_CACHE[grammar_key]
    if grammar_key in _TS_UNAVAILABLE:
        return None
    spec = _TS_GRAMMARS.get(grammar_key)
    if spec is None:
        return None
    try:
        from tree_sitter import Language

        if isinstance(spec, tuple):
            module = importlib.import_module(spec[0])
            language = Language(getattr(module, spec[1])())
        else:
            module = importlib.import_module(spec)
            language = Language(module.language())  # type: ignore[attr-defined]
    except Exception:
        _TS_UNAVAILABLE.add(grammar_key)
        return None
    _TS_LANGUAGE_CACHE[grammar_key] = language
    return language


def _extract_treesitter(source: str, language: str) -> list[SymbolSpan] | None:
    """Extract symbols with tree-sitter, or None to use the heuristic fallback.

    Falls back when no grammar covers the language, the grammar package is
    missing, parsing fails, or the parse tree has errors and yielded nothing.
    """
    grammar_key = language.lower().strip()
    if grammar_key not in _TS_GRAMMARS:
        return None
    ts_lang = _ts_language(grammar_key)
    if ts_lang is None:
        return None
    try:
        from tree_sitter import Parser

        src = source.encode("utf-8", errors="replace")
        tree = Parser(ts_lang).parse(src)
    except Exception:
        return None
    lines = source.splitlines()
    symbols: list[SymbolSpan] = []
    try:
        _walk_ts(tree.root_node, src, lines, [], symbols, grammar_key)
    except Exception:
        return None
    if not symbols and tree.root_node.has_error:
        return None
    return symbols


def _ts_text(node: Any, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _ts_node_name(node: Any, src: bytes) -> str:
    """Best-effort symbol name: the ``name`` field, else an identifier child."""
    named = node.child_by_field_name("name")
    if named is not None:
        return _ts_text(named, src)
    for child in node.named_children:
        if child.type in _TS_NAME_TYPES:
            return _ts_text(child, src)
    return ""


def _ts_first_descendant(node: Any, types: set[str]) -> Any | None:
    """First descendant (document order) whose type is in ``types``."""
    stack = list(node.named_children)
    while stack:
        child = stack.pop(0)
        if child.type in types:
            return child
        stack[0:0] = child.named_children
    return None


def _ts_span(
    lines: list[str], node: Any, name: str, kind: str, qualified_name: str = ""
) -> SymbolSpan:
    start = node.start_point[0] + 1  # tree-sitter rows are 0-based
    end = node.end_point[0] + 1
    return SymbolSpan(
        name=name,
        kind=kind,
        start_line=start,
        end_line=end,
        source="\n".join(lines[start - 1 : end]),
        qualified_name=qualified_name,
    )


def _walk_ts(
    node: Any,
    src: bytes,
    lines: list[str],
    enclosing: list[str],
    out: list[SymbolSpan],
    grammar_key: str,
) -> None:
    """Recursive tree walk; ``enclosing`` is the stack of containing type names."""
    node_type = node.type
    if node_type in ("ERROR", "MISSING"):
        return

    if grammar_key in ("python", "py"):
        _walk_ts_python(node, src, lines, enclosing, out)
    elif grammar_key in ("javascript", "js", "jsx", "typescript", "ts", "tsx"):
        _walk_ts_javascript(node, src, lines, enclosing, out)
    elif grammar_key == "go":
        _walk_ts_go(node, src, lines, enclosing, out)
    elif grammar_key in ("rust", "rs"):
        _walk_ts_rust(node, src, lines, enclosing, out)
    elif grammar_key == "java":
        _walk_ts_java(node, src, lines, enclosing, out)
    else:  # pragma: no cover — grammar_key always matches above
        for child in node.named_children:
            _walk_ts(child, src, lines, enclosing, out, grammar_key)


def _walk_ts_children(
    node: Any,
    src: bytes,
    lines: list[str],
    enclosing: list[str],
    out: list[SymbolSpan],
    grammar_key: str,
) -> None:
    for child in node.named_children:
        _walk_ts(child, src, lines, enclosing, out, grammar_key)


def _walk_ts_python(
    node: Any,
    src: bytes,
    lines: list[str],
    enclosing: list[str],
    out: list[SymbolSpan],
) -> None:
    node_type = node.type
    if node_type == "decorated_definition":
        inner = next(
            (
                c
                for c in node.named_children
                if c.type in ("function_definition", "class_definition")
            ),
            None,
        )
        if inner is None:
            return
        _emit_ts_python(inner, node, src, lines, enclosing, out)
        return
    if node_type in ("function_definition", "class_definition"):
        _emit_ts_python(node, node, src, lines, enclosing, out)
        return
    _walk_ts_children(node, src, lines, enclosing, out, "python")


def _emit_ts_python(
    node: Any,
    span_node: Any,
    src: bytes,
    lines: list[str],
    enclosing: list[str],
    out: list[SymbolSpan],
) -> None:
    """Emit a python def/class; ``span_node`` is the outer decorated node."""
    name = _ts_node_name(node, src)
    if not name:
        return
    if node.type == "class_definition":
        out.append(_ts_span(lines, span_node, name, "class"))
        _walk_ts_children(node, src, lines, [*enclosing, name], out, "python")
        return
    if _py_is_method(node):
        qualified = f"{enclosing[-1]}.{name}" if enclosing else ""
        out.append(_ts_span(lines, span_node, name, "method", qualified))
    else:
        out.append(_ts_span(lines, span_node, name, "function"))
    _walk_ts_children(node, src, lines, enclosing, out, "python")


def _py_is_method(node: Any) -> bool:
    """A def is a method when directly inside a class body (via decorators)."""
    parent = node.parent
    if parent is not None and parent.type == "decorated_definition":
        parent = parent.parent
    return (
        parent is not None
        and parent.type == "block"
        and parent.parent is not None
        and parent.parent.type == "class_definition"
    )


def _walk_ts_javascript(
    node: Any,
    src: bytes,
    lines: list[str],
    enclosing: list[str],
    out: list[SymbolSpan],
) -> None:
    node_type = node.type
    if node_type in ("function_declaration", "generator_function_declaration"):
        name = _ts_node_name(node, src)
        if name:
            out.append(_ts_span(lines, node, name, "function"))
        _walk_ts_children(node, src, lines, enclosing, out, "javascript")
        return
    if node_type in ("class_declaration", "abstract_class_declaration"):
        name = _ts_node_name(node, src)
        if not name:
            return
        out.append(_ts_span(lines, node, name, "class"))
        _walk_ts_children(node, src, lines, [*enclosing, name], out, "javascript")
        return
    if node_type in (
        "method_definition",
        "method_signature",
        "abstract_method_signature",
    ):
        name = _ts_node_name(node, src)
        if name:
            qualified = f"{enclosing[-1]}.{name}" if enclosing else ""
            out.append(_ts_span(lines, node, name, "method", qualified))
        _walk_ts_children(node, src, lines, enclosing, out, "javascript")
        return
    if node_type == "variable_declarator":
        value = node.child_by_field_name("value")
        if value is not None and value.type in _TS_FUNCTION_VALUES:
            name_node = node.child_by_field_name("name")
            name = _ts_text(name_node, src) if name_node is not None else ""
            if name:
                out.append(_ts_span(lines, node, name, "function"))
        _walk_ts_children(node, src, lines, enclosing, out, "javascript")
        return
    _walk_ts_children(node, src, lines, enclosing, out, "javascript")


def _walk_ts_go(
    node: Any,
    src: bytes,
    lines: list[str],
    enclosing: list[str],
    out: list[SymbolSpan],
) -> None:
    node_type = node.type
    if node_type == "function_declaration":
        name = _ts_node_name(node, src)
        if name:
            out.append(_ts_span(lines, node, name, "function"))
        return
    if node_type == "method_declaration":
        name = _ts_node_name(node, src)
        if not name:
            return
        receiver = ""
        recv_node = node.child_by_field_name("receiver")
        if recv_node is not None:
            ident = _ts_first_descendant(recv_node, {"type_identifier"})
            if ident is not None:
                receiver = _ts_text(ident, src)
        out.append(_ts_span(lines, node, name, "method", f"{receiver}.{name}" if receiver else ""))
        return
    if node_type == "type_declaration":
        for child in node.named_children:
            if child.type != "type_spec":
                continue
            spec_name = _ts_node_name(child, src)
            inner = next((c for c in child.named_children if c.type.endswith("_type")), None)
            kind = (
                "struct"
                if inner is not None and inner.type == "struct_type"
                else "interface"
                if inner is not None and inner.type == "interface_type"
                else ""
            )
            if spec_name and kind:
                out.append(_ts_span(lines, node, spec_name, kind))
        return
    _walk_ts_children(node, src, lines, enclosing, out, "go")


def _walk_ts_rust(
    node: Any,
    src: bytes,
    lines: list[str],
    enclosing: list[str],
    out: list[SymbolSpan],
) -> None:
    node_type = node.type
    if node_type == "function_item":
        name = _ts_node_name(node, src)
        if name:
            out.append(_ts_span(lines, node, name, "function"))
        _walk_ts_children(node, src, lines, enclosing, out, "rust")
        return
    if node_type == "struct_item":
        name = _ts_node_name(node, src)
        if name:
            out.append(_ts_span(lines, node, name, "struct"))
        _walk_ts_children(node, src, lines, enclosing, out, "rust")
        return
    if node_type == "impl_item":
        ident = _ts_first_descendant(node, {"type_identifier"})
        name = _ts_text(ident, src) if ident is not None else ""
        if name:
            out.append(_ts_span(lines, node, name, "impl"))
        _walk_ts_children(node, src, lines, enclosing, out, "rust")
        return
    _walk_ts_children(node, src, lines, enclosing, out, "rust")


_TS_JAVA_TYPES = {
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
}

_TS_JAVA_METHODS = {
    "method_declaration",
    "constructor_declaration",
    "compact_constructor_declaration",
}


def _walk_ts_java(
    node: Any,
    src: bytes,
    lines: list[str],
    enclosing: list[str],
    out: list[SymbolSpan],
) -> None:
    node_type = node.type
    if node_type in _TS_JAVA_TYPES:
        name = _ts_node_name(node, src)
        if not name:
            return
        out.append(_ts_span(lines, node, name, "class"))
        _walk_ts_children(node, src, lines, [*enclosing, name], out, "java")
        return
    if node_type in _TS_JAVA_METHODS:
        name = _ts_node_name(node, src)
        if name:
            qualified = f"{enclosing[-1]}.{name}" if enclosing else ""
            out.append(_ts_span(lines, node, name, "method", qualified))
        _walk_ts_children(node, src, lines, enclosing, out, "java")
        return
    _walk_ts_children(node, src, lines, enclosing, out, "java")
