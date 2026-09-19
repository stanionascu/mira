"""Tree-sitter symbol extraction — capabilities the regex fallback misses.

These tests pin behavior only a real parser provides: symbols nested inside
other symbols' bodies (JS/TS class methods, Python methods with qualified
names). The legacy heuristic skips method bodies via brace-jumping, so these
fail without the tree-sitter backend.
"""

from __future__ import annotations

import pytest

from mira.index.extract import extract_symbols, find_symbol_by_name

tree_sitter = pytest.importorskip("tree_sitter")

JS_CLASS = """\
export class AuthService {
  constructor(store) {
    this.store = store;
  }

  async authenticate(token) {
    return this.store.verify(token);
  }

  refresh() {
    return null;
  }
}

export async function topLevel() {
  return 1;
}
"""

PY_CLASS = """\
class Service:
    def run(self, arg):
        return arg

    def _helper(self):
        return None


def top_level():
    return 1
"""


def test_js_class_methods_extracted():
    symbols = extract_symbols(JS_CLASS, "javascript")
    methods = {s.qualified_name: s for s in symbols if s.kind == "method"}
    assert "AuthService.authenticate" in methods
    assert "AuthService.refresh" in methods
    assert "this.store.verify(token)" in methods["AuthService.authenticate"].source


def test_js_methods_findable_by_qualified_name():
    span = find_symbol_by_name(JS_CLASS, "javascript", "AuthService.authenticate")
    assert span is not None
    assert span.name == "authenticate"


def test_js_top_level_symbols_still_found():
    symbols = extract_symbols(JS_CLASS, "javascript")
    by_name = {s.name: s for s in symbols}
    assert by_name["AuthService"].kind == "class"
    assert by_name["topLevel"].kind == "function"


def test_python_methods_get_qualified_names():
    symbols = extract_symbols(PY_CLASS, "python")
    methods = {s.qualified_name: s for s in symbols if s.kind == "method"}
    assert "Service.run" in methods
    assert "Service._helper" in methods
    assert methods["Service.run"].start_line == 2


def test_treesitter_backend_is_engaged():
    from mira.index.extract import _ts_language

    assert _ts_language("python") is not None
    assert _ts_language("go") is not None


def test_short_form_labels_use_grammars():
    assert find_symbol_by_name(PY_CLASS, "py", "Service.run") is not None
    assert find_symbol_by_name(JS_CLASS, "js", "AuthService.refresh") is not None


TSX_COMPONENT = """\
import React from "react";

export function Header({ title }: { title: string }) {
  return <h1>{title}</h1>;
}

const Footer: React.FC = () => {
  return <footer>done</footer>;
};
"""


def test_tsx_label_parses_components():
    symbols = extract_symbols(TSX_COMPONENT, "tsx")
    by_name = {s.name: s for s in symbols}
    assert by_name["Header"].kind == "function"
    assert by_name["Footer"].kind == "function"


def test_typescript_label_finds_functions_and_classes():
    src = "export abstract class Base {\n  abstract load(): void;\n}\n\nexport function run(): void {}\n"
    symbols = extract_symbols(src, "typescript")
    by_name = {s.name: s for s in symbols}
    assert by_name["Base"].kind == "class"
    assert by_name["run"].kind == "function"
    assert "Base.load" in {s.qualified_name for s in symbols}


def test_fallback_to_heuristics_when_grammar_missing(monkeypatch):
    import mira.index.extract as extract_mod

    monkeypatch.setattr(extract_mod, "_TS_LANGUAGE_CACHE", {})
    monkeypatch.setattr(extract_mod, "_TS_UNAVAILABLE", {"python"})
    symbols = extract_symbols(PY_CLASS, "python")
    assert "Service" in {s.name for s in symbols}
    assert "run" in {s.name for s in symbols}
