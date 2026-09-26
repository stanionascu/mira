"""React/React-DOM version alignment for the dashboard UI.

React 19 requires ``react`` and ``react-dom`` to resolve to the exact same
version; any divergence crashes every page load with minified React error
#527 ("Incompatible React versions"). ``vite build`` still succeeds with a
mismatch, so without this check CI stays green while the deployed dashboard
is a blank page.
"""

from __future__ import annotations

import json
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent.parent / "ui" / "mira"


def _load_json(name: str) -> dict:
    return json.loads((UI_DIR / name).read_text())


class TestReactVersionAlignment:
    def test_package_json_ranges_agree(self):
        manifest = _load_json("package.json")
        deps = manifest["dependencies"]
        assert deps["react"] == deps["react-dom"], (
            f"react ({deps['react']}) and react-dom ({deps['react-dom']}) ranges "
            "must agree so a fresh install cannot resolve mismatched versions "
            "(React error #527)"
        )

    def test_lockfile_resolves_identical_versions(self):
        lock = _load_json("package-lock.json")
        packages = lock["packages"]
        react = packages["node_modules/react"]["version"]
        react_dom = packages["node_modules/react-dom"]["version"]
        assert react == react_dom, (
            f"package-lock.json resolves react@{react} with react-dom@{react_dom}; "
            "React 19 requires the exact same version (React error #527)"
        )
