"""Light review mode — normalization bundle and engine wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mira.config import MiraConfig, apply_light_mode
from mira.core.engine import ReviewEngine
from mira.llm.prompts.review import build_walkthrough_prompt
from mira.llm.provider import LLMProvider
from mira.models import FileChangeType, FileDiff, HunkInfo


def _light_config(**overrides) -> MiraConfig:
    config = MiraConfig()
    config.review.light_mode = True
    for key, value in overrides.items():
        setattr(config.review, key, value)
    return config


def _walkthrough_files() -> list[FileDiff]:
    return [
        FileDiff(
            path="src/a.py",
            change_type=FileChangeType.MODIFIED,
            language="python",
            added_lines=3,
            deleted_lines=1,
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=2,
                    target_start=1,
                    target_length=3,
                    content="@@ -1,2 +1,3 @@\n-old\n+new\n+more",
                )
            ],
        )
    ]


def test_off_returns_config_unchanged():
    config = MiraConfig()
    assert apply_light_mode(config) is config
    assert config.review.agentic_tools is True
    assert config.review.context_token_budget == 8_000


def test_on_forces_bundle_and_keeps_passes():
    config = _light_config(ensemble_runs=3)
    light = apply_light_mode(config)
    review = light.review
    assert review.agentic_tools is False
    assert review.security_agentic is False
    assert review.context_token_budget == 2_000
    assert review.jit_java_go is False
    assert review.overlap.enabled is False
    assert review.walkthrough_sequence_diagram is False
    assert review.ensemble_runs == 1
    # Passes that stay on.
    assert review.security_pass is True
    assert review.self_critique is True
    assert review.walkthrough is True
    assert review.include_summary is True


def test_does_not_mutate_caller_and_is_idempotent():
    config = _light_config(ensemble_runs=3)
    light = apply_light_mode(config)
    assert light is not config
    assert config.review.agentic_tools is True
    assert config.review.ensemble_runs == 3
    assert config.review.overlap.enabled is True
    again = apply_light_mode(light)
    assert again.review.model_dump() == light.review.model_dump()


def test_engine_normalizes_at_startup():
    config = _light_config()
    llm = MagicMock(spec=LLMProvider)
    engine = ReviewEngine(config=config, llm=llm)
    assert engine.config.review.agentic_tools is False
    assert engine.config.review.context_token_budget == 2_000
    assert engine.config.review.overlap.enabled is False
    # Caller's object untouched.
    assert config.review.agentic_tools is True


def test_walkthrough_prompt_drops_diagram_in_light_mode():
    light = apply_light_mode(_light_config())
    messages = build_walkthrough_prompt(_walkthrough_files(), light, pr_title="t")
    assert "graph LR" not in messages[0]["content"]

    default = MiraConfig()
    messages = build_walkthrough_prompt(_walkthrough_files(), default, pr_title="t")
    assert "graph LR" in messages[0]["content"]


def _mock_llm(sample_response: str) -> MagicMock:
    llm = MagicMock(spec=LLMProvider)
    llm.review = AsyncMock(return_value=sample_response)
    llm.walkthrough = AsyncMock(return_value='{"summary": "s", "change_groups": []}')
    llm.complete = AsyncMock(return_value=sample_response)
    llm.count_tokens = MagicMock(return_value=100)
    return llm


def _review_config(light: bool) -> MiraConfig:
    config = MiraConfig()
    config.review.light_mode = light
    config.review.code_context = False
    config.review.security_pass = False
    config.review.include_summary = False
    config.review.walkthrough = False
    return config


@pytest.mark.asyncio
async def test_file_history_skipped_in_light_mode(
    sample_diff_text: str, sample_llm_response_text: str
):
    engine = ReviewEngine(
        config=_review_config(light=True), llm=_mock_llm(sample_llm_response_text)
    )
    engine._pr_info = SimpleNamespace(owner="o", repo="r", platform="github")
    engine.provider = AsyncMock()
    await engine.review_diff(sample_diff_text)
    engine.provider.get_file_history.assert_not_called()


@pytest.mark.asyncio
async def test_file_history_fetched_by_default(
    sample_diff_text: str, sample_llm_response_text: str
):
    engine = ReviewEngine(
        config=_review_config(light=False), llm=_mock_llm(sample_llm_response_text)
    )
    engine._pr_info = SimpleNamespace(owner="o", repo="r", platform="github")
    engine.provider = AsyncMock()
    engine.provider.get_file_history = AsyncMock(return_value={})
    await engine.review_diff(sample_diff_text)
    engine.provider.get_file_history.assert_called_once()
