"""Light review mode — normalization bundle and engine wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mira.config import MiraConfig, apply_light_mode
from mira.core.engine import ReviewEngine
from mira.llm.prompts.review import build_review_prompt, build_walkthrough_prompt
from mira.llm.provider import LLMProvider
from mira.models import FileChangeType, FileDiff, HunkInfo, PRInfo


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


# Markers for the repo/PR-derived context sections in review.jinja2. Light
# mode must render none of these; the static instruction sections stay.
_CONTEXT_MARKERS = (
    "## Already addressed in earlier review rounds",
    "## Team coding conventions",
    "## Already Flagged Issues",
    "## Custom Review Rules",
    "## Team Preferences (Learned from Feedback)",
    "## File History",
)


def _review_files() -> list[FileDiff]:
    return [
        FileDiff(
            path="src/a.py",
            change_type=FileChangeType.MODIFIED,
            language="python",
            added_lines=1,
            deleted_lines=0,
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=1,
                    target_start=1,
                    target_length=2,
                    content="@@ -1 +1,2 @@\n x = 1\n+y = 2",
                )
            ],
        )
    ]


def test_light_prompt_has_no_context_sections():
    """All-empty context (exactly what the engine passes in light mode)."""
    messages = build_review_prompt(
        _review_files(),
        _light_config(),
        pr_title="Add y",
        existing_comments=None,
        code_context="",
        learned_rules=None,
        custom_rules=None,
        file_history=None,
        review_round=1,
        resolved_threads=None,
        team_conventions="",
    )
    system, user = messages[0]["content"], messages[1]["content"]
    for marker in _CONTEXT_MARKERS:
        assert marker not in system
    # Static instructions and the diff itself stay.
    assert "## Diffs to Review" in system
    assert "## Common Footguns" in system
    assert "## Pull Request" in system
    assert "You only see changed code segments" in system
    assert "+y = 2" in user


def test_light_prompt_keeps_round_behavior():
    """Round 2+ strictness still engages; only its evidence is gone."""
    messages = build_review_prompt(
        _review_files(), _light_config(), review_round=2, resolved_threads=None
    )
    system = messages[0]["content"]
    assert "## This is a follow-up review" in system
    assert "## Already addressed in earlier review rounds" not in system


def _mock_pr_provider(sample_diff_text: str) -> AsyncMock:
    provider = AsyncMock()
    provider.get_pr_info.return_value = PRInfo(
        title="Test PR",
        description="",
        base_branch="main",
        head_branch="feature",
        url="https://github.com/o/r/pull/1",
        number=1,
        owner="o",
        repo="r",
    )
    provider.get_pr_diff.return_value = sample_diff_text
    provider.get_all_bot_threads.return_value = []
    provider.get_unresolved_bot_threads.return_value = []
    return provider


def _review_messages(mock_llm: MagicMock) -> str:
    """Joined prompt text of the first main-pass review call."""
    messages = mock_llm.review.call_args[0][0]
    return "\n".join(m["content"] for m in messages)


@pytest.mark.asyncio
async def test_light_review_pr_sends_diff_only(
    sample_diff_text: str, sample_llm_response_text: str
):
    config = _review_config(light=True)
    config.review.agentic_tools = False
    engine = ReviewEngine(config=config, llm=_mock_llm(sample_llm_response_text))
    engine.provider = _mock_pr_provider(sample_diff_text)
    with (
        patch("mira.core.engine.IndexStore") as mock_store_cls,
        patch("mira.core.engine.build_code_context") as mock_ctx,
        patch("mira.dashboard.api._app_db") as mock_db,
    ):
        await engine.review_pr("https://github.com/o/r/pull/1")

    # No context I/O happened (the store still opens for recording writes).
    store = mock_store_cls.open.return_value
    store.get_learned_rules_text.assert_not_called()
    store.list_review_context.assert_not_called()
    store.get_summaries.assert_not_called()
    store.get_all_review_context_text.assert_not_called()
    mock_ctx.assert_not_called()
    mock_db.get_repo.assert_not_called()
    # …and none reached the model.
    prompt = _review_messages(engine.llm)
    for marker in _CONTEXT_MARKERS:
        assert marker not in prompt
    assert "## Diffs to Review" in prompt
    assert "You only see changed code segments" in prompt


@pytest.mark.asyncio
async def test_default_review_pr_includes_context(
    sample_diff_text: str, sample_llm_response_text: str
):
    config = _review_config(light=False)
    config.review.code_context = True
    config.review.agentic_tools = False
    engine = ReviewEngine(config=config, llm=_mock_llm(sample_llm_response_text))
    engine.provider = _mock_pr_provider(sample_diff_text)
    store = MagicMock()
    store.get_learned_rules_text.return_value = ["Always log errors."]
    store.list_review_context.return_value = []
    store.get_all_review_context_text.return_value = ""
    store.get_summaries.return_value = {}
    store.all_paths.return_value = []
    with (
        patch("mira.core.engine.IndexStore") as mock_store_cls,
        patch(
            "mira.core.engine.build_code_context", AsyncMock(return_value="CTX-BLOCK")
        ) as mock_ctx,
        patch("mira.dashboard.api._app_db") as mock_db,
    ):
        mock_store_cls.open.return_value = store
        mock_db.get_repo.return_value = SimpleNamespace(conventions="Conventions!")
        mock_db.get_global_rules_text.return_value = []
        await engine.review_pr("https://github.com/o/r/pull/1")

    mock_store_cls.open.assert_called()
    mock_ctx.assert_called()
    mock_db.get_repo.assert_called()
    prompt = _review_messages(engine.llm)
    assert "## Team coding conventions" in prompt
    assert "## Team Preferences (Learned from Feedback)" in prompt
    assert "CTX-BLOCK" in prompt
