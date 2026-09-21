"""The activity feed captures PR author + avatar, and the per-PR detail
endpoint returns review passes (with their comments + reviewed files) plus
human replies for the conversation timeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from mira.dashboard import api
from mira.dashboard.db import AppDatabase
from mira.dashboard.routers import core
from mira.index.store import IndexStore


@pytest.fixture
def patched_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppDatabase:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    db = AppDatabase(url="", admin_password="admin")
    monkeypatch.setattr(api, "_app_db", db)
    return db


def _seed(db: AppDatabase) -> None:
    db.register_repo("acme", "web")
    store = IndexStore.open("acme", "web")

    # Two review passes on PR #1.
    r1 = store.record_review(
        pr_number=1,
        pr_title="Fix auth",
        pr_url="https://github.com/acme/web/pull/1",
        comments_posted=2,
        blockers=1,
        warnings=1,
        suggestions=0,
        categories="bug,security",
        created_at=100.0,
        author="octocat",
        author_avatar_url="https://avatars.example/octocat.png",
        reviewed_paths=json.dumps(["src/auth.ts", "src/session.ts"]),
    )
    store.add_review_comments(
        r1.id,
        1,
        "https://github.com/acme/web/pull/1",
        [
            {
                "path": "src/auth.ts",
                "line": 42,
                "severity": "blocker",
                "category": "security",
                "title": "Token not validated",
                "body": "Validate the JWT before trusting claims.",
            },
            {
                "path": "src/session.ts",
                "line": 10,
                "severity": "warning",
                "category": "bug",
                "title": "Off-by-one",
                "body": "Loop bound.",
            },
        ],
    )
    store.record_review(
        pr_number=1,
        pr_title="Fix auth",
        pr_url="https://github.com/acme/web/pull/1",
        comments_posted=1,
        blockers=0,
        warnings=1,
        suggestions=0,
        categories="bug",
        created_at=300.0,
        author="octocat",
        author_avatar_url="https://avatars.example/octocat.png",
        reviewed_paths=json.dumps(["src/auth.ts"]),
    )

    # A human reply on the PR.
    store.record_reply(
        pr_number=1,
        pr_url="https://github.com/acme/web/pull/1",
        author="alice",
        author_avatar_url="https://avatars.example/alice.png",
        body="Good catch, fixed.",
        comment_path="src/auth.ts",
        comment_line=42,
        in_reply_to_id=999,
        created_at=200.0,
    )
    store.close()


def test_activity_list_includes_author(patched_db: AppDatabase):
    _seed(patched_db)
    out = core.list_activity()
    assert out.events
    ev = out.events[0]
    assert ev.author_username == "octocat"
    assert ev.author_avatar_url == "https://avatars.example/octocat.png"


def test_activity_detail_returns_timeline(patched_db: AppDatabase):
    _seed(patched_db)
    detail = api.get_activity_detail("acme", "web", 1)

    assert detail.author_username == "octocat"
    assert len(detail.reviews) == 2
    # Comments are attached to the right pass, and reviewed paths are parsed.
    first = next(r for r in detail.reviews if r.created_at == 100.0)
    assert len(first.comments) == 2
    assert first.comments[0].body.startswith("Validate the JWT")
    assert first.reviewed_paths == ["src/auth.ts", "src/session.ts"]
    # Human reply captured for the timeline.
    assert len(detail.replies) == 1
    assert detail.replies[0].author == "alice"
    assert detail.replies[0].body == "Good catch, fixed."


def test_activity_detail_404_for_unknown_pr(patched_db: AppDatabase):
    _seed(patched_db)
    with pytest.raises(HTTPException) as exc:
        api.get_activity_detail("acme", "web", 999)
    assert exc.value.status_code == 404


def test_token_split_round_trips_to_detail_and_stats(patched_db: AppDatabase):
    patched_db.register_repo("acme", "web")
    store = IndexStore.open("acme", "web")
    store.record_review(
        pr_number=2,
        pr_title="Split",
        pr_url="https://github.com/acme/web/pull/2",
        comments_posted=0,
        blockers=0,
        warnings=0,
        tokens_used=150,
        prompt_tokens=100,
        completion_tokens=50,
        cached_tokens=20,
        created_at=400.0,
    )
    store.close()

    detail = api.get_activity_detail("acme", "web", 2)
    (review,) = detail.reviews
    assert (review.prompt_tokens, review.completion_tokens, review.cached_tokens) == (100, 50, 20)

    store = IndexStore.open("acme", "web")
    try:
        stats = store.get_review_stats()
    finally:
        store.close()
    assert stats["total_prompt_tokens"] == 100
    assert stats["total_completion_tokens"] == 50
    assert stats["total_cached_tokens"] == 20


def test_token_split_columns_migrated_on_open(patched_db: AppDatabase, tmp_path: Path):
    import sqlite3

    patched_db.register_repo("acme", "web")
    store = IndexStore.open("acme", "web")
    store.record_review(
        pr_number=3,
        pr_title="Legacy",
        pr_url="https://github.com/acme/web/pull/3",
        comments_posted=0,
        blockers=0,
        warnings=0,
        tokens_used=90,
        created_at=500.0,
    )
    store.close()

    # Simulate a pre-split database by dropping the new columns.
    db_path = tmp_path / "acme" / "web.db"
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE review_events DROP COLUMN prompt_tokens")
    conn.execute("ALTER TABLE review_events DROP COLUMN completion_tokens")
    conn.execute("ALTER TABLE review_events DROP COLUMN cached_tokens")
    conn.commit()
    conn.close()

    # Reopening migrates; the legacy row reads back with a zero split.
    store = IndexStore.open("acme", "web")
    try:
        (legacy,) = store.list_review_events_for_pr(3)
        assert legacy.tokens_used == 90
        assert (legacy.prompt_tokens, legacy.completion_tokens, legacy.cached_tokens) == (0, 0, 0)
        store.record_review(
            pr_number=3,
            pr_title="Legacy",
            pr_url="https://github.com/acme/web/pull/3",
            comments_posted=0,
            blockers=0,
            warnings=0,
            tokens_used=150,
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=20,
            created_at=600.0,
        )
        rows = store.list_review_events_for_pr(3)
        assert len(rows) == 2
    finally:
        store.close()
