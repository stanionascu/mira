"""/api/activity flattens review events across all repos, newest first, and
supports repo + search filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from mira.core.review_status import ReviewTracker
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
    db.register_repo("acme", "api")

    web = IndexStore.open("acme", "web")
    web.record_review(
        pr_number=1,
        pr_title="Fix auth redirect loop",
        pr_url="https://github.com/acme/web/pull/1",
        comments_posted=3,
        blockers=1,
        warnings=2,
        suggestions=0,
        categories="bug,security",
        created_at=100.0,
    )
    web.record_review(
        pr_number=2,
        pr_title="Add dark mode",
        pr_url="https://github.com/acme/web/pull/2",
        comments_posted=1,
        blockers=0,
        warnings=0,
        suggestions=1,
        categories="style",
        created_at=300.0,
    )
    web.close()

    apirepo = IndexStore.open("acme", "api")
    apirepo.record_review(
        pr_number=7,
        pr_title="Rate limit login endpoint",
        pr_url="https://github.com/acme/api/pull/7",
        comments_posted=2,
        blockers=1,
        warnings=1,
        suggestions=0,
        categories="security,performance",
        created_at=200.0,
    )
    apirepo.close()


def test_lists_all_repos_newest_first(patched_db: AppDatabase):
    _seed(patched_db)
    out = core.list_activity()

    assert [(e.repo, e.pr_number) for e in out.events] == [
        ("web", 2),  # created_at 300
        ("api", 7),  # created_at 200
        ("web", 1),  # created_at 100
    ]
    # owner/repo are attached to each flattened event
    assert all(e.owner == "acme" for e in out.events)
    assert set(out.repos) == {"acme/web", "acme/api"}


def test_repo_filter_narrows_results(patched_db: AppDatabase):
    _seed(patched_db)
    out = core.list_activity(repo="acme/api")
    assert [e.pr_number for e in out.events] == [7]
    # repo dropdown still lists every repo, not just the filtered one
    assert set(out.repos) == {"acme/web", "acme/api"}


def test_search_matches_title_repo_and_category(patched_db: AppDatabase):
    _seed(patched_db)

    by_title = core.list_activity(q="dark")
    assert [e.pr_number for e in by_title.events] == [2]

    by_category = core.list_activity(q="security")
    assert {e.pr_number for e in by_category.events} == {1, 7}

    by_repo = core.list_activity(q="api")
    assert [e.pr_number for e in by_repo.events] == [7]


def test_search_ands_multiple_terms(patched_db: AppDatabase):
    _seed(patched_db)
    # "security" matches PRs 1 and 7; adding "web" narrows to just PR 1.
    out = core.list_activity(q="security web")
    assert [e.pr_number for e in out.events] == [1]


@pytest.fixture
def fresh_tracker(monkeypatch: pytest.MonkeyPatch) -> ReviewTracker:
    """Isolate the module-global review tracker for in-progress tests."""
    import mira.core.review_status as review_status

    tracker = ReviewTracker()
    monkeypatch.setattr(review_status, "tracker", tracker)
    return tracker


def test_in_progress_empty_by_default(patched_db: AppDatabase, fresh_tracker: ReviewTracker):
    _seed(patched_db)
    out = core.list_activity()
    assert out.in_progress == []


def test_in_progress_lists_active_reviews(patched_db: AppDatabase, fresh_tracker: ReviewTracker):
    _seed(patched_db)
    assert fresh_tracker.try_start(
        "acme/web", 9, "New checkout flow", "https://github.com/acme/web/pull/9"
    )
    out = core.list_activity()
    assert [(j.repo, j.pr_number) for j in out.in_progress] == [("web", 9)]
    job = out.in_progress[0]
    assert job.owner == "acme"
    assert job.pr_title == "New checkout flow"
    assert job.pr_url == "https://github.com/acme/web/pull/9"
    assert job.started_at > 0
    # Persisted events are untouched by the in-progress list.
    assert len(out.events) == 3


def test_in_progress_honors_repo_and_search_filters(
    patched_db: AppDatabase, fresh_tracker: ReviewTracker
):
    _seed(patched_db)
    fresh_tracker.try_start("acme/web", 9, "New checkout flow", "")
    fresh_tracker.try_start("acme/api", 10, "Rotate signing keys", "")

    assert [j.pr_number for j in core.list_activity(repo="acme/api").in_progress] == [10]
    assert [j.pr_number for j in core.list_activity(q="checkout").in_progress] == [9]
    assert [j.pr_number for j in core.list_activity(q="checkout api").in_progress] == []


def test_in_progress_drops_completed_reviews(patched_db: AppDatabase, fresh_tracker: ReviewTracker):
    _seed(patched_db)
    fresh_tracker.try_start("acme/web", 9, "New checkout flow", "")
    assert len(core.list_activity().in_progress) == 1
    fresh_tracker.complete("acme/web", 9)
    assert core.list_activity().in_progress == []
