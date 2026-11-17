"""The window changes between pages while its total size stays the same.

Scenario from the bug report: between page 1 and page 2 the source corrects
one record to a datestamp outside the window and adds one late record that is
inside the window.  ``completeListSize`` is still 4, so a size-only check sees
nothing; a cursor-based continuation would skip the record that shifted from
page 2 to page 1.  The harvester must notice the window's members/content
changed, discard the whole pagination, and re-request the original closed
window from page 1.  Only a clean re-fetch may publish catalogue + watermark.
"""

from __future__ import annotations

import asyncio
from urllib.parse import parse_qsl, urlparse

import httpx

from app.harvest import HarvestConfig, Harvester
from app.oai import OaiClient, OaiPage, OaiRecord
from conftest import wait_for_outcome


class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, app) -> None:
        self._inner = httpx.ASGITransport(app=app)
        self.requests: list[dict[str, str]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(dict(parse_qsl(urlparse(str(request.url)).query)))
        return await self._inner.handle_async_request(request)


def make_harvester(store, app, *, max_attempts: int = 3):
    transport = RecordingTransport(app)

    def factory() -> OaiClient:
        return OaiClient("http://demo.local/oai", transport=transport)

    harvester = Harvester(
        store,
        HarvestConfig(
            oai_url="http://demo.local/oai",
            max_attempts=max_attempts,
            retry_backoff_ms=0,
        ),
        client_factory=factory,
    )
    return harvester, transport


def _two_page_window(repo, store):
    """Publish a baseline, then make the next window need exactly two pages."""
    repo.set_meta("page_size", "3")
    repo.set_meta("token_ttl_seconds", "3600")


async def test_same_size_member_change_discards_pages_and_refetches(store, demo_app, repo):
    harvester, wire = make_harvester(store, demo_app)
    first = await harvester.start_run()
    assert await wait_for_outcome(first) == "succeeded"

    await asyncio.sleep(1.05)
    # Four records in the next window: page 1 = two, page 2 = two.  Moving one
    # out and adding one late record keeps the size at 4.
    for index in range(4):
        repo.add_record(identifier=f"oai:demo:inc{index}", title=f"增量 {index}")

    repo.arm_pagination_mutation({"auto": True, "repeat": 1})
    wire.requests.clear()

    second = await harvester.start_run()
    assert await wait_for_outcome(second) == "succeeded"

    run = store.get_run(second.run_id)
    assert run["token_restarts"] == 1
    assert "分页" in (run["last_error"] or "")
    # The window really did change members while keeping the same size.
    assert run["complete_list_size"] == 4

    # The retry re-requested the *original* closed window from page 1 with no
    # stale token.
    initial_lists = [
        r
        for r in wire.requests
        if r.get("verb") == "ListRecords" and "resumptionToken" not in r
    ]
    assert len(initial_lists) == 2
    assert initial_lists[0]["from"] == initial_lists[1]["from"]
    assert initial_lists[0]["until"] == initial_lists[1]["until"]
    assert initial_lists[1].get("from") == store.get_run(second.run_id)["window_from"]

    # Every record that belongs to the final window is published exactly once,
    # including the one the old cursor would have skipped.
    state = store.published_state()
    identifiers = {
        row["identifier"]
        for row in store.catalog_page(page_size=200)["records"]
    }
    expected = {
        row["identifier"]
        for row in repo._select_window(
            run["window_from"], run["window_until"]
        )
        if not row["deleted"]
    }
    assert expected <= identifiers
    # The record the old cursor would have skipped is published.
    assert "oai:demo:inc3" in identifiers
    # The late record is in the catalogue, the corrected-out one is untouched
    # by this round (it was already published by the baseline).
    late = [i for i in identifiers if i.startswith("oai:demo:late")]
    assert late, "late record inserted during paging must be published"
    assert "oai:demo:inc0" not in identifiers, (
        "the record corrected out of the window must not be published by this round"
    )
    assert state["watermark"] == run["window_until"]


async def test_member_change_on_every_retry_fails_without_publishing(store, demo_app, repo):
    harvester, wire = make_harvester(store, demo_app, max_attempts=3)
    first = await harvester.start_run()
    assert await wait_for_outcome(first) == "succeeded"
    baseline = store.published_state()
    baseline_items = store.catalog_page(page_size=200)["records"]

    await asyncio.sleep(1.05)
    for index in range(4):
        repo.add_record(identifier=f"oai:demo:inc{index}", title=f"增量 {index}")

    # Each of the three attempts sees another one-out / one-in change.
    repo.arm_pagination_mutation({"auto": True, "repeat": 3})

    second = await harvester.start_run()
    assert await wait_for_outcome(second) == "failed"
    window_from = store.get_run(second.run_id)["window_from"]
    window_until = store.get_run(second.run_id)["window_until"]

    run = store.get_run(second.run_id)
    assert run["status"] == "failed"
    assert run["token_restarts"] == 3
    assert "重取" in (run["last_error"] or "")
    assert run["staged_count"] == 0

    after = store.published_state()
    assert after["watermark"] == baseline["watermark"]
    assert after["catalog_size"] == baseline["catalog_size"]
    assert after["success_count"] == baseline["success_count"]
    assert store.catalog_page(page_size=200)["records"] == baseline_items

    # One initial (page-1) request per attempt, always the same closed window.
    initial_lists = [
        r
        for r in wire.requests
        if r.get("verb") == "ListRecords"
        and "resumptionToken" not in r
        and r.get("from") == window_from
        and r["until"] == window_until
    ]
    assert len(initial_lists) == 3
    assert {(r["from"], r["until"]) for r in initial_lists} == {
        (run["window_from"], run["window_until"])
    }


async def test_page_and_token_must_share_one_source_state(store):
    """A page whose continuation token belongs to another snapshot is rejected."""

    class SplitStateClient:
        def __init__(self):
            self.calls = 0
            self._context = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def identify_time(self):
            return "2026-01-01T00:00:10Z"

        async def list_records(self, *, from_, until, token=None):
            self.calls += 1
            if self.calls == 1:
                return OaiPage(
                    records=[record("oai:s:1", "2026-01-01T00:00:01Z")],
                    token="t1",
                    complete_list_size=2,
                    source_state="state-A",
                    token_state="state-A",
                )
            if self.calls == 2:
                # Page content says B, but its token carries no marker: the
                # next cursor cannot be proven to belong to this content.
                return OaiPage(
                    records=[record("oai:s:2", "2026-01-01T00:00:02Z")],
                    token="t2",
                    complete_list_size=2,
                    source_state="state-B",
                    token_state=None,
                )
            # Clean retry.
            return OaiPage(
                records=[
                    record("oai:s:1", "2026-01-01T00:00:01Z"),
                    record("oai:s:2", "2026-01-01T00:00:02Z"),
                ],
                complete_list_size=2,
                source_state="state-B",
                token_state=None,
            )

    client = SplitStateClient()
    harvester = Harvester(
        store,
        HarvestConfig(oai_url="scripted", max_attempts=3, retry_backoff_ms=0),
        client_factory=lambda: client,
    )
    control = await harvester.start_run()
    assert await wait_for_outcome(control) == "succeeded"
    run = store.get_run(control.run_id)
    assert run["token_restarts"] == 1
    assert "源状态" in (run["last_error"] or "")
    assert {r["identifier"] for r in store.catalog_page(page_size=10)["records"]} == {
        "oai:s:1",
        "oai:s:2",
    }


async def test_restart_mid_pagination_old_page_cannot_leak_into_new_round(
    store, demo_app, repo
):
    """Cancel + immediate restart while page 2 of the old round is in flight.

    The old round already staged page 1.  The new round must be a completely
    fresh pagination: no old token, no old staged record, and the published
    catalogue must be exactly the new round's clean snapshot.
    """
    baseline_harvester, _ = make_harvester(store, demo_app)
    baseline = await baseline_harvester.start_run()
    assert await wait_for_outcome(baseline) == "succeeded"
    baseline_watermark = store.published_state()["watermark"]

    await asyncio.sleep(1.05)
    for index in range(4):
        repo.add_record(identifier=f"oai:demo:inc{index}", title=f"增量 {index}")

    # The old round is slow, so it is still running at page 1 when we restart.
    repo.set_meta("slow_seconds", "0.4")
    repo.set_meta("page_size", "2")
    old_harvester, old_wire = make_harvester(store, demo_app)
    old = await old_harvester.start_run()
    # Identify (0.4s) + page 1 (0.4s) are done; page 2 is in flight.
    await asyncio.sleep(1.0)
    assert store.get_run(old.run_id)["status"] == "running"
    assert store.get_run(old.run_id)["pages_fetched"] >= 1
    assert any("resumptionToken" in r for r in old_wire.requests)

    # Cancel + immediate restart, and let the new round run against a fast,
    # stable source.  The old round's in-flight page was already requested.
    await old_harvester.cancel_run(old.run_id, "被新一轮取代", "superseded")
    repo.set_meta("slow_seconds", "0")
    new_harvester, new_wire = make_harvester(store, demo_app)
    new = await new_harvester.start_run()
    assert new.run_id != old.run_id
    assert await wait_for_outcome(new) == "succeeded"

    new_run = store.get_run(new.run_id)
    assert new_run["window_from"] == baseline_watermark
    assert new_run["staged_count"] == 0
    identifiers = {
        row["identifier"] for row in store.catalog_page(page_size=200)["records"]
    }
    assert {f"oai:demo:inc{i}" for i in range(4)} <= identifiers
    # No duplicates in the published catalogue.
    rows = store.catalog_page(page_size=200)["records"]
    assert len(rows) == len({row["identifier"] for row in rows})

    # Old round: cancelled, staging dropped, never published.
    old_run = store.get_run(old.run_id)
    assert old_run["status"] == "cancelled"
    assert old_run["staged_count"] == 0
    assert old_run["watermark_after"] is None

    # The new round's request was a fresh page 1 of the original closed window.
    first_new_list = next(
        r
        for r in new_wire.requests
        if r.get("verb") == "ListRecords" and "resumptionToken" not in r
    )
    assert first_new_list.get("from") == baseline_watermark
    assert first_new_list["until"] == new_run["window_until"]

    # Let any late old response unwind; nothing may change.
    await asyncio.sleep(1.5)
    assert store.published_state()["watermark"] == new_run["window_until"]
    assert store.get_run(old.run_id)["status"] == "cancelled"
    assert store.get_run(new.run_id)["status"] == "succeeded"


def record(identifier, datestamp, title="标题", deleted=False):
    return OaiRecord(
        identifier=identifier,
        datestamp=datestamp,
        deleted=deleted,
        title=None if deleted else title,
    )
