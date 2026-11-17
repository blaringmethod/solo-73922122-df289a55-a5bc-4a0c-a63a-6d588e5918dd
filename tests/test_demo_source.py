"""The demo OAI source supports the panel and the mid-pagination fault."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.oai import parse_list_records


def test_panel_renders_and_settings_save(demo_app, repo):
    with TestClient(demo_app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "翻页期间变更" in page.text
        assert "sourceState" in page.text

        response = client.post(
            "/demo/settings",
            data={
                "page_size": 2,
                "slow_seconds": 0,
                "token_failures": 0,
                "token_ttl_seconds": 3600,
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert repo.get_meta("page_size") == "2"


def test_every_list_records_page_labelled_with_stable_source_state(demo_app, repo):
    repo.set_meta("page_size", "2")
    with TestClient(demo_app) as client:
        first = parse_list_records(
            client.get(
                "/oai", params={"verb": "ListRecords", "metadataPrefix": "oai_dc"}
            ).content
        )
        assert first.source_state
        assert first.token
        assert first.token_state == first.source_state

        second = parse_list_records(
            client.get(
                "/oai", params={"verb": "ListRecords", "resumptionToken": first.token}
            ).content
        )
        # Same window, no change in between: same snapshot, same total.
        assert second.source_state == first.source_state
        assert second.complete_list_size == first.complete_list_size


def test_armed_mutation_rejects_stale_token_and_keeps_size(demo_app, repo):
    repo.set_meta("page_size", "2")
    with TestClient(demo_app) as client:
        first = parse_list_records(
            client.get(
                "/oai", params={"verb": "ListRecords", "metadataPrefix": "oai_dc"}
            ).content
        )
        size_before = first.complete_list_size

        response = client.post(
            "/demo/pagination-mutation",
            data={"mode": "auto", "repeat": 1, "late_title": "迟到", "correct_title": ""},
            follow_redirects=False,
        )
        assert response.status_code == 303

        stale = client.get(
            "/oai", params={"verb": "ListRecords", "resumptionToken": first.token}
        ).content.decode()
        assert "badResumptionToken" in stale

    # One member moved out of the window, one late member moved in: the total
    # is unchanged, which is why the token carries a content-based snapshot.
    with TestClient(demo_app) as client:
        fresh = parse_list_records(
            client.get(
                "/oai", params={"verb": "ListRecords", "metadataPrefix": "oai_dc"}
            ).content
        )
    assert fresh.complete_list_size == size_before
    assert fresh.source_state != first.source_state
    assert repo.state()["pagination_mutation_pending"] is False
