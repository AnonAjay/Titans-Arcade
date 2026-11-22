import json
from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

import server.app as app_module
from shared.station_client import StationNetworkClient


def test_station_client_offline_fallback(tmp_path):
    outbox = tmp_path / "outbox.jsonl"
    # Unreachable port
    client = StationNetworkClient(
        station="A",
        server_url="http://127.0.0.1:59999",
        outbox_path=outbox,
        poll_interval_s=0.05,
    )
    assert client.is_online is False
    assert client.pop_next_player() is None

    # Start and stop after brief moment
    client.start()
    time.sleep(0.1)
    client.stop(timeout_s=0.5)
    assert client.is_online is False


def test_station_client_sync_and_drain_outbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outbox = tmp_path / "outbox.jsonl"
    score_data = {
        "event_id": "TEST-SCORE-01",
        "player_id": "P001",
        "station": "A",
        "game": "dodge",
        "score": 1500,
        "accuracy": 0.85,
        "best_streak": 4,
        "duration_s": 30.0,
        "finished_at": "2026-11-20T00:00:00Z",
    }
    outbox.write_text(json.dumps(score_data) + "\n", encoding="utf-8")

    client = StationNetworkClient(
        station="A",
        server_url="http://testserver",
        outbox_path=outbox,
        poll_interval_s=0.1,
    )

    # Mock _http_request directly for test
    def mock_http(method, endpoint, data=None):
        if endpoint == "/api/score":
            return {"accepted": True, "duplicate": False}
        return None

    client._http_request = mock_http
    client._sync_outbox_locked()

    # Outbox must be drained!
    remaining = outbox.read_text(encoding="utf-8").strip()
    assert remaining == ""


def test_station_client_pop_next_player():
    client = StationNetworkClient(station="B", server_url="http://testserver")
    assert client.pop_next_player() is None

    # Inject mock pending player
    client._pending_player = {"player_id": "P002", "name": "Rohan", "height_cm": 180.0}
    popped = client.pop_next_player()
    assert popped is not None
    assert popped["player_id"] == "P002"
    assert popped["name"] == "Rohan"

    # Second pop is None (consumed atomically)
    assert client.pop_next_player() is None


@pytest.mark.xfail(
    strict=True,
    reason="known defect: malformed outbox lines are silently discarded",
)
def test_station_client_retains_malformed_outbox_line_for_recovery(tmp_path):
    """A bad line must remain available for diagnosis instead of disappearing."""
    outbox = tmp_path / "outbox.jsonl"
    malformed_line = "{\"event_id\": \"A-P009-1\",\"score\":"
    outbox.write_text(malformed_line + "\n", encoding="utf-8")

    client = StationNetworkClient(
        station="A",
        server_url="http://testserver",
        outbox_path=outbox,
    )
    client._http_request = lambda method, endpoint, data=None: None
    client._sync_outbox_locked()

    assert outbox.read_text(encoding="utf-8") == malformed_line + "\n"
