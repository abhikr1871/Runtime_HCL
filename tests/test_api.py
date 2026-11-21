"""End-to-end checks of the brief's contract, using mock LLM + sample catalogue."""
from conftest import ask


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["db"] is True and body["products"] > 0


def test_budget_search_respects_constraint(client):
    body = ask(client, "gift for my father who likes trekking under 3000", session="budget")
    assert body["answer_type"] == "recommendation"
    assert body["products"], body
    assert all(p["price"] <= 3000 for p in body["products"])
    assert body["trace_id"]


def test_vague_question_gets_clarification(client):
    body = ask(client, "I need a laptop", session="vague")
    assert body["answer_type"] == "clarification"
    assert body["follow_up_question"]


def test_not_in_catalogue(client):
    body = ask(client, "Do you sell electric cars?", session="nf")
    assert body["answer_type"] == "not_found"


def test_follow_up_cheaper(client):
    first = ask(client, "bluetooth speaker", session="cheap")
    assert first["answer_type"] == "recommendation"
    cheapest = min(p["price"] for p in first["products"])
    second = ask(client, "show me cheaper ones", session="cheap")
    assert all(p["price"] < cheapest for p in second["products"])
    if not second["products"]:  # nothing cheaper exists: must say so and offer the nearest options
        assert second["answer_type"] == "not_found"


def test_compare_first_two(client):
    ask(client, "trekking", session="cmp")
    body = ask(client, "compare the first two", session="cmp")
    assert body["answer_type"] == "comparison"
    assert len(body["comparison"]["rows"]) == 2


def test_trace_is_replayable(client):
    body = ask(client, "running shoes", session="trace")
    r = client.get(f"/trace/{body['trace_id']}")
    assert r.status_code == 200
    assert r.json()["record"]["response"]["trace_id"] == body["trace_id"]


def test_ingest_then_searchable(client):
    new = {"product_id": "P-9100", "name": "Yoga Mat Pro", "category": "sports-accessories",
           "brand": "Decathlon", "price_inr": 999, "stock": 10, "rating": 4.5, "tags": ["yoga", "mat"]}
    r = client.post("/ingest", json=[new])
    assert r.status_code == 200 and r.json()["created"] == 1
    body = ask(client, "yoga mat", session="ingest")
    assert any(p["product_id"] == "P-9100" for p in body["products"])
    assert client.get("/products/P-9100").status_code == 200
