"""End-to-end checks of the brief's contract, using mock LLM + sample catalogue."""
from conftest import ask


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["db"] is True and body["products"] > 0


def test_cors_allows_react_dev_server(client):
    r = client.options("/chat", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
                                         "Access-Control-Request-Headers": "content-type"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    other = client.options("/chat", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in other.headers


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


def test_product_list_for_catalogue_page(client):
    r = client.get("/products", params={"max_price": 3000, "in_stock": True})
    assert r.status_code == 200
    items = r.json()
    assert items and all(p["price_inr"] <= 3000 and p["stock"] > 0 for p in items)
    assert client.get("/products", params={"limit": 2}).json().__len__() <= 2


def test_ingest_returns_a_replayable_trace(client):
    r = client.post("/ingest", json=[{"product_id": "P-9911", "name": "Trace Test Bottle", "category": "kitchen",
                                      "price_inr": 499, "stock": 5}, {"name": "no id"}]).json()
    assert r["created"] + r["updated"] == 1 and len(r["errors"]) == 1 and r["trace_id"]
    rec = client.get(f"/trace/{r['trace_id']}").json()["record"]
    assert rec["kind"] == "ingest" and rec["changed_ids"] == ["P-9911"] and rec["rows"] == 2


def test_vague_request_with_no_product_asks_what_they_want(client):
    body = ask(client, "suggest something good", session="vague2")
    assert body["answer_type"] == "clarification" and body["follow_up_options"]
    assert not body["products"]


def test_clarifying_question_comes_with_one_tap_answers(client):
    body = ask(client, "I need a laptop", session="vague3")
    assert body["answer_type"] == "clarification" and body["follow_up_options"]
