"""API-layer tests: routes, validation, and serialisation shapes.

The LLM client factory is monkeypatched with a stub, so `create_session`'s
full path runs -- goal building, session construction, the opening -- without
a key or a network call. TestClient drives the async endpoints synchronously.
"""

import pytest
from fastapi.testclient import TestClient

import api.main as m
from llm.client import Usage
from llm.schemas import (
    AgentReply,
    Attachment,
    ExpansionResponse,
    FulfilledRequest,
    GeneratedArgument,
    GoalImpact,
    MeetingNotes,
    RoutedMessage,
    ScoredArgument,
)


class StubClient:
    def __init__(self):
        self.usage = Usage()

    async def parse(self, system_prompt, user_prompt, response_model, temperature=None):
        self.usage.add(1, 1)
        return {
            ScoredArgument: ScoredArgument(confidence=80, rationale="ok"),
            ExpansionResponse: ExpansionResponse(
                argument=GeneratedArgument(
                    text="gen", node_type="empirical", rationale="r",
                    goal_impacts=[
                        GoalImpact(goal_id="sdg_1", direction="positive"),
                        GoalImpact(goal_id="custom_1", direction="positive"),
                    ],
                )
            ),
            RoutedMessage: RoutedMessage(kind="argument", confidence=95),
            Attachment: Attachment(
                target_id="topic", relation_type="support", node_type="empirical",
                confidence=90, rationale="clear", restated="Restated.",
                goal_impacts=[GoalImpact(goal_id="custom_1", direction="positive")],
            ),
            FulfilledRequest: FulfilledRequest(
                argument=GeneratedArgument(text="req", node_type="empirical", rationale="r"),
                target_id="topic", relation_type="attack",
                placement_confidence=90, placement_rationale="bears there",
            ),
            AgentReply: AgentReply(message="Noted.", concedes=True, question="And?"),
            MeetingNotes: MeetingNotes(summary="Notes.", key_points=["[a1] x"]),
        }[response_model]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(m, "LLMClient", StubClient)
    m.SESSIONS.clear()
    m.SESSION_DOMAIN.clear()
    return TestClient(m.app)


GOALS = {"sdgs": [1, 11], "indicators": ["rough_sleeping_count"], "custom": ["families first"]}


def start(client, **overrides):
    body = {
        "persona_ids": ["homelessness_researcher"],
        "topic": "Test claim",
        "goals": GOALS,
        **overrides,
    }
    res = client.post("/api/session", json=body)
    assert res.status_code == 200, res.text
    return res.json()


# --- catalogue ----------------------------------------------------------------


def test_catalogue_serves_domains_indicators_and_sdgs(client):
    cat = client.get("/api/personas").json()
    assert [d["id"] for d in cat["domains"]] == ["climate", "barcelona_homelessness"]
    barcelona = cat["domains"][1]
    assert any(p["kind"] == "expert" for p in barcelona["partners"])
    assert all(d["partners"][-1]["kind"] == "contrarian" for d in cat["domains"])
    assert {i["id"] for i in barcelona["indicators"]} >= {"rough_sleeping_count", "mortality"}
    assert all(isinstance(i["sdg"], list) for i in barcelona["indicators"])
    assert len(cat["sdgs"]) == 17 and cat["sdgs"][0] == {"goal": 1, "name": "No Poverty"}
    assert [c["key"] for c in cat["confidence_levels"]] == [
        "unsure", "somewhat", "moderately", "very", "certain",
    ]
    assert cat["confidence_levels"][2]["tau"] == 0.5


# --- session creation ---------------------------------------------------------


def test_create_session_grounds_goals_and_opens_bare(client):
    """The start is neutral and costs nothing: just the claim at 0.5, the
    partner's spec-rendered introduction, the deliberation directed by the
    user from here."""
    state = start(client)
    assert [g["id"] for g in state["goals"]] == [
        "sdg_1", "sdg_11", "ind_rough_sleeping_count", "custom_1"
    ]
    assert [p["kind"] for p in state["personas"]] == ["expert"]
    assert [n["id"] for n in state["nodes"]] == ["topic"]
    assert state["stance"] == pytest.approx(0.5)  # neutral, not elicited
    assert state["turn"]["reply"] == ""
    assert [i["persona_id"] for i in state["turn"]["intros"]] == ["homelessness_researcher"]
    assert "Homelessness Researcher" in state["turn"]["intros"][0]["text"]
    assert state["usage"]["calls"] == 0
    assert state["turn"]["added_nodes"] == []

    # compact lens in state: all neutral, verdicts without contributions
    assert {e["id"] for e in state["lens"]} == {g["id"] for g in state["goals"]}
    assert all(e["verdict"] == "contested" for e in state["lens"])
    assert all("contributions" not in e for e in state["lens"])


def test_expand_carries_judged_tags(client):
    state = start(client)
    sid = state["session_id"]
    res = client.post(f"/api/session/{sid}/expand",
                      json={"node_id": "topic", "relation_type": "attack"}).json()
    node = next(n for n in res["nodes"] if n["text"] == "gen")
    assert "citation" not in node
    assert {t["goal_id"] for t in node["goal_impacts"]} == {"sdg_1", "custom_1"}
    assert all(t["source"] == "judged" for t in node["goal_impacts"])


@pytest.mark.parametrize("goals,detail", [
    ({"sdgs": [99]}, "unknown SDG"),
    ({"indicators": ["nope"]}, "unknown indicator"),
])
def test_create_session_rejects_unknown_goals(client, goals, detail):
    res = client.post("/api/session", json={
        "persona_ids": ["city_hall_politician"], "topic": "T", "goals": goals,
    })
    assert res.status_code == 400 and detail in res.json()["detail"]


def test_create_session_rejects_unknown_partner_and_domain(client):
    assert client.post("/api/session", json={
        "persona_ids": ["ghost"], "topic": "T"}).status_code == 400
    assert client.post("/api/session", json={
        "domain_id": "atlantis", "persona_ids": ["x"], "topic": "T"}).status_code == 400


def test_duplicate_goal_selection_is_deduplicated(client):
    """sdgs=[1,1] must not ground two identical virtual roots -- that
    crashed every subsequent serialise before build_goals deduped."""
    state = start(client, goals={
        "sdgs": [1, 1], "indicators": ["mortality", "mortality"],
        "custom": ["families first", "families first", "  "],
    })
    assert [g["id"] for g in state["goals"]] == ["sdg_1", "ind_mortality", "custom_1"]
    sid = state["session_id"]
    assert client.get(f"/api/session/{sid}").status_code == 200


def test_overlong_custom_goal_is_rejected(client):
    res = client.post("/api/session", json={
        "persona_ids": ["city_hall_politician"], "topic": "T",
        "goals": {"custom": ["x" * 201]},
    })
    assert res.status_code == 400 and "custom goal too long" in res.json()["detail"]


def test_goalless_session_still_works(client):
    state = start(client, goals={"sdgs": [], "indicators": [], "custom": []})
    assert state["goals"] == [] and state["lens"] == []
    sid = state["session_id"]
    assert client.get(f"/api/session/{sid}/lens").json() == {
        "topic": "Test claim", "lens": [], "history": {},
    }


# --- lens and retag -----------------------------------------------------------


def test_lens_endpoint_returns_contributions_and_history(client):
    sid = start(client)["session_id"]
    client.post(f"/api/session/{sid}/expand",
                json={"node_id": "topic", "relation_type": "attack"})
    full = client.get(f"/api/session/{sid}/lens").json()
    entry = next(e for e in full["lens"] if e["id"] == "sdg_1")
    assert entry["contributions"] and "basis" not in entry
    # two snapshots (opening + expansion): every trajectory has length 2
    assert all(len(v) == 2 for v in full["history"].values())


def test_retag_round_trip_and_validation(client):
    sid = start(client)["session_id"]
    state = client.post(f"/api/session/{sid}/expand",
                        json={"node_id": "topic", "relation_type": "support"}).json()
    node = next(n for n in state["nodes"] if n["text"] == "gen")

    res = client.put(f"/api/session/{sid}/node/{node['id']}/tags", json={
        "tags": [{"goal_id": "custom_1", "direction": "negative"}]})
    assert res.status_code == 200
    updated = next(n for n in res.json()["nodes"] if n["id"] == node["id"])
    assert updated["goal_impacts"] == [
        {"goal_id": "custom_1", "direction": "negative", "source": "user"}
    ]
    lens = {e["id"]: e for e in res.json()["lens"]}
    assert lens["custom_1"]["verdict"] == "undermined"

    assert client.put(f"/api/session/{sid}/node/{node['id']}/tags", json={
        "tags": [{"goal_id": "nope", "direction": "negative"}]}).status_code == 400
    assert client.put(f"/api/session/{sid}/node/ghost/tags", json={
        "tags": []}).status_code == 404


# --- turns --------------------------------------------------------------------


def land(client, sid, text, confidence=None):
    """The UI's two calls: the reading comes back pending, then it is applied."""
    first = client.post(f"/api/session/{sid}/contribute", json={"text": text}).json()
    assert first["turn"]["pending"] is not None
    assert first["turn"]["pending"]["provenance"] == "user"
    return client.post(
        f"/api/session/{sid}/contribute",
        json={"text": text, "attachment": first["turn"]["pending"], "confidence": confidence},
    )


def test_contribute_lands_the_users_point_silently(client):
    sid = start(client)["session_id"]
    res = land(client, sid, "my point").json()
    assert res["turn"]["reply"] == "" and res["turn"]["question"] == ""
    assert res["turn"]["speaker"] is None
    node = next(n for n in res["nodes"] if n["text"] == "Restated.")
    assert node["goal_impacts"][0]["goal_id"] == "custom_1"
    assert node["base_score"] == 0.5 and node["rationale"] == ""
    assert node["author"] is None and node["author_name"] is None
    assert res["turn"]["added_nodes"] == [node["id"]]
    assert [s["label"] for s in res["turn"]["steps"]] == ["your argument"]
    assert res["usage"]["calls"] == 2                      # router + placement only


def test_right_click_target_and_persona_expand(client):
    sid = start(client, persona_ids=["homelessness_researcher", "devils_advocate"])["session_id"]
    first = client.post(f"/api/session/{sid}/contribute", json={"text": "my point", "target_id": "topic"}).json()
    assert first["turn"]["pending"]["target_id"] == "topic"
    landed = client.post(f"/api/session/{sid}/contribute",
                         json={"text": "my point", "attachment": first["turn"]["pending"], "confidence": "very"}).json()
    user_id = landed["turn"]["added_nodes"][0]
    assert client.post(f"/api/session/{sid}/contribute", json={"text": "x", "target_id": "ghost"}).status_code == 404

    ok = client.post(f"/api/session/{sid}/expand",
                     json={"node_id": user_id, "relation_type": "attack", "persona_id": "devils_advocate"}).json()
    node = next(n for n in ok["nodes"] if n["id"] == ok["turn"]["added_nodes"][0])
    assert node["author"] == "devils_advocate" and node["author_name"] == "Devil's Advocate"
    assert node["parent_id"] == user_id and ok["turn"]["speaker"] == "devils_advocate"
    bad = client.post(f"/api/session/{sid}/expand",
                      json={"node_id": user_id, "relation_type": "support", "persona_id": "devils_advocate"})
    assert bad.status_code == 400
    unknown = client.post(f"/api/session/{sid}/expand",
                          json={"node_id": user_id, "relation_type": "attack", "persona_id": "ghost"})
    assert unknown.status_code == 404

def test_contribute_takes_the_users_confidence(client):
    sid = start(client)["session_id"]
    res = land(client, sid, "my point", confidence="certain").json()
    node = next(n for n in res["nodes"] if n["text"] == "Restated.")
    assert node["base_score"] == 0.8

    bad = land(client, sid, "my point", confidence="totally")
    assert bad.status_code == 400


def test_contribute_force_request_fulfils(client):
    sid = start(client)["session_id"]
    res = client.post(f"/api/session/{sid}/contribute",
                      json={"text": "attack it", "force": "request"}).json()
    node = next(n for n in res["nodes"] if n["text"] == "req")
    assert node["provenance"] == "generated"

    bad = client.post(f"/api/session/{sid}/contribute",
                      json={"text": "x", "force": "nonsense"})
    assert bad.status_code == 422


def test_report_and_leverage_shapes(client):
    sid = start(client)["session_id"]

    # bare opening: leverage has nothing to rank yet
    assert client.get(f"/api/session/{sid}/leverage").json()["leverage"] == []

    client.post(f"/api/session/{sid}/expand",
                json={"node_id": "topic", "relation_type": "attack"})
    rep = client.get(f"/api/session/{sid}/report").json()
    assert {"lens", "lens_history", "agreements", "divergences",
            "trade_offs", "transcript"} <= set(rep)
    # two snapshots (opening + expansion): every trajectory has length 2
    assert all(len(v) == 2 for v in rep["lens_history"].values())

    lev = client.get(f"/api/session/{sid}/leverage").json()["leverage"]
    assert lev and {"id", "text", "sensitivity", "kind"} <= set(lev[0])


def test_rescore_and_drop_return_full_turn_shape(client):
    sid = start(client)["session_id"]
    state = client.post(f"/api/session/{sid}/expand",
                        json={"node_id": "topic", "relation_type": "attack"}).json()
    node = next(n for n in state["nodes"] if not n["is_topic"])

    res = client.patch(f"/api/session/{sid}/node/{node['id']}", json={"base_score": 0.1}).json()
    assert res["turn"]["delta"]["direction"] in ("strengthened", "weakened", "unchanged")
    assert res["turn"]["question"] == ""

    res = client.delete(f"/api/session/{sid}/node/{node['id']}").json()
    assert node["id"] not in [n["id"] for n in res["nodes"]]

    assert client.get("/api/session/missing").status_code == 404


def test_demo_gate_when_passphrase_set(client, monkeypatch):
    monkeypatch.setenv("DEMO_PASSPHRASE", "sesame")
    assert client.get("/api/personas").status_code == 401
    wrong = client.get("/api/personas", headers={"X-Demo-Passphrase": "nope"})
    assert wrong.status_code == 401
    ok = client.get("/api/personas", headers={"X-Demo-Passphrase": "sesame"})
    assert ok.status_code == 200
    # health stays open for platform liveness checks
    assert client.get("/api/health").status_code == 200


def test_demo_gate_inert_without_passphrase(client, monkeypatch):
    monkeypatch.delenv("DEMO_PASSPHRASE", raising=False)
    assert client.get("/api/personas").status_code == 200


def test_summary_endpoint_returns_notes_and_meta(client):
    sid = start(client)["session_id"]
    res = client.get(f"/api/session/{sid}/summary")
    assert res.status_code == 200
    body = res.json()
    assert body["notes"]["summary"] == "Notes."
    assert body["notes"]["key_points"] == ["[a1] x"]
    assert body["opening_stance"] == 0.5 and body["turns"] == 0
    assert {"topic", "personas", "stance", "history", "lens"} <= set(body)
