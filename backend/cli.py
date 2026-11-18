"""Interactive CLI for a debate session.

Exists to exercise the pipeline before any UI does, and to make the turn
operations concrete:

    /why <id>      ask the persona to support a node       -> one node appears
    /but <id>      ask it for a consideration against       -> one node appears
    /set <id> <v>  override a base score                    -> no LLM call
    /drop <id>     reject an argument outright              -> no LLM call
    /leverage      what would change its mind               -> no LLM call
    anything else  contribute an argument of your own

Run:  python cli.py [--persona ID] [--topic "..."]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from domains import DOMAINS, get_domain
from engine.lens import SDG_NAMES, Goal
from engine.attribution import DeltaExplanation
from llm.client import CLASSIFIER_MODEL, LLMClient, MissingAPIKey
from pipeline.session import CONFIDENCE_LEVELS, DEFAULT_CONFIDENCE, DebateSession

BAR_WIDTH = 24


def bar(value: float) -> str:
    filled = round(value * BAR_WIDTH)
    return "█" * filled + "·" * (BAR_WIDTH - filled)


def render_graph(session: DebateSession) -> str:
    strengths = session.strengths()
    qbaf = session.qbaf
    lines: list[str] = []

    def walk(node_id: str, indent: int, marker: str) -> None:
        arg = qbaf.arguments[node_id]
        tag = {"empirical": "EMP", "predictive": "PRD", "normative": "NRM"}[arg.node_type]
        origin = {"generated": " ", "user": "*", "document": "D"}[arg.provenance]
        lines.append(
            f"{'  ' * indent}{marker}{origin}[{node_id}] {tag} "
            f"τ={arg.base_score:.2f} σ={strengths[node_id]:.2f}  {arg.text}"
        )
        extras = []
        if arg.goal_impacts:
            tags = ",".join(
                f"{'↑' if t['direction'] == 'positive' else '↓'}{t['goal_id']}"
                for t in arg.goal_impacts
            )
            extras.append(f"[{tags}]")
        if extras:
            lines.append(f"{'  ' * indent}   ↳ {'  '.join(extras)}")
        for rel in qbaf.relations:
            if rel.target_id != node_id:
                continue
            walk(rel.source_id, indent + 1, "− " if rel.relation_type == "attack" else "+ ")

    walk(qbaf.topic_id, 0, "")
    return "\n".join(lines)


def render_delta(delta: DeltaExplanation) -> str:
    if abs(delta.topic_delta) < 1e-9:
        return "  (position unchanged)"
    return (
        f"  {delta.topic_before:.0%} → {delta.topic_after:.0%} "
        f"({delta.direction}, {delta.topic_delta:+.0%})"
    )


def ask_confidence() -> str:
    """The user's own confidence in the point they just typed, 1-5."""
    scale = "  ".join(f"{i}={label}" for i, (_, label, _) in enumerate(CONFIDENCE_LEVELS, 1))
    default = [k for k, _, _ in CONFIDENCE_LEVELS].index(DEFAULT_CONFIDENCE) + 1
    while True:
        raw = input(f"  how sure are you? [{scale}] (Enter={default}) ").strip()
        if not raw:
            return DEFAULT_CONFIDENCE
        if raw.isdigit() and 1 <= int(raw) <= len(CONFIDENCE_LEVELS):
            return CONFIDENCE_LEVELS[int(raw) - 1][0]
        print("  ! pick a number from the scale")


def show_stance(session: DebateSession) -> None:
    stance = session.stance()
    print(f"\n  stance  {bar(stance)}  {stance:.0%}")
    lens = session.lens()
    if lens:
        parts = " · ".join(
            f"{e['id']} {e['verdict']} ({e['sigma']:.2f})"
            for e in lens
        )
        print(f"  goals   {parts}")
    print()


async def run(domain_id: str, persona_id: str, topic: str, model: str | None) -> None:
    domain = get_domain(domain_id)
    personas = tuple(domain.get_partner(pid.strip()) for pid in persona_id.split(","))
    persona = personas[0]
    client = LLMClient(model=model) if model else LLMClient()
    # Default lens set for the CLI: the SDGs the domain's indicators map to.
    sdgs = sorted({n for ind in domain.indicators for n in ind.sdg})
    goals = tuple(
        Goal(id=f"sdg_{n}", label=f"SDG {n} · {SDG_NAMES[n]}", kind="sdg")
        for n in sdgs
    )
    session = DebateSession(
        personas=personas, topic=topic, client=client, goals=goals,
        classifier=LLMClient(model=CLASSIFIER_MODEL) if CLASSIFIER_MODEL else None,
    )

    print(f"\n  {persona.name}")
    print(f"  {persona.one_line}")
    print(f"  conservativeness {persona.conservativeness}")
    print(f"\n  topic: {topic}\n")

    result = await session.open()
    for pid, intro in result.intros:
        print(f"\n  {session.personas[pid].name}: {intro}")
    show_stance(session)
    print(render_graph(session))
    print("\n  (/why <id>, /but <id>, /set <id> <v>, /drop <id>, /leverage, /lens, /summary, /quit)")

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        if raw in ("/quit", "/q"):
            break

        try:
            if raw.startswith("/why ") or raw.startswith("/but "):
                node_id = raw.split(maxsplit=1)[1].strip()
                relation = "support" if raw.startswith("/why") else "attack"
                print("  thinking...")
                turn = await session.expand(node_id, relation)
                if not turn.added_nodes:
                    print("  (nothing new to offer in that direction)")
                    continue
                print(render_delta(turn.delta))

            elif raw.startswith("/set "):
                _, node_id, value = raw.split(maxsplit=2)
                delta = session.set_base_score(node_id, float(value))
                print(render_delta(delta))

            elif raw.startswith("/drop "):
                delta = session.remove(raw.split(maxsplit=1)[1].strip())
                print(render_delta(delta))

            elif raw == "/lens":
                print()
                for e in session.lens():
                    print(f"  [{e['id']}] {e['verdict']:10s} σ={e['sigma']:.2f}  {e['label']}")
                    for c in e["contributions"]:
                        mark = "+" if c["direction"] == "positive" else "−"
                        print(f"      {mark} [{c['id']}] σ={c['sigma']:.2f} ({c['source']})")
                continue

            elif raw == "/summary":
                print("  thinking...")
                notes = (await session.summarise())["notes"]
                print(f"\n  {notes['summary']}\n")
                for title, key in (("Key points", "key_points"), ("Agreed", "agreements"),
                                   ("Still diverging", "disagreements"), ("Next", "next_steps")):
                    if notes[key]:
                        print(f"  {title}:")
                        for line in notes[key]:
                            print(f"    - {line}")
                continue

            elif raw == "/leverage":
                print()
                for arg, score in session.leverage():
                    kind = session.disagreement_kind(arg.arg_id)
                    print(f"  [{arg.arg_id}] {kind:10s} d={score:+.2f}  {arg.text}")
                continue

            else:
                print("  thinking...")
                turn = await session.contribute(raw)
                if turn.pending_intent is not None:
                    print(f'\n  I read this as a request to me: "{turn.pending_intent}"')
                    choice = input("  fulfil it? [Y/n=it's my argument/s=skip] ").strip().lower()
                    if choice in ("", "y"):
                        print("  thinking...")
                        turn = await session.contribute(raw, force="request")
                    elif choice == "n":
                        print("  thinking...")
                        turn = await session.contribute(raw, force="argument")
                    else:
                        print("  skipped")
                        continue
                followup = turn.followup_request
                if turn.pending is not None:
                    p = turn.pending
                    if p.provenance == "generated":
                        print(f'\n  At your request I drafted: "{p.restated}"')
                        print(
                            f"  I would attach it {p.relation_type}ing [{p.target_id}] "
                            f"({p.confidence}% sure of the placement) — {p.rationale}"
                        )
                    else:
                        print(f'\n  Your point, as I read it: "{p.restated}"')
                        print(
                            f"  It {p.relation_type}s [{p.target_id}] "
                            f"({p.confidence}% sure of the placement) — {p.rationale}"
                        )
                    if input("  apply? [Y/n] ").strip().lower() in ("", "y"):
                        confidence = ask_confidence() if p.provenance == "user" else None
                        print("  thinking...")
                        turn = await session.contribute(
                            raw, attachment=p, confidence=confidence
                        )
                    else:
                        print("  skipped")
                        continue
                if followup:
                    print(f'\n  (also fulfilling your request: "{followup}")')
                    print("  thinking...")
                    turn = await session.contribute(followup, force="request")
                if turn.reply:
                    print(f"\n  {session.personas.get(turn.speaker, persona).name}: {turn.reply}")
                if turn.question:
                    print(f"  {persona.name} asks: {turn.question}")
                for label, step in turn.steps:
                    print(f"  {label}:{render_delta(step)}")
                print(render_delta(turn.delta))

        except (KeyError, ValueError, IndexError) as exc:
            print(f"  ! {exc}")
            continue

        show_stance(session)
        print(render_graph(session))

    print(f"\n  usage: {client.usage.summary()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="barcelona_homelessness",
                        choices=sorted(DOMAINS))
    parser.add_argument("--persona", default=None,
                        help="persona id(s) within the domain, comma-separated (default: first)")
    parser.add_argument("--topic", default=None,
                        help="claim to deliberate (default: domain's first seed claim)")
    parser.add_argument("--model", default=None, help="override OPENAI_MODEL")
    args = parser.parse_args()

    domain = get_domain(args.domain)
    persona_id = args.persona or domain.partners[0].id
    topic = args.topic or domain.seed_claims[0]

    try:
        asyncio.run(run(args.domain, persona_id, topic, args.model))
    except MissingAPIKey as exc:
        sys.exit(f"\n{exc}\n")
    except KeyError as exc:
        sys.exit(f"\n{exc}\n")


if __name__ == "__main__":
    main()
