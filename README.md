# OmouAI

**Argumentative human-AI policy deliberation with simulated personas.**

OmouAI lets a person deliberate a contested policy claim with LLM-simulated
personas: stakeholders with stated value commitments, a domain expert who
argues from the evidence, and a Devil's Advocate who only argues against.
Every argument, the person's and the personas', enters one shared
quantitative bipolar argumentation framework (QBAF). A deterministic gradual
semantics turns that framework into a position on the claim and into an
evaluation of external goals such as the UN Sustainable Development Goals,
so every change in the outcome traces back to a specific argument.

*Omou* means *deliberating* in Japanese and *together* in Greek: OmouAI is
deliberating together with AI.

This repository accompanies the AAAI 2027 demonstration paper
*OmouAI: Argumentative Human-AI Policy Deliberation with Simulated Personas*
([citation](#citation)).

## How it works

- **The LLM does three jobs.** It writes a persona's arguments, scores them
  (a base score τ) from that persona's commitments, and reads where your own
  free text attaches to the map. It never decides the stance.
- **The semantics decides the rest.** A gradual semantics computes every
  argument's strength σ from its base score and the strengths of its
  attackers and supporters. The claim's strength is the position. No model
  call is involved.
- **Each party owns its base scores.** A persona's arguments carry that
  persona's score; your arguments carry the confidence you state.
- **Goals are evaluation lenses.** Each selected goal is evaluated by the same
  semantics over the arguments that bear on it, without entering the
  argument map.

Two case studies are included: **climate policy** (six stakeholder positions
that accept the science and disagree about what to do) and **homelessness
policy in Barcelona** (stakeholders, a homelessness researcher, and indicators
from the Catalan homelessness law's monitoring framework). The Devil's
Advocate joins both.

## Requirements

- Python 3.11 or later
- Node.js 20 or later
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Installation

```bash
git clone https://github.com/vstylianos/OmouAI.git
cd OmouAI

# Python environment (venv shown; a conda environment works the same way)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Front end
cd frontend && npm install && cd ..

# API key
cp .env.example .env    # then put your key in .env
```

## Running

Start the API and the interface in two terminals:

```bash
# Terminal 1: API on port 8000
cd backend
../.venv/bin/python -m uvicorn api.main:app --port 8000

# Terminal 2: interface on port 5173
cd frontend
npm run dev
```

Open <http://localhost:5173>.

**Single process.** To serve everything from the API instead, build the
interface once with `cd frontend && npm run build`, then start only the API
and open <http://localhost:8000>. Sessions are held in memory, so restarting
the API ends them.

## Using the interface

### 1. Set up the deliberation

The start screen takes three steps.

1. **Case & personas.** Choose a case study and one or more personas.
2. **The claim.** Pick a suggested claim or write your own, phrased so that
   one can be for or against it.
3. **Goals.** Choose what the deliberation is measured against: the UN
   Sustainable Development Goals, the case's indicators (Barcelona), or goals
   you write yourself.

Click **Start the deliberation**. The claim starts neutral (0.5) and the map
is empty. Every argument that appears is one you made or asked for.

### 2. Build the argument map

Right-click any argument in the map, starting with the claim itself:

- **Make my own point.** Write why the argument holds or doesn't. OmouAI shows
  how it read your point (supports or attacks); correct it if needed, then say
  how sure you are: *Unsure, Somewhat, Moderately, Very* or *Certain*. Your
  confidence becomes the base score of your argument.
- **Ask a persona for a reason.** Choose a persona and **+ a reason for** or
  **− a reason against**. The persona writes and scores the argument from its
  own commitments, or declines when it has nothing new to add. The Devil's
  Advocate gives only reasons against.
- **Remove this argument.** Removes it together with everything that hangs
  on it.

Personas never respond on their own. They argue only when you ask.

### 3. Talk to the personas

The chat panel is for questions and requests, addressed to one persona at a
time (choose who you are talking to above the input box). Ask a persona where
it stands, or for an argument, e.g. *"Give me an argument that advances
SDG 1."* If you type an argument into the chat instead, OmouAI proposes where
it attaches and asks you to confirm before it enters the map.

### 4. Read the outcome

- **Position on the claim.** The meter at the top shows the claim's computed
  strength: for, neutral or against.
- **Arguments.** The map itself. Click an argument to inspect it: who made
  it, its base score and strength, what supports and objects to it, and the
  goals it bears on (you can add or remove goal tags).
- **Goal evaluation.** Each selected goal as *supported*, *contested* or
  *undermined* by the arguments that bear on it. *By goal* shows one tree per
  goal; *Shared* shows each argument once.
- **Where we stand.** How the position moved, where you agree with the
  personas, where you still diverge, and where further argument matters most.
- **Summary** (top bar). Meeting notes written from the computed record: key
  points, agreements, disagreements and next steps. They can be printed or
  saved as PDF.

### 5. Settings

The gear icon in the top bar opens the settings:

- **How the position is computed.** Switch the gradual semantics: Modular
  (linear), DF-QuAD or Quadratic energy. The whole trajectory recomputes.
- **How readily the persona updates** (Modular only). From *holds firm* to
  *moves easily*.
- **Expert mode.** Shows τ, σ, argument types and polarity on every card.
- **New deliberation.** Returns to the start screen.

## Command line

The same pipeline runs in a terminal, without the browser:

```bash
cd backend
../.venv/bin/python cli.py                                   # Barcelona case, first persona
../.venv/bin/python cli.py --domain climate --persona degrowth_advocate,devils_advocate
../.venv/bin/python cli.py --topic "Your own claim"
```

## Configuration

Set these in `.env` at the project root.

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required. |
| `OPENAI_MODEL` | Model for the personas' calls. Defaults to `gpt-5.6-terra`. |
| `CLASSIFIER_MODEL` | Optional separate model for the neutral routing and placement calls. Unset means the same model. |
| `DEMO_PASSPHRASE` | Optional. If set, the interface asks for this passphrase, for hosted demos. |

Asking a persona for an argument costs two model calls (writing and scoring);
your own points cost one or two. The counter in the top bar shows the calls
and tokens used in the session.

## Tests

```bash
cd backend
../.venv/bin/python -m pytest tests -q
```

The tests run offline and need no API key.

## Project layout

```
backend/
  engine/      QBAF, gradual semantics, attribution, goal lenses (no LLM)
  llm/         async OpenAI client and structured-output schemas
  personas/    persona specifications (climate stakeholders, Devil's Advocate)
  domains/     case studies: claims, personas, indicators
  pipeline/    prompts and the deliberation session
  api/         FastAPI service
  cli.py       terminal client
  tests/
frontend/      React + React Flow interface
```

## Responsible use

OmouAI is an argumentation-support tool, not a prescriptive decision-making
system. Generated arguments reflect the biases of the underlying model and the
value systems of the represented stakeholders. Domain experts should evaluate
the outputs before they inform any real-world decision.

## Citation

```bibtex
@inproceedings{vasileiou2027omouai,
  title     = {{OmouAI}: Argumentative Human-AI Policy Deliberation with Simulated Personas},
  author    = {Vasileiou, Stylianos Loukas and Rago, Antonio and Yeoh, William and Curto, Georgina},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence, Demonstration Track},
  year      = {2027}
}
```

## License

Released under the [Apache License 2.0](LICENSE).

## Acknowledgments

This research is partially supported by the National Science Foundation
(award #2232055).
