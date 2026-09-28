# 🧬 BioSimulateAI

**AI-Powered Biological Model Compilation, Simulation & Calibration Copilot**

BioSimulateAI is a web-based platform that translates natural-language descriptions of biological systems into executable mathematical models. It integrates multi-scale simulation (ODE, PDE, ABM), database-backed Retrieval-Augmented Generation (RAG), closed-loop AI-driven model refinement, and literature-grounded parameter calibration (MAPLE).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![License](https://img.shields.io/badge/License-Research-yellow)

---

## Features

### Natural Language → Mathematical Model
- Type biological descriptions like *"p53 activates Mdm2 transcription. Mdm2 promotes p53 degradation. DNA damage stabilizes p53."*
- **Verified IR compiler** (`nl_compiler.py`): the AI engine (Purdue GenAI Studio by default) only
  fills a strict intermediate representation - species, processes (production, degradation,
  conversion, binding, custom rate law) and parameters, each process with a verbatim quote from
  your text. The equations are then assembled **deterministically**, so conversions and binding
  conserve mass by construction.
- **Checked before it is shown**: every stated interaction's sign is verified on the compiled
  equations (∂(dX/dt)/∂A), the model must simulate finite, non-negative and bounded, and every
  entity you named must be represented. Failures are repaired (up to two AI rounds) or the
  deterministic compiler's verified model is used, and the UI says which happened.
- **Provenance panel**: which engine built the model, the sentence behind each process, which
  numbers came from your text and which are defaults, and every assumption the compiler made.
- **Offline rule-based compiler** for common phrasings, and **exact equations** (`dX/dt = …`,
  typed or read from a photo by a vision model) compiled with no reinterpretation.
- Benchmark: `python verify_nl_compiler.py [--engine purdue|bedrock] [--rules-only]` (21 descriptions
  with numeric behavioural checks, run on the rules path and on the live AI engine).

### Organism-scale BMP patterning (Umulis et al. 2010)
The **Embryo BMP** tab implements Umulis, Shimmi, O'Connor & Othmer (2010), *Developmental Cell*
18:260-274 (`bmp_embryo.py`): the reaction-transport equations and every fitted parameter of
Supplemental Tables S1/S2/S3/S8, solved on the surface of a 400 × 180 µm prolate-spheroid embryo
and on the AP-midline cross-section (BDF with an analytic sparse Jacobian; ligand mass balance
checked to 10⁻³). The paper's separate refit for the ellipsoid (Table S11, Case 1) is available
as a second parameter set.
- **Solver verification**: the same code reproduces the fully specified predecessor model
  (Umulis et al. 2006 PNAS, Supp. Fig. 13) within 10 % from 30 min to steady state.
- **Paper claims** (`python verify_bmp_umulis2010.py`): the contracting dorsal stripe, sog+/-
  widening, tld-/- loss of signal, loss of localisation with the in-vitro Sog/BMP on-rate, the
  role of feedback, the no-feedback set also forming a stripe, and scaling behaviour (a 750 µm
  embryo splits into two stripes from ~21 % EL; paper: ~25 %) are each measured and reported
  pass/fail - 15 of 17 pass. Claims about the paper's 3D model are checked on the embryo surface
  at x/L = 0.5. The ellipsoid refit reproduces the 60-min dorsal-midline level of Fig. 4F
  (37.8 vs 37.9 nM).
  The two that do not pass - the Fig. 4F time course before 60 min, and the size of the sog+/-
  widening - depend on inputs the paper did not publish (the FISH-derived Sog field and the
  image-derived initial state). In the model, laterally secreted Sog/Tsg floods the dorsal side for
  the first ~20 min and holds the signal near zero; the paper's curve already reads 16.8 nM at
  15 min. Both failures are shown in the app with that explanation.
- **What had to be interpreted** is listed in the app: Table S1's printed Sog secretion
  (1.36 µM/min) abolishes the stripe when used literally, so the 2006 value (400 nM/min) is the
  default and the printed value is selectable.

### Interaction graph
Arrows are derived from the equations (sign of the partial derivative of each species' production
terms). Weak influences (< 5 % of the strongest on a target) are folded away for readability;
**Show every connection** draws all of them, and no species is ever left unconnected.

### Guides
Three short motion-graphic explainers (Guides tab) rendered from the app's own outputs by
`python tools/make_videos.py` (needs matplotlib and ffmpeg, which the web service itself does not):
the language-to-model pipeline, the embryo model, and reading the graph.

### Exact SBML import
BioModels / SBML models are translated exactly (`sbml_import.py`: MathML kinetic laws,
stoichiometry, compartments, local parameters, function definitions, assignment and rate rules).
`python tools/verify_sbml_import.py` compares 8 curated BioModels entries against libroadrunner;
all agree to better than 10⁻⁴. Events and delays are reported as not imported.

### Published-model presets
Two presets are the papers' own equations and parameter sets, checked against the numbers in
those papers (`test_paper_models.py`):
- **Goldbeter, Dupont & Berridge (1990)** PNAS 87:1461 - Ca²⁺ oscillations. Oscillations occur
  for β = 28.9-77.4 % (paper: 29.1-77.5 %), the steady state above that range is 0.67 µM
  (paper: "close to 0.7 µM"), and the period and amplitude are within 3 % of the digitised Fig. 3.
- **Zhabotinsky (2000)** Biophys J 79:2211 - CaMKII bistability, with the Fig. 9 parameters
  (Ca²⁺-independent phosphatase). The four step experiments of Fig. 9B come out as published,
  and the cytosolic set of Fig. 10A is single-valued.

The fold-change preset (after Lyashenko et al. 2020) is a reduced model and says so.

### Multi-Scale Simulation Engine
| Scale | Method | Implementation |
|-------|--------|----------------|
| **Intracellular** | ODE (Ordinary Differential Equations) | `simulation_engine.py`: SymPy symbolic compilation → SciPy `solve_ivp` |
| **Tissue-level** | PDE (Reaction-Diffusion) | `simulation_engine.py`: Finite difference Euler scheme with Neumann BCs |
| **Cell-level** | ABM (Cellular Potts Model) | `abm_engine.py`: Full GGH/CPM with Hamiltonian energy minimization |
| **Coupled** | ODE ↔ ABM ↔ PDE | `multiscale.py`: Time-scale separated coupling coordinator |

### Database RAG (Retrieval-Augmented Generation)
Real-time retrieval from 5 biological knowledge databases to ground model generation:

| Database | What it provides | API |
|----------|------------------|-----|
| **Reactome** | Curated pathway structures & reactions | REST Content Service |
| **STRING DB** | Protein-protein interaction networks | JSON API |
| **OmniPath** | Directed signaling with literature refs | REST API |
| **SIGNOR** | Curated causal relationships (mechanism, effect, PMID, score) | `getData.php` download service, by UniProt accession (gene symbols mapped through UniProt) |
| **BioModels** | Published SBML mathematical models | REST API; SBML from the CC0 GitHub mirror of the curated collection (`sys-bio/temp-biomodels`) when BioModels does not respond |

When a database does not answer, a small set of offline examples keeps the interface usable. Every
offline record is marked `"offline": true`, the interface says so, and none cites a paper.

Retrieved interactions are transformed into natural-language descriptions and fed to the parser/LLM for blueprint generation.

### Closed-Loop AI Feedback
- Define **target behaviors** (peak time, peak value, decay ratio, steady state)
- Automated **target evaluation** against simulation results
- **AI-driven refinement**: the LLM adjusts parameters/topology to meet targets
- **Rule-based fallback**: heuristic parameter tuning when LLM unavailable
- **Sensitivity-guided refinement**: local sensitivity analysis ranks parameter importance

###  MAPLE Calibration Pipeline
Inspired by Eliason & Popel (2026), implements structured parameter extraction from scientific literature:
- LLM-powered extraction with **Pydantic schema validation**
- **Hallucination detection**: values cross-checked against verbatim source snippets
- **DOI resolution**: citation verification via DOI.org API
- **Forward model templates**: 15 built-in model types (Hill, Michaelis-Menten, etc.)
- **SBML import**: BioModels XML → internal blueprint conversion

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Frontend (HTML/JS)                 │
│  Cytoscape.js │ Chart.js │ KaTeX │ Canvas Renderer   │
└──────────────────────┬───────────────────────────────┘
                       │ REST API (FastAPI)
┌──────────────────────┴───────────────────────────────┐
│                    Backend (Python)                    │
│                                                       │
│  agent.py          - NLP parser + open-source LLM agent│
│  simulation_engine - ODE compiler + PDE solver        │
│  abm_engine        - Cellular Potts Model (CPM/GGH)   │
│  multiscale        - ODE↔ABM↔PDE coupling coordinator │
│  maple_extractor   - LLM parameter extraction         │
│  maple_schemas     - Pydantic validation schemas       │
│  db_interface      - External database connectors      │
│  abm_blueprints    - Preset ABM configurations         │
│  main.py           - FastAPI routes & server            │
└──────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd biosimulator

# Install dependencies
pip install -r requirements.txt

# (Optional) For local open-source LLM inference (CPU), a prebuilt wheel:
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install huggingface_hub

# Run the server
python main.py
```

### AI engine

Open the **🧠** control in the header and choose:

- **Purdue GenAI Studio** (recommended): put `PURDUE_GENAI_API_KEY` in `.env` (see
  `.env.example`). The key stays on the server. `gpt-oss:120b` compiles a model in ~10-30 s;
  photos of equations are read by a Purdue vision model (gemma4 / qwen3-vl).
- **Rule-based (offline)**: deterministic compiler, instant, no model.
- **AWS Bedrock**, **Remote endpoint** (any OpenAI-compatible server) or **Local model**
  (GGUF via `llama-cpp-python`) remain available. Every engine goes through the same verified
  IR compiler (the older free-form path is kept only as `"compiler": "legacy"` on the API), and
  Bedrock's vision models can read photos of equations too.

On a public deployment (`BIOSIM_REQUIRE_PAID_ACCESS_TOKEN=1`) AI engines need the deployment
access token unless `BIOSIM_PURDUE_PUBLIC=1`; without it the UI falls back to the rule-based
compiler and says so. AI requests are rate-limited per client and globally.

The app will be available at **http://127.0.0.1:8000**

---

## Usage

### 1. Describe a Biological System
Type natural language or load a preset:
```
EGF binds to EGFR and activates it.
EGFR activates RAS.
RAS activates RAF.
RAF activates MEK.
MEK activates ERK.
ERK inhibits EGFR.
EGF starts at 10.0.
EGFR starts at 1.0.
```

### 2. Compile Blueprint
Click **"Compile Blueprint"** to generate:
- Interactive network graph (Cytoscape.js)
- JSON blueprint structure
- Hill-function ODE equations (KaTeX)
- Parameter sliders

### 3. Run Simulation
Adjust parameters via sliders and simulate:
- **ODE**: Time-series concentration dynamics (Chart.js)
- **PDE**: Animated 2D reaction-diffusion heatmaps (Canvas)
- **ABM**: Cellular Potts Model lattice visualization (Canvas)

### 4. Database RAG
Search external databases to retrieve real biological interactions:
- Query results auto-populate the text input
- Grounds model generation in curated knowledge

### 5. Closed-Loop Optimization
Define target behaviors and run iterative refinement:
- AI evaluates targets vs. simulation
- Automatically adjusts parameters or adds feedback loops
- Sensitivity analysis identifies most impactful parameters

### 6. MAPLE Calibration
Extract quantitative parameters from literature:
- Enter parameter name, units, and context
- LLM extracts values with provenance tracking
- Validation catches hallucinated values and fake citations

---

## Project Structure

```
biosimulator/
├── main.py                 # FastAPI server & API routes
├── nl_compiler.py          # Verified plain-language -> IR -> equations compiler
├── bmp_embryo.py           # Umulis et al. 2010 Drosophila BMP embryo model + validation
├── llm_provider.py         # AI engines: Purdue GenAI Studio, Bedrock, remote, local GGUF
├── agent.py                # Legacy parser, refinement and target evaluation
├── tools/make_videos.py    # Renders the Guides videos from real outputs
├── simulation_engine.py    # ODE compiler (SymPy) & PDE solver
├── abm_engine.py           # Cellular Potts Model engine
├── abm_blueprints.py       # Preset ABM configurations
├── multiscale.py           # Multi-scale simulation coordinator
├── maple_extractor.py      # MAPLE parameter extraction pipeline
├── maple_schemas.py        # Pydantic schemas & validators
├── db_interface.py         # External database API connectors
├── test_maple.py           # MAPLE validation tests
├── test_abm.py             # ABM simulation tests
├── test_math.py            # Math/ODE compilation tests
├── static/
│   ├── index.html          # Main application page
│   ├── style.css           # UI stylesheet
│   ├── app.js              # Frontend application logic
│   └── favicon.png         # Application icon
└── README.md
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/blueprint` | POST | Description → verified blueprint (`llm.engine`: `purdue` / `off` / `bedrock` …) |
| `/api/compile` | POST | Compile blueprint → LaTeX equations, parameters, derived graph (`derived_edges`, `derived_edges_all`) |
| `/api/llm/env` | GET | Configured engines (booleans and model names only) |
| `/api/llm/purdue/models` | GET | Models available to the server's Purdue key |
| `/api/bmp/info` | GET | Umulis 2010 equations, parameters with sources, assumptions |
| `/api/bmp/validation` | GET | Model-vs-paper checks (`?fresh=true` recomputes the 1D checks) |
| `/api/bmp/cross-section` | POST | Solve the DV cross-section for a mechanism / genotype / parameter change |
| `/api/bmp/surface` | POST | Solve on the embryo surface (size-capped, one at a time) |
| `/api/simulate` | POST | Run ODE or PDE simulation |
| `/api/sample` | POST | Latin-Hypercube parameter-space exploration |
| `/api/llm/models` | GET | List open-source LLMs + download status |
| `/api/llm/download` | POST | Download a model's weights from HuggingFace |
| `/api/llm/status` | GET | Poll a model's download status |
| `/api/optimize` | POST | Fit parameters to target data |
| `/api/refine` | POST | Closed-loop AI model refinement |
| `/api/sensitivity` | POST | Local sensitivity analysis |
| `/api/abm/simulate` | POST | Run ABM Cellular Potts simulation |
| `/api/abm/presets` | GET | List available ABM presets |
| `/api/multiscale/simulate` | POST | Run coupled ODE-ABM-PDE simulation |
| `/api/maple/extract` | POST | LLM parameter extraction with validation |
| `/api/maple/validate` | POST | Schema validation for calibration targets |
| `/api/reactome/search` | GET | Search Reactome pathways |
| `/api/string/network` | POST | Fetch STRING protein interactions |
| `/api/omnipath/interactions` | POST | Query OmniPath signaling data |
| `/api/signor/search` | GET | Curated SIGNOR relations for one human protein |
| `/api/biomodels/search` | GET | Search BioModels repository |
| `/api/biomodels/import` | POST | Import SBML model as blueprint |

---

## Technologies

- **Backend**: Python, FastAPI, Uvicorn
- **Symbolic Math**: SymPy (ODE compilation, LaTeX generation)
- **Numerical Solvers**: SciPy (solve_ivp, least_squares), NumPy
- **Validation**: Pydantic v2 (schema validation, hallucination detection)
- **LLM**: Open-source models via llama-cpp-python (local GGUF: Llama / Gemma / Qwen / gpt-oss) or any OpenAI-compatible remote endpoint (text parsing, parameter extraction, model refinement)
- **Frontend**: Vanilla HTML/CSS/JS
- **Visualization**: Cytoscape.js (network graphs), Chart.js (time series), KaTeX (LaTeX), Canvas API (heatmaps, CPM lattices)

---

## License

Research use. See project documentation for details.
