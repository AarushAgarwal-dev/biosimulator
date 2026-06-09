# 🧬 BioSimulateAI

**AI-Powered Biological Model Compilation, Simulation & Calibration Copilot**

BioSimulateAI is a web-based platform that translates natural-language descriptions of biological systems into executable mathematical models. It integrates multi-scale simulation (ODE, PDE, ABM), database-backed Retrieval-Augmented Generation (RAG), closed-loop AI-driven model refinement, and literature-grounded parameter calibration (MAPLE).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![License](https://img.shields.io/badge/License-Research-yellow)

---

## Features

### 🔬 Natural Language → Mathematical Model
- Type biological descriptions like *"EGF binds to EGFR and activates it. EGFR activates RAS."*
- **Rule-based parser** (regex) or **Gemini LLM** compiles text into structured JSON blueprints
- Automatic generation of Hill-function ODE systems with symbolic math (SymPy)
- LaTeX equation rendering via KaTeX

### 📊 Multi-Scale Simulation Engine
| Scale | Method | Implementation |
|-------|--------|----------------|
| **Intracellular** | ODE (Ordinary Differential Equations) | `simulation_engine.py` — SymPy symbolic compilation → SciPy `solve_ivp` |
| **Tissue-level** | PDE (Reaction-Diffusion) | `simulation_engine.py` — Finite difference Euler scheme with Neumann BCs |
| **Cell-level** | ABM (Cellular Potts Model) | `abm_engine.py` — Full GGH/CPM with Hamiltonian energy minimization |
| **Coupled** | ODE ↔ ABM ↔ PDE | `multiscale.py` — Time-scale separated coupling coordinator |

### 🗄️ Database RAG (Retrieval-Augmented Generation)
Real-time retrieval from 5 biological knowledge databases to ground model generation:

| Database | What it provides | API |
|----------|------------------|-----|
| **Reactome** | Curated pathway structures & reactions | REST Content Service |
| **STRING DB** | Protein-protein interaction networks | JSON API |
| **OmniPath** | Directed signaling with literature refs | REST API |
| **SIGNOR** | Curated signaling relationships | REST API |
| **BioModels** | Published SBML mathematical models | EBI REST API |

Retrieved interactions are transformed into natural-language descriptions and fed to the parser/LLM for blueprint generation.

### 🔄 Closed-Loop AI Feedback
- Define **target behaviors** (peak time, peak value, decay ratio, steady state)
- Automated **target evaluation** against simulation results
- **AI-driven refinement**: Gemini adjusts parameters/topology to meet targets
- **Rule-based fallback**: heuristic parameter tuning when LLM unavailable
- **Sensitivity-guided refinement**: local sensitivity analysis ranks parameter importance

### 📐 MAPLE Calibration Pipeline
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
│  agent.py          - NLP parser + Gemini LLM agent    │
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
pip install fastapi uvicorn numpy scipy sympy pydantic requests

# (Optional) For Gemini LLM features
pip install google-genai

# Run the server
python main.py
```

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
├── agent.py                # NLP text parser & Gemini LLM agent
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
| `/api/blueprint` | POST | Parse natural language → blueprint JSON |
| `/api/compile` | POST | Compile blueprint → LaTeX equations + parameters |
| `/api/simulate` | POST | Run ODE or PDE simulation |
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
| `/api/signor/search` | GET | Search SIGNOR pathway data |
| `/api/biomodels/search` | GET | Search BioModels repository |
| `/api/biomodels/import` | POST | Import SBML model as blueprint |

---

## Technologies

- **Backend**: Python, FastAPI, Uvicorn
- **Symbolic Math**: SymPy (ODE compilation, LaTeX generation)
- **Numerical Solvers**: SciPy (solve_ivp, least_squares), NumPy
- **Validation**: Pydantic v2 (schema validation, hallucination detection)
- **LLM**: Google Gemini API (text parsing, parameter extraction, model refinement)
- **Frontend**: Vanilla HTML/CSS/JS
- **Visualization**: Cytoscape.js (network graphs), Chart.js (time series), KaTeX (LaTeX), Canvas API (heatmaps, CPM lattices)

---

## License

Research use. See project documentation for details.
