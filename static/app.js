// ==========================================================================
// STATE MANAGEMENT
// ==========================================================================
const state = {
    blueprint: null,
    simulationResults: null,
    equations: null,
    equationsVerbose: null,
    eqMode: 'symbols',      // 'symbols' | 'full'
    customParams: {},
    paramBounds: {},        // pname -> {min, max, step}  (user-editable ranges)
    showBounds: false,      // whether the min/max editors are visible
    sampleResult: null,     // last parameter-space exploration result
    targets: [],
    currentDb: 'reactome',
    llm: { engine: 'off', model: 'llama-3.2-3b', remote_url: '', remote_model: '', remote_key: '',
           bedrock_model: 'mistral.mistral-large-3-675b-instruct', bedrock_region: 'us-east-2',
           bedrock_bearer_token: '', bedrock_access_key: '', bedrock_secret_key: '', bedrock_session_token: '' },

    // PDE Animation Player
    pdePlaying: false,
    pdePlayInterval: null,
    pdeCurrentFrame: 0,
    pdeSelectedSpecies: ''
};

// Default Showcases
const Presets = {
    egfr: {
        text: `EGF binds to EGFR and activates it.
EGFR activates RAS.
RAS activates RAF.
RAF activates MEK.
MEK activates ERK.
ERK inhibits EGFR.
EGF starts at 10.0.
EGFR starts at 1.0.
RAS starts at 1.0.
RAF starts at 1.0.`,
        targets: [
            { species: "ERK", type: "peak_time", min: 5.0, max: 15.0 },
            { species: "ERK", type: "peak_value", min: 0.6, max: 1.0 },
            { species: "EGFR", type: "decay_ratio", max: 0.2 } // EGFR should decay to less than 20% of its peak
        ]
    },
    turing: {
        text: `A reaction-diffusion system containing Activator U and Inhibitor V.
U activates itself and activates V.
V inhibits U.
U starts at 1.0.
V starts at 1.0.
U diffuses slowly, V diffuses quickly.`,
        targets: [
            // Target checks for spatial heterogeneity (variance of pattern)
            { species: "U", type: "steady_state", value: 1.0, tolerance: 10.0 }
        ]
    },
    oscillator: {
        // Three-gene negative-feedback loop (Goodwin/repressilator-style).
        // Compiles to a delayed negative-feedback loop that the closed-loop
        // optimizer can drive into a sustained limit cycle via an oscillation target.
        text: `GENA activates GENB.
GENB activates GENC.
GENC inhibits GENA.
GENA starts at 1.0.
GENB starts at 0.5.
GENC starts at 0.2.`,
        targets: [
            { species: "GENA", type: "oscillation", min: 0.3 }
        ],
        t_max: 200.0
    },
    bistable: {
        // A stimulus drives a switch species; the optimizer adds self-activation
        // (positive feedback) and tunes it into a two-state bistable switch.
        text: `STIM activates CAMKII.
STIM starts at 1.0.
CAMKII starts at 0.1.`,
        targets: [
            { species: "CAMKII", type: "bistability", min: 0.5 }
        ],
        t_max: 100.0
    },
    foldchange: {
        // EGF -> Akt fold-change detection. The optimizer wires an incoherent
        // feedforward loop so the response tracks the input's fold-change, not its level.
        text: `EGF activates AKT.
EGF starts at 1.0.
AKT starts at 0.1.`,
        targets: [
            { species: "AKT", type: "fold_change", input: "EGF", output: "AKT", max: 0.15 }
        ],
        t_max: 60.0
    }
};

// ==========================================================================
// PUBLISHED-PAPER MODELS (exact custom kinetics)
// These load the real ODE systems from the source papers directly (rate laws +
// mass-conserving shared fluxes), bypassing the generic Hill compiler. Each is
// tuned so a single "Run Simulation" reproduces the paper's hallmark behaviour.
// ==========================================================================
const PaperModels = {
    berridge: {
        title: "Ca²⁺ oscillations (Berridge & Goldbeter, 1990)",
        hallmark: "sustained cytosolic Ca²⁺ oscillations",
        description:
`Published model: Berridge–Goldbeter (1990) two-pool Ca²⁺ oscillator.
Cytosolic Ca²⁺ (Z) and internal-store Ca²⁺ (Y) exchange via CICR.
This loads the exact rate laws (v2 pump, v3 store release) as a
mass-conserving ODE system. Click "Run Simulation" to see the
self-sustained calcium spikes.`,
        blueprint: {
            type: "ODE",
            name: "Berridge–Goldbeter Ca2+ oscillator",
            nodes: [
                { id: "Z", name: "Cytosolic Ca2+", initial_value: 0.1 },
                { id: "Y", name: "Internal-store Ca2+", initial_value: 0.1 }
            ],
            edges: [
                { id: "e1", source: "Z", target: "Y", type: "activation" },
                { id: "e2", source: "Y", target: "Z", type: "activation" }
            ],
            parameters: {
                v0: 1.0, v1: 7.3, beta: 0.5, VM2: 65.0, VM3: 500.0,
                K2: 1.0, KR: 2.0, KA: 0.9, kf: 1.0, k: 10.0
            },
            fluxes: {
                v2: "VM2*Z**2/(K2**2+Z**2)",
                v3: "VM3*(Y**2/(KR**2+Y**2))*(Z**4/(KA**4+Z**4))"
            },
            odes: {
                Z: "v0 + v1*beta - v2 + v3 + kf*Y - k*Z",
                Y: "v2 - v3 - kf*Y"
            },
            plot_species: ["Z", "Y"],
            simulation_config: { t_max: 10.0 }
        },
        targets: [
            { species: "Z", type: "oscillation", min: 0.2 }
        ]
    },

    zhabotinsky: {
        title: "CaMKII bistable memory (Zhabotinsky, 2000)",
        hallmark: "a transient Ca²⁺ pulse that latches CaMKII permanently ON",
        description:
`Published model: Zhabotinsky (2000) CaMKII autophosphorylation switch.
An 11-state holoenzyme (P0..P10) with Ca²⁺/CaM-driven phosphorylation and
saturable PP1 dephosphorylation — the exact mechanism that makes the kinase
bistable. Baseline Ca²⁺ is held at 2.0, inside the bistable window (~1.8-2.15),
so both OFF and ON states are stable. A brief Ca²⁺ pulse (t=20-60) to 3.0 flips
the switch; active CaMKII (A) then stays ON permanently = molecular memory.`,
        blueprint: {
            type: "ODE",
            name: "Zhabotinsky CaMKII bistable switch",
            nodes: [
                { id: "Ca", name: "Calcium stimulus", initial_value: 2.0 },
                { id: "P0", name: "Unphosphorylated CaMKII", initial_value: 2.0 },
                { id: "P1", initial_value: 0.0 }, { id: "P2", initial_value: 0.0 },
                { id: "P3", initial_value: 0.0 }, { id: "P4", initial_value: 0.0 },
                { id: "P5", initial_value: 0.0 }, { id: "P6", initial_value: 0.0 },
                { id: "P7", initial_value: 0.0 }, { id: "P8", initial_value: 0.0 },
                { id: "P9", initial_value: 0.0 }, { id: "P10", initial_value: 0.0 },
                { id: "A", name: "Active CaMKII", initial_value: 0.0 }
            ],
            edges: [
                { id: "e1", source: "Ca", target: "A", type: "activation" },
                { id: "e2", source: "A", target: "A", type: "activation" }
            ],
            parameters: {
                k1: 0.5, k2: 2.0, KH1: 4.0, KM: 0.4, ep: 0.05,
                kca: 5.0, Cabase: 2.0, amp: 1.0, sr: 2.0, t_on: 20.0, t_off: 60.0, kobs: 50.0
            },
            fluxes: {
                v1: "10*k1*(Ca/KH1)**8*P0/(1 + (Ca/KH1)**4)**2",
                v2: "k1*(Ca/KH1)**4/(1 + (Ca/KH1)**4)",
                Ssum: "1*P1+2*P2+3*P3+4*P4+5*P5+6*P6+7*P7+8*P8+9*P9+10*P10",
                v3: "k2*ep/(KM + Ssum)"
            },
            odes: {
                P0: "-v1 + v3*1*P1",
                P1: "v1 - v2*1.0*P1 - v3*1*P1 + v3*2*P2",
                P2: "v2*1.0*P1 - v2*1.8*P2 - v3*2*P2 + v3*3*P3",
                P3: "v2*1.8*P2 - v2*2.3*P3 - v3*3*P3 + v3*4*P4",
                P4: "v2*2.3*P3 - v2*2.7*P4 - v3*4*P4 + v3*5*P5",
                P5: "v2*2.7*P4 - v2*2.8*P5 - v3*5*P5 + v3*6*P6",
                P6: "v2*2.8*P5 - v2*2.7*P6 - v3*6*P6 + v3*7*P7",
                P7: "v2*2.7*P6 - v2*2.3*P7 - v3*7*P7 + v3*8*P8",
                P8: "v2*2.3*P7 - v2*1.8*P8 - v3*8*P8 + v3*9*P9",
                P9: "v2*1.8*P8 - v2*1.0*P9 - v3*9*P9 + v3*10*P10",
                P10: "v2*1.0*P9 - v3*10*P10",
                Ca: "kca*(Cabase + amp/(1+exp(-sr*(t-t_on))) - amp/(1+exp(-sr*(t-t_off))) - Ca)",
                A: "kobs*(Ssum - A)"
            },
            plot_species: ["Ca", "A"],
            simulation_config: { t_max: 350.0 }
        },
        targets: [
            { species: "A", type: "steady_state", value: 15.0, tolerance: 4.0 }
        ]
    },

    lyashenko: {
        title: "Fold-change detection (Lyashenko et al., 2020)",
        hallmark: "equal responses to equal fold-changes, at ANY absolute ligand level",
        description:
`Published model: Lyashenko et al. (2020) receptor-based relative sensing / cell memory.
The receptor pool R adapts to (remembers) the ambient ligand background, and the
downstream response S is driven by the RATIO ligand/background (L/R). So a given
fold-change gives the SAME response regardless of absolute level = Weber's law / FCD.
Here ligand climbs an exact 2x staircase (1->2->4->8->16, a full decade); every
response pulse has identical height, and S re-adapts to baseline after each step.`,
        blueprint: {
            type: "ODE",
            name: "Lyashenko fold-change detection",
            nodes: [
                { id: "L", name: "Ligand", initial_value: 1.0 },
                { id: "R", name: "Adapted background (receptor memory)", initial_value: 1.0 },
                { id: "S", name: "Relative response", initial_value: 1.0 }
            ],
            edges: [
                { id: "e1", source: "L", target: "R", type: "activation" },
                { id: "e2", source: "L", target: "S", type: "activation" },
                { id: "e3", source: "R", target: "S", type: "inhibition" }
            ],
            parameters: {
                a: 0.2, kf: 20.0, kl: 40.0, sr: 8.0,
                L1: 1.0, L2: 2.0, L3: 4.0, L4: 8.0, L5: 16.0,
                t1: 20.0, t2: 45.0, t3: 70.0, t4: 95.0
            },
            odes: {
                // R adapts to the ligand background at a constant (level-independent) rate;
                // S tracks the ligand-to-background ratio -> response depends only on fold-change.
                L: "kl*(L1 + (L2-L1)/(1+exp(-sr*(t-t1))) + (L3-L2)/(1+exp(-sr*(t-t2))) + (L4-L3)/(1+exp(-sr*(t-t3))) + (L5-L4)/(1+exp(-sr*(t-t4))) - L)",
                R: "a*(L - R)",
                S: "kf*(L/R - S)"
            },
            plot_species: ["L", "R", "S"],
            simulation_config: { t_max: 125.0 }
        },
        targets: [
            { species: "S", type: "steady_state", value: 1.0, tolerance: 0.4 }
        ]
    }
};

// Color palettes for PDE rendering
const PDE_PALETTES = {
    thermal: (v) => {
        // v in [0, 1]
        // Map from dark blue -> cyan -> green -> yellow -> red
        const r = Math.min(255, Math.max(0, Math.floor(255 * (v - 0.5) * 2)));
        const g = Math.min(255, Math.max(0, Math.floor(255 * (1 - 2 * Math.abs(v - 0.5)))));
        const b = Math.min(255, Math.max(0, Math.floor(255 * (0.5 - v) * 2)));
        return `rgb(${r},${g},${b})`;
    },
    plasma: (v) => {
        // v in [0, 1]
        // Dark purple -> pink -> orange -> yellow
        const r = Math.floor(255 * Math.pow(v, 0.8));
        const g = Math.floor(200 * Math.pow(v, 1.5));
        const b = Math.floor(255 * (1 - Math.pow(v - 0.5, 2) * 4));
        return `rgb(${r},${g},${b})`;
    }
};

let cyInstance = null;
let chartInstance = null;

// ==========================================================================
// INITIALIZATION
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initPresets();
    initEventListeners();
    
    // Load EGFR preset by default
    loadPreset('egfr');
});

// Tab Switching
function initTabs() {
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const tabId = btn.getAttribute("data-tab");
            
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            
            btn.classList.add("active");
            document.getElementById(`tab-${tabId}`).classList.add("active");
            
            // Re-layout cytoscape graph if entering blueprint tab
            if (tabId === 'blueprint' && cyInstance) {
                cyInstance.resize();
                cyInstance.layout({ name: 'cose' }).run();
            }
        });
    });
}

function initPresets() {
    document.getElementById("load-egfr-btn").addEventListener("click", () => loadPreset('egfr'));
    document.getElementById("load-turing-btn").addEventListener("click", () => loadPreset('turing'));
    const oscBtn = document.getElementById("load-oscillator-btn");
    if (oscBtn) oscBtn.addEventListener("click", () => loadPreset('oscillator'));
    const biBtn = document.getElementById("load-bistable-btn");
    if (biBtn) biBtn.addEventListener("click", () => loadPreset('bistable'));
    const fcBtn = document.getElementById("load-foldchange-btn");
    if (fcBtn) fcBtn.addEventListener("click", () => loadPreset('foldchange'));

    // Published-paper models (exact custom kinetics, loaded directly)
    [["load-berridge-btn", "berridge"], ["load-zhabotinsky-btn", "zhabotinsky"],
     ["load-lyashenko-btn", "lyashenko"]].forEach(([id, key]) => {
        const b = document.getElementById(id);
        if (b) b.addEventListener("click", () => loadPaperModel(key));
    });
}

// Load a published-paper model directly as a custom-kinetics blueprint (no text parse).
async function loadPaperModel(name) {
    const pm = PaperModels[name];
    if (!pm) return;

    updateStatus("Loading published model...", "yellow");

    // Deep-copy so slider edits don't mutate the template
    state.blueprint = JSON.parse(JSON.stringify(pm.blueprint));
    state.targets = JSON.parse(JSON.stringify(pm.targets || []));
    state.tmaxOverride = pm.blueprint.simulation_config ? pm.blueprint.simulation_config.t_max : null;

    // Show the paper description for context (this is not re-parsed)
    document.getElementById("bio-input").value = pm.description;
    if (state.tmaxOverride) document.getElementById("sim-tmax").value = state.tmaxOverride;

    // Toggle active state across every preset button
    ["load-egfr-btn", "load-turing-btn", "load-oscillator-btn", "load-bistable-btn",
     "load-foldchange-btn", "load-berridge-btn", "load-zhabotinsky-btn", "load-lyashenko-btn"]
        .forEach(id => {
            const b = document.getElementById(id);
            if (b) b.classList.toggle("active", id === `load-${name}-btn`);
        });

    // Blueprint JSON + graph + targets + compile (equations & sliders)
    document.getElementById("blueprint-json-viewer").textContent = JSON.stringify(state.blueprint, null, 2);
    renderCytoscape();
    renderTargets();
    await compileBlueprint();

    // Show the loaded model, then invite the user to run it
    document.querySelector("[data-tab='blueprint']").click();
    if (typeof showToast === "function") {
        showToast(`Loaded ${pm.title}. Click "Run Simulation" to reproduce ${pm.hallmark}.`, "info", 7000);
    }
    updateStatus("Ready", "green");
}

function loadPreset(name) {
    const preset = Presets[name];
    document.getElementById("bio-input").value = preset.text;

    // Load targets
    state.targets = JSON.parse(JSON.stringify(preset.targets));
    renderTargets();

    // Presets may set a simulation horizon (oscillations need a long run).
    state.tmaxOverride = preset.t_max || null;
    if (preset.t_max) document.getElementById("sim-tmax").value = preset.t_max;

    // Toggle active classes on buttons
    document.getElementById("load-egfr-btn").classList.toggle("active", name === 'egfr');
    document.getElementById("load-turing-btn").classList.toggle("active", name === 'turing');
    [["load-oscillator-btn", "oscillator"], ["load-bistable-btn", "bistable"],
     ["load-foldchange-btn", "foldchange"]].forEach(([id, key]) => {
        const b = document.getElementById(id);
        if (b) b.classList.toggle("active", name === key);
    });
}

// ==========================================================================
// TARGET RENDERING & MANAGEMENT
// ==========================================================================
function renderTargets() {
    const container = document.getElementById("targets-list");
    container.innerHTML = "";
    
    state.targets.forEach((tgt, idx) => {
        const row = document.createElement("div");
        row.className = "target-row";
        
        // Species select or text
        let optionsHtml = '';
        if (state.blueprint && state.blueprint.nodes) {
            optionsHtml = state.blueprint.nodes.map(n => 
                `<option value="${n.id}" ${n.id === tgt.species ? 'selected' : ''}>${n.id}</option>`
            ).join('');
        } else {
            optionsHtml = `<option value="${tgt.species}" selected>${tgt.species}</option>`;
        }
        
        row.innerHTML = `
            <select class="tgt-species" data-idx="${idx}">
                ${optionsHtml}
            </select>
            <select class="tgt-type" data-idx="${idx}">
                <option value="peak_time" ${tgt.type === 'peak_time' ? 'selected' : ''}>Peak Time</option>
                <option value="peak_value" ${tgt.type === 'peak_value' ? 'selected' : ''}>Peak Val</option>
                <option value="decay_ratio" ${tgt.type === 'decay_ratio' ? 'selected' : ''}>Decay %</option>
                <option value="steady_state" ${tgt.type === 'steady_state' ? 'selected' : ''}>Steady State</option>
                <option value="oscillation" ${tgt.type === 'oscillation' ? 'selected' : ''}>Oscillation</option>
                <option value="bistability" ${tgt.type === 'bistability' ? 'selected' : ''}>Bistability</option>
                <option value="fold_change" ${tgt.type === 'fold_change' ? 'selected' : ''}>Fold-change</option>
            </select>
            <input type="number" step="0.1" class="tgt-val-min" placeholder="Min" value="${tgt.min !== undefined ? tgt.min : (tgt.value !== undefined ? tgt.value : '')}" data-idx="${idx}">
            <input type="number" step="0.1" class="tgt-val-max" placeholder="Max" value="${tgt.max !== undefined ? tgt.max : ''}" data-idx="${idx}">
            <button class="remove-target-btn" data-idx="${idx}">&times;</button>
        `;
        
        container.appendChild(row);
    });
    
    // Bind listeners to updates
    container.querySelectorAll("select, input").forEach(el => {
        el.addEventListener("change", updateTargetFromUI);
    });
    
    container.querySelectorAll(".remove-target-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
            const idx = parseInt(e.target.getAttribute("data-idx"));
            state.targets.splice(idx, 1);
            renderTargets();
        });
    });
}

function updateTargetFromUI(e) {
    const idx = parseInt(e.target.getAttribute("data-idx"));
    const tgt = state.targets[idx];
    
    if (e.target.classList.contains("tgt-species")) {
        tgt.species = e.target.value;
        if (tgt.type === 'fold_change') tgt.output = tgt.species;
    } else if (e.target.classList.contains("tgt-type")) {
        tgt.type = e.target.value;
        // Sensible defaults for the new nonlinear-behaviour targets.
        if (tgt.type === 'oscillation' && tgt.min === undefined) tgt.min = 0.3;
        if (tgt.type === 'bistability' && tgt.min === undefined) tgt.min = 0.5;
        if (tgt.type === 'fold_change') {
            tgt.output = tgt.species;
            const nodes = state.blueprint ? state.blueprint.nodes.map(n => n.id) : [];
            tgt.input = nodes.find(n => n !== tgt.species) || tgt.species;
            if (tgt.max === undefined) tgt.max = 0.15;
        }
    } else if (e.target.classList.contains("tgt-val-min")) {
        const val = parseFloat(e.target.value);
        if (tgt.type === 'steady_state') {
            tgt.value = val;
            tgt.min = undefined;
        } else {
            tgt.min = val;
            tgt.value = undefined;
        }
    } else if (e.target.classList.contains("tgt-val-max")) {
        tgt.max = parseFloat(e.target.value);
    }
}

document.getElementById("add-target-btn").addEventListener("click", () => {
    const defaultSpecies = state.blueprint && state.blueprint.nodes.length > 0 ? state.blueprint.nodes[0].id : "ERK";
    state.targets.push({ species: defaultSpecies, type: "peak_time", min: 5, max: 20 });
    renderTargets();
});

// ==========================================================================
// API & NETWORK CALLS
// ==========================================================================
function updateStatus(text, type = "green") {
    const dot = document.querySelector(".status-dot");
    const textLabel = document.querySelector("#system-status span:last-child");
    
    dot.className = `status-dot ${type}`;
    textLabel.textContent = text;
}

// Resolve the current AI-engine configuration sent with LLM-backed requests.
function getLlmConfig() {
    const l = state.llm || {};
    if (l.engine === 'local') {
        return { engine: 'local', model: l.model || 'llama-3.2-3b' };
    }
    if (l.engine === 'remote') {
        return { engine: 'remote', base_url: l.remote_url || '', model: l.remote_model || '', api_key: l.remote_key || '' };
    }
    if (l.engine === 'bedrock') {
        return { engine: 'bedrock',
                 model: l.bedrock_model || 'mistral.mistral-large-3-675b-instruct',
                 region: l.bedrock_region || 'us-east-2',
                 aws_bearer_token: l.bedrock_bearer_token || '',
                 aws_access_key_id: l.bedrock_access_key || '',
                 aws_secret_access_key: l.bedrock_secret_key || '',
                 aws_session_token: l.bedrock_session_token || '' };
    }
    return { engine: 'off' };
}

// 1. Text Parsing & Compilation
document.getElementById("btn-parse-text").addEventListener("click", async () => {
    const text = document.getElementById("bio-input").value.trim();
    if (!text) return alert("Please type a biological process description first!");
    
    updateStatus("Parsing text...", "yellow");
    
    try {
        const response = await fetch("/api/blueprint", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: text,
                llm: getLlmConfig()
            })
        });

        if (!response.ok) throw new Error(await response.text());

        const parsedData = await response.json();

        // Surface any LLM fallback notice without blocking, then strip it.
        if (parsedData._llm_notice) {
            showToast(parsedData._llm_notice, 'warn', 5000);
            delete parsedData._llm_notice;
        }
        
        if (parsedData.validation_errors && parsedData.validation_errors.length > 0) {
            alert("⚠️ Model Compiler Validation Failed:\n\n" + parsedData.validation_errors.join("\n\n"));
            updateStatus("Validation Failed", "red");
            return;
        }
        
        state.blueprint = parsedData;
        
        // Update Blueprint JSON representation
        document.getElementById("blueprint-json-viewer").textContent = JSON.stringify(state.blueprint, null, 2);
        
        // Render network graph
        renderCytoscape();
        
        // Re-render targets dropdown species
        renderTargets();
        
        // Run compiler to get equations & sliders
        await compileBlueprint();
        
        // Switch to Blueprint Tab
        document.querySelector("[data-tab='blueprint']").click();
        
        updateStatus("Ready", "green");
    } catch (e) {
        console.error(e);
        updateStatus("Parsing failed", "red");
        alert("Compilation failed: " + e.message);
    }
});

async function compileBlueprint() {
    if (!state.blueprint) return;
    
    try {
        const response = await fetch("/api/compile", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ blueprint: state.blueprint })
        });
        
        const data = await response.json();
        state.equations = data.equations;
        state.equationsVerbose = data.equations_verbose || data.equations;

        // Render Equations via KaTeX
        renderEquations();
        
        // Load default values into parameters state
        state.customParams = {};
        if (data.parameters) {
            Object.assign(state.customParams, data.parameters);
        }
        
        // Set simulation tmax (a preset may request a longer horizon, e.g. oscillations)
        const tmax = state.tmaxOverride || state.blueprint.simulation_config?.t_max || 50;
        document.getElementById("sim-tmax").value = tmax;
        if (state.blueprint.simulation_config) state.blueprint.simulation_config.t_max = tmax;
        state.tmaxOverride = null;

        // RenderSliders
        renderParameterSliders();

        // Boundary/initial conditions panel + exploration target list
        renderBoundaryConditions();
        populateExploreTargets();
    } catch (e) {
        console.error("Compilation error", e);
    }
}

// Render sliders for ODE parameters
function renderParameterSliders() {
    const container = document.getElementById("parameters-sliders-container");
    container.innerHTML = "";
    
    if (state.blueprint.type === "PDE") {
        // PDEs don't compile standard ODE reaction rate sliders by default, but we can allow spatial variables.
        container.innerHTML = `
            <div class="slider-group">
                <div class="slider-labels">
                    <span class="slider-name">u diffusion</span>
                    <span class="slider-value" id="val-diff-u">${state.blueprint.spatial.diffusion.U || 0.05}</span>
                </div>
                <input type="range" min="0.001" max="0.5" step="0.001" value="${state.blueprint.spatial.diffusion.U || 0.05}" id="slide-diff-u">
            </div>
            <div class="slider-group">
                <div class="slider-labels">
                    <span class="slider-name">v diffusion</span>
                    <span class="slider-value" id="val-diff-v">${state.blueprint.spatial.diffusion.V || 1.0}</span>
                </div>
                <input type="range" min="0.1" max="5.0" step="0.1" value="${state.blueprint.spatial.diffusion.V || 1.0}" id="slide-diff-v">
            </div>
        `;
        
        const bindSlider = (slideId, valId, key) => {
            const el = document.getElementById(slideId);
            el.addEventListener("input", (e) => {
                const val = parseFloat(e.target.value);
                document.getElementById(valId).textContent = val;
                state.blueprint.spatial.diffusion[key] = val;
            });
        };
        bindSlider("slide-diff-u", "val-diff-u", "U");
        bindSlider("slide-diff-v", "val-diff-v", "V");
        return;
    }
    
    // Sort parameter names
    const sortedParamNames = Object.keys(state.customParams).sort();

    if (sortedParamNames.length === 0) {
        container.innerHTML = `<p class="placeholder-text">No adjustable parameters compiled.</p>`;
        return;
    }

    // Prune bounds for parameters that no longer exist
    Object.keys(state.paramBounds).forEach(p => {
        if (!(p in state.customParams)) delete state.paramBounds[p];
    });

    container.classList.toggle("show-bounds", !!state.showBounds);

    sortedParamNames.forEach(pname => {
        const val = state.customParams[pname];
        if (!state.paramBounds[pname]) state.paramBounds[pname] = defaultBounds(pname, val);
        const b = state.paramBounds[pname];

        const group = document.createElement("div");
        group.className = "slider-group";
        group.dataset.param = pname;

        group.innerHTML = `
            <div class="slider-labels">
                <span class="slider-name">${pname}</span>
                <input type="number" class="slider-value-input" id="val-${pname}" data-param="${pname}" value="${round2(val)}" step="${b.step}" title="Type an exact value">
            </div>
            <input type="range" min="${b.min}" max="${b.max}" step="${b.step}" value="${clampVal(val, b.min, b.max)}" data-param="${pname}">
            <div class="bounds-row">
                <span class="bounds-label">min</span>
                <input type="number" class="bound-min" data-param="${pname}" value="${b.min}" step="${b.step}">
                <span class="bounds-label">max</span>
                <input type="number" class="bound-max" data-param="${pname}" value="${b.max}" step="${b.step}">
                <button class="unit-toggle" data-param="${pname}" type="button" title="Set range to 0–1 (unitless)">0–1</button>
            </div>
        `;

        container.appendChild(group);
    });

    // Slider drag → update the editable value box
    container.querySelectorAll("input[type='range']").forEach(slider => {
        slider.addEventListener("input", (e) => {
            const pname = e.target.getAttribute("data-param");
            const val = parseFloat(e.target.value);
            const box = document.getElementById(`val-${pname}`);
            if (box) box.value = round2(val);
            state.customParams[pname] = val;
        });
    });

    // Typed value → move the slider (growing its range if the value is outside)
    container.querySelectorAll(".slider-value-input").forEach(inp => {
        inp.addEventListener("change", (e) => setParamValueFromInput(e.target));
    });

    // Bound editors → update slider range
    container.querySelectorAll(".bound-min, .bound-max").forEach(inp => {
        inp.addEventListener("change", (e) => updateBoundFromInput(e.target));
    });

    // 0–1 quick toggle (unitless)
    container.querySelectorAll(".unit-toggle").forEach(btn => {
        btn.addEventListener("click", (e) => {
            const pname = e.currentTarget.getAttribute("data-param");
            setBounds(pname, 0, 1);
        });
    });
}

// --- Parameter bounds helpers -------------------------------------------------
function clampVal(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
function round2(v) { return Math.round(v * 100) / 100; }

// Typed value box → update slider (and expand the range if the value is outside it)
function setParamValueFromInput(target) {
    const pname = target.dataset.param;
    const b = state.paramBounds[pname];
    const group = document.querySelector(`.slider-group[data-param="${pname}"]`);
    if (!group || !b) return;
    const slider = group.querySelector("input[type='range']");
    let v = parseFloat(target.value);
    if (isNaN(v)) { target.value = round2(state.customParams[pname]); return; }

    if (v > b.max) { b.max = v; group.querySelector(".bound-max").value = round2(v); }
    if (v < b.min) { b.min = v; group.querySelector(".bound-min").value = round2(v); }
    b.step = Math.max(1e-6, (b.max - b.min) / 100);

    slider.min = b.min;
    slider.max = b.max;
    slider.step = b.step;
    slider.value = v;
    state.customParams[pname] = v;
    target.value = round2(v);
}

function defaultBounds(pname, val) {
    let min = 0.0, max = 5.0, step = 0.05;
    if (pname.startsWith("deg_")) { min = 0.0; max = 2.0; step = 0.01; }
    else if (pname.startsWith("syn_")) { min = 0.0; max = 5.0; step = 0.05; }
    else if (pname.endsWith("_n")) { min = 1.0; max = 6.0; step = 0.5; }
    else if (pname.endsWith("_Kd")) { min = 0.1; max = 10.0; step = 0.1; }
    else if (pname.endsWith("_k")) { min = 0.0; max = 5.0; step = 0.05; }
    if (val > max) max = Math.ceil(val * 1.5);
    return { min, max, step };
}

function updateBoundFromInput(target) {
    const pname = target.getAttribute("data-param");
    const v = parseFloat(target.value);
    if (isNaN(v)) return;
    const b = state.paramBounds[pname];
    if (target.classList.contains("bound-min")) b.min = v; else b.max = v;
    if (b.max < b.min) { const t = b.min; b.min = b.max; b.max = t; }
    applyBoundsToSlider(pname);
}

function setBounds(pname, min, max) {
    const b = state.paramBounds[pname] || (state.paramBounds[pname] = defaultBounds(pname, state.customParams[pname]));
    b.min = min; b.max = max;
    applyBoundsToSlider(pname);
}

function applyBoundsToSlider(pname) {
    const b = state.paramBounds[pname];
    const group = document.querySelector(`.slider-group[data-param="${pname}"]`);
    if (!group) return;
    const span = Math.max(1e-6, b.max - b.min);
    b.step = span / 100;
    const slider = group.querySelector("input[type='range']");
    slider.min = b.min; slider.max = b.max; slider.step = b.step;
    let val = clampVal(parseFloat(slider.value), b.min, b.max);
    slider.value = val;
    state.customParams[pname] = val;
    const box = document.getElementById(`val-${pname}`);
    if (box) box.value = round2(val);
    group.querySelector(".bound-min").value = b.min;
    group.querySelector(".bound-max").value = b.max;
}

function renderEquations() {
    const container = document.getElementById("equations-container");
    container.innerHTML = "";

    const eqs = (state.eqMode === 'full' && state.equationsVerbose)
        ? state.equationsVerbose
        : state.equations;
    if (!eqs) return;

    Object.keys(eqs).forEach(var_name => {
        const latex_str = eqs[var_name];
        const item = document.createElement("div");
        item.className = "math-item";

        // Render using KaTeX
        katex.render(latex_str, item, { throwOnError: false, displayMode: true });
        container.appendChild(item);
    });
}

// 2. Database Queries
const dbTabButtons = document.querySelectorAll(".db-tab-btn");
dbTabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
        dbTabButtons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.currentDb = btn.getAttribute("data-db");
        document.getElementById("db-query-input").placeholder = 
            state.currentDb === 'reactome' ? "Search pathways (e.g. EGFR)..." : "Enter proteins (e.g. EGFR, ERK)...";
    });
});

document.getElementById("btn-db-search").addEventListener("click", async () => {
    const query = document.getElementById("db-query-input").value.trim();
    if (!query) return;
    const db = state.currentDb;

    // Protein-interaction databases → open the Connection Library so each
    // interaction can be reviewed and picked individually (instead of dumping
    // all of them into the description box at once).
    if (db === 'string' || db === 'omnipath' || db === 'signor') {
        openConnModalWith(db, query);
        return;
    }

    const container = document.getElementById("db-results-container");
    container.innerHTML = `<p class="placeholder-text">Searching database...</p>`;

    try {
        if (db === 'reactome') {
            const response = await fetch(`/api/reactome/search?q=${encodeURIComponent(query)}`);
            const data = await response.json();

            container.innerHTML = "";
            if (data.length === 0) {
                container.innerHTML = `<p class="placeholder-text">No pathways found.</p>`;
                return;
            }

            data.forEach(item => {
                const div = document.createElement("div");
                div.className = "search-result-item";
                div.innerHTML = `
                    <strong>${item.name}</strong><br>
                    <span style="font-size:10px; color:var(--text-secondary);">${item.id} | ${item.species}</span>
                `;
                div.addEventListener("click", () => loadReactomePathwayText(item.id, item.name));
                container.appendChild(div);
            });
        } else if (db === 'biomodels') {
            const response = await fetch(`/api/biomodels/search?q=${encodeURIComponent(query)}`);
            const data = await response.json();

            container.innerHTML = "";
            if (!data.length) {
                container.innerHTML = `<p class="placeholder-text">No models found.</p>`;
                return;
            }

            data.forEach(item => {
                const div = document.createElement("div");
                div.className = "search-result-item";
                div.innerHTML = `
                    <strong>${item.id}</strong><br>
                    <span style="font-size:11px; color:var(--text-primary);">${item.name}</span><br>
                    <span style="font-size:10px; color:var(--text-secondary);">${(item.description || '').substring(0, 90)}</span>
                `;
                div.addEventListener("click", () => {
                    document.getElementById("biomodel-id-input").value = item.id;
                    document.querySelector("[data-tab='maple']").click();
                });
                container.appendChild(div);
            });
        }
    } catch (e) {
        container.innerHTML = `<p class="placeholder-text" style="color:var(--accent-red)">Search failed.</p>`;
    }
});

async function loadReactomePathwayText(pathwayId, pathwayName) {
    updateStatus("Fetching pathway reactions...", "yellow");
    const container = document.getElementById("db-results-container");
    
    try {
        const response = await fetch(`/api/reactome/reactions?pathway_id=${pathwayId}`);
        const reactions = await response.json();

        const lines = [`# Pathway: ${pathwayName} (${pathwayId})`];

        // Derive protein pairs from reaction display names (uppercase gene-like tokens)
        const pairs = [];
        (reactions || []).forEach(rxn => {
            const tokens = (rxn.name || '')
                .replace(/[(),:;]/g, ' ')
                .split(/\s+/)
                .filter(w => w.length >= 2 && /^[A-Z][A-Z0-9-]*[A-Z0-9]$/.test(w) && isNaN(w));
            const uniq = [...new Set(tokens)];
            if (uniq.length >= 2) pairs.push([uniq[0], uniq[1]]);
        });

        if (pairs.length > 0) {
            const nodes = new Set();
            pairs.forEach(([a, b]) => { lines.push(`${a} activates ${b}.`); nodes.add(a); nodes.add(b); });
            lines.push(`${pairs[0][0]} starts at 10.0.`);
            document.getElementById("bio-input").value = lines.join('\n');
            showToast(`Loaded ${pairs.length} interaction${pairs.length !== 1 ? 's' : ''} from “${pathwayName}”. Review, then Compile.`, 'success');
        } else if ((reactions || []).length > 0) {
            // We have reactions but couldn't derive clean edges; list them as notes
            reactions.slice(0, 20).forEach(rxn => lines.push(`# ${rxn.name}`));
            lines.push(`# Couldn't auto-derive edges. Edit above, or use the 📖 Connection Library for these proteins.`);
            document.getElementById("bio-input").value = lines.join('\n');
            showToast(`Loaded ${reactions.length} reaction names from “${pathwayName}” as notes.`, 'info');
        } else {
            lines.push(`# No machine-readable reactions in this pathway.`);
            lines.push(`# Try the 📖 Connection Library with its proteins instead.`);
            document.getElementById("bio-input").value = lines.join('\n');
            showToast(`“${pathwayName}” has no parseable reactions. Try the Connection Library.`, 'warn');
        }

        updateStatus("Ready", "green");
    } catch (e) {
        updateStatus("Ready", "green");
        showToast("Failed to load pathway details.", 'error');
    }
}

// 3. Simulation Solver
document.getElementById("btn-run-simulation").addEventListener("click", runSimulation);

async function runSimulation() {
    if (!state.blueprint) return alert("Please compile a blueprint first!");
    
    updateStatus("Simulating model...", "yellow");
    
    // Read simulation tmax
    const tmax = parseFloat(document.getElementById("sim-tmax").value);
    state.blueprint.simulation_config.t_max = tmax;
    
    try {
        const response = await fetch("/api/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                blueprint: state.blueprint,
                custom_params: state.customParams
            })
        });
        
        if (!response.ok) throw new Error(await response.text());
        
        state.simulationResults = await response.json();
        
        // Display plots
        renderVisualizer();
        
        // Evaluate targets for the feedback screen
        await evaluateSimulationTargets();

        updateStatus("Ready", "green");
    } catch (e) {
        console.error(e);
        updateStatus("Simulation failed", "red");
        alert("Simulation failed: " + e.message);
    }
}

// ==========================================
// VISUALIZATION RENDERERS (CHART.JS & CANVAS)
// ==========================================
function renderVisualizer() {
    const isPde = state.blueprint.type === "PDE";
    const odeCanvas = document.getElementById("ode-chart");
    const pdeCanvas = document.getElementById("pde-canvas");
    const pdeControls = document.getElementById("pde-controls");
    
    // Stop any running animations
    stopPdeAnimation();
    
    if (isPde) {
        odeCanvas.style.display = "none";
        pdeCanvas.style.display = "block";
        pdeControls.style.display = "flex";
        document.getElementById("vis-title").textContent = "Spatial Pattern Dynamics (2D Grid)";
        
        setupPdeVisualizer();
    } else {
        odeCanvas.style.display = "block";
        pdeCanvas.style.display = "none";
        pdeControls.style.display = "none";
        document.getElementById("vis-title").textContent = "Time-Series Concentration Dynamics";
        
        renderOdeChart();
    }
}

function renderOdeChart() {
    const t = state.simulationResults.t;
    const species = state.simulationResults.species;
    const ctx = document.getElementById("ode-chart").getContext("2d");

    const datasets = [];
    const colors = [
        '#00f2fe', '#b180ff', '#00ff87', '#ff007f', '#4facfe', '#ffaa00', '#f43f5e', '#10b981'
    ];

    // A model may declare plot_species to spotlight its key readouts (e.g. a
    // multi-state kinase cascade plots only the stimulus + total activity).
    let plotList = Object.keys(species);
    if (state.blueprint && Array.isArray(state.blueprint.plot_species)) {
        const wanted = state.blueprint.plot_species.filter(s => s in species);
        if (wanted.length) plotList = wanted;
    }

    plotList.forEach((nid, index) => {
        const color = colors[index % colors.length];
        datasets.push({
            label: nid,
            data: species[nid],
            borderColor: color,
            backgroundColor: color + '11',
            borderWidth: 2,
            tension: 0.15,
            pointRadius: 1,
            pointHoverRadius: 5
        });
    });
    
    if (chartInstance) {
        chartInstance.destroy();
    }
    
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: t.map(val => val.toFixed(1)),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' },
                    title: { display: true, text: 'Time (minutes)', color: '#9ca3af' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' },
                    title: { display: true, text: 'Concentration (units)', color: '#9ca3af' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#f3f4f6', font: { family: 'Outfit' } }
                }
            }
        }
    });
}

function setupPdeVisualizer() {
    const results = state.simulationResults;
    const speciesList = Object.keys(results.species);
    
    const select = document.getElementById("pde-species-select");
    select.innerHTML = "";
    speciesList.forEach(name => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
    });
    
    state.pdeSelectedSpecies = speciesList[0];
    
    // Set Slider Limits
    const numFrames = results.species[state.pdeSelectedSpecies].length;
    const slider = document.getElementById("pde-frame-slider");
    slider.max = numFrames - 1;
    slider.value = 0;
    state.pdeCurrentFrame = 0;
    
    // Draw initial frame
    drawPdeFrame();
    
    // Play automatically
    startPdeAnimation();
}

function drawPdeFrame() {
    const canvas = document.getElementById("pde-canvas");
    const ctx = canvas.getContext("2d");
    const results = state.simulationResults;
    
    const xSize = results.x_size;
    const ySize = results.y_size;
    
    // Set canvas dimensions relative to grid scale
    canvas.width = xSize * 5;
    canvas.height = ySize * 5;
    
    const frameData = results.species[state.pdeSelectedSpecies][state.pdeCurrentFrame];
    if (!frameData) return;
    
    // Find min/max for normalization
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < xSize; i++) {
        for (let j = 0; j < ySize; j++) {
            const val = frameData[i][j];
            if (val < min) min = val;
            if (val > max) max = val;
        }
    }
    
    const scale = max - min;
    
    // Draw cells
    const cellW = 5;
    const cellH = 5;
    for (let i = 0; i < xSize; i++) {
        for (let j = 0; j < ySize; j++) {
            const rawVal = frameData[i][j];
            const normVal = scale > 0.001 ? (rawVal - min) / scale : 0.5;
            
            // Render with plasma palette
            ctx.fillStyle = PDE_PALETTES.plasma(normVal);
            ctx.fillRect(i * cellW, j * cellH, cellW, cellH);
        }
    }
}

function startPdeAnimation() {
    if (state.pdePlayInterval) clearInterval(state.pdePlayInterval);
    
    state.pdePlaying = true;
    document.getElementById("btn-pde-play").textContent = "Pause";
    
    const results = state.simulationResults;
    const numFrames = results.species[state.pdeSelectedSpecies].length;
    
    state.pdePlayInterval = setInterval(() => {
        state.pdeCurrentFrame = (state.pdeCurrentFrame + 1) % numFrames;
        document.getElementById("pde-frame-slider").value = state.pdeCurrentFrame;
        drawPdeFrame();
    }, 100);
}

function stopPdeAnimation() {
    state.pdePlaying = false;
    document.getElementById("btn-pde-play").textContent = "Play";
    if (state.pdePlayInterval) {
        clearInterval(state.pdePlayInterval);
        state.pdePlayInterval = null;
    }
}

// Bind PDE Playback Controls
document.getElementById("btn-pde-play").addEventListener("click", () => {
    if (state.pdePlaying) {
        stopPdeAnimation();
    } else {
        startPdeAnimation();
    }
});

document.getElementById("btn-pde-pause").addEventListener("click", stopPdeAnimation);

document.getElementById("pde-frame-slider").addEventListener("input", (e) => {
    stopPdeAnimation();
    state.pdeCurrentFrame = parseInt(e.target.value);
    drawPdeFrame();
});

document.getElementById("pde-species-select").addEventListener("change", (e) => {
    state.pdeSelectedSpecies = e.target.value;
    drawPdeFrame();
});

// ==========================================
// CYTOSCAPE GRAPH RENDERER
// ==========================================
function renderCytoscape() {
    const container = document.getElementById("cy-container");
    if (!state.blueprint || !state.blueprint.nodes) return;
    
    // Format nodes and edges for Cytoscape
    const elements = [];
    const nodeIds = new Set();

    state.blueprint.nodes.forEach(node => {
        nodeIds.add(node.id);
        elements.push({
            data: {
                id: node.id,
                label: `${node.id}\n(${node.initial_value || 0})`
            }
        });
    });

    // Defensive: only draw edges whose endpoints are declared nodes. Cytoscape
    // throws on a dangling edge, which would crash the whole compile.
    (state.blueprint.edges || []).forEach((edge, index) => {
        if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) return;
        elements.push({
            data: {
                id: `e${index}`,
                source: edge.source,
                target: edge.target,
                type: edge.type // activation / inhibition
            }
        });
    });
    
    cyInstance = cytoscape({
        container: container,
        elements: elements,
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': 'rgba(138, 43, 226, 0.8)',
                    'border-width': '2px',
                    'border-color': '#00f2fe',
                    'label': 'data(label)',
                    'color': '#f3f4f6',
                    'font-family': 'Outfit',
                    'font-size': '11px',
                    'text-wrap': 'wrap',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'width': '65px',
                    'height': '65px',
                    'box-shadow': '0 0 10px rgba(0,242,254,0.3)'
                }
            },
            {
                selector: 'edge[type="activation"]',
                style: {
                    'width': 3,
                    'line-color': '#00ff87',
                    'target-arrow-color': '#00ff87',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier'
                }
            },
            {
                selector: 'edge[type="inhibition"]',
                style: {
                    'width': 3,
                    'line-color': '#ff007f',
                    'target-arrow-color': '#ff007f',
                    'target-arrow-shape': 'tee',
                    'curve-style': 'bezier'
                }
            }
        ],
        layout: {
            name: 'cose',
            padding: 30
        }
    });
}

// ==========================================
// CLOSED-LOOP AGENT LOOP (STAGE 3)
// ==========================================
// Evaluate targets on the backend so multi-condition behaviours (bistability,
// fold-change) are judged with the extra simulations they require.
async function evaluateSimulationTargets() {
    if (!state.blueprint || state.targets.length === 0) return 0;

    const container = document.getElementById("target-eval-list");
    let results;
    try {
        const resp = await fetch("/api/evaluate", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                blueprint: state.blueprint,
                targets: state.targets,
                custom_params: state.customParams
            })
        });
        const data = await resp.json();
        results = data.results || [];
    } catch (e) {
        container.innerHTML = `<p class="placeholder-text err">Target evaluation failed: ${e.message}</p>`;
        return 0;
    }

    container.innerHTML = "";
    let metCount = 0;
    results.forEach(r => {
        if (r.met) metCount++;
        const item = document.createElement("div");
        item.className = `eval-item ${r.met ? 'met' : 'failed'}`;
        item.innerHTML = `
            <div class="eval-icon">${r.met ? '🟢' : '🔴'}</div>
            <div class="eval-details">
                <div class="target-title">${r.species || ''} ${(r.type || '').replace(/_/g, ' ')}</div>
                <div class="target-status-msg">${r.detail || ''}</div>
            </div>
        `;
        container.appendChild(item);
    });

    // Update circular progress
    const pct = Math.round((metCount / state.targets.length) * 100);
    document.getElementById("target-score-percentage").textContent = `${pct}%`;
    const circle = document.getElementById("target-progress-bar");
    const radius = circle.r.baseVal.value;
    const circumference = radius * 2 * Math.PI;
    circle.style.strokeDashoffset = circumference - (pct / 100) * circumference;

    return metCount;
}

// Helpers for array calculations in js
function np_array(arr) { return arr; }

// --- Oscillation detectors (mirror agent.py) ---
function relAmplitude(seg) {
    if (!seg.length) return 0;
    let mn = Infinity, mx = -Infinity, sum = 0;
    for (const v of seg) { if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
    return (mx - mn) / (Math.abs(sum / seg.length) + 1e-6);
}
function sustainedOscAmplitude(y) {
    if (!y || y.length < 8 || y.some(v => !isFinite(v))) return 0;
    const L = y.length;
    const q3 = y.slice(Math.floor(L / 2), Math.floor(3 * L / 4));
    const q4 = y.slice(Math.floor(3 * L / 4));
    return Math.min(relAmplitude(q3), relAmplitude(q4));
}
function countSustainedPeaks(y) {
    if (!y || y.length < 6) return 0;
    const seg = y.slice(Math.floor(y.length / 2));
    let mn = Infinity, mx = -Infinity;
    for (const v of seg) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const rng = mx - mn;
    if (rng < 1e-9) return 0;
    const thr = 0.05 * rng;
    let n = 0;
    for (let i = 1; i < seg.length - 1; i++) {
        if (seg[i - 1] < seg[i] && seg[i] > seg[i + 1] && (seg[i] - Math.min(seg[i - 1], seg[i + 1])) > thr) n++;
    }
    return n;
}

function argmax(arr) {
    let max = -Infinity;
    let idx = -1;
    for(let i=0; i<arr.length; i++) {
        if(arr[i] > max) { max = arr[i]; idx = i; }
    }
    return idx;
}

// Log writer for Console
function writeConsole(text, type = "info") {
    const consoleLog = document.getElementById("feedback-log");
    
    // Clear placeholder
    const ph = consoleLog.querySelector(".log-placeholder");
    if (ph) ph.remove();
    
    const line = document.createElement("div");
    line.className = `log-line ${type}`;
    line.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    consoleLog.appendChild(line);
    consoleLog.scrollTop = consoleLog.scrollHeight;
}

// Feedback Refinement Iteration Loop
document.getElementById("btn-run-feedback-loop").addEventListener("click", runClosedLoopFeedback);

async function runClosedLoopFeedback() {
    if (!state.blueprint) return alert("Please compile a blueprint first!");
    if (state.blueprint.type === "PDE") {
        showToast("Closed-loop optimization is available for ODE models.", "warn");
        return;
    }
    if (!state.targets.length) {
        showToast("Add at least one target behavior first.", "warn");
        return;
    }

    // Switch to Feedback Tab
    document.querySelector("[data-tab='feedback']").click();

    document.getElementById("feedback-log").innerHTML = "";
    writeConsole("Starting automated closed-loop optimization...", "info");
    updateStatus("Optimizing...", "yellow");

    const maxIterations = 20;      // safety cap (usually converges in 1–2)
    const stagnationLimit = 3;     // stop if no improvement for this many rounds
    const total = state.targets.length;
    const tmax = parseFloat(document.getElementById("sim-tmax").value);

    let iteration = 1;
    let allMet = false;
    let stagnation = 0;
    let best = { met: -1, blueprint: null };

    // Simulate the current blueprint, render, and return how many targets are met.
    const simulateCurrent = async () => {
        state.blueprint.simulation_config.t_max = tmax;
        const resp = await fetch("/api/simulate", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ blueprint: state.blueprint, custom_params: state.customParams })
        });
        if (!resp.ok) throw new Error(await resp.text());
        state.simulationResults = await resp.json();
        renderOdeChart();
        return await evaluateSimulationTargets();
    };

    while (iteration <= maxIterations && !allMet) {
        writeConsole(`--- ITERATION ${iteration} / ${maxIterations} ---`, "info");

        let met;
        try {
            writeConsole("Simulating model...", "info");
            met = await simulateCurrent();
        } catch (e) {
            writeConsole(`Simulation failed: ${e.message}`, "error");
            break;
        }

        writeConsole(`${met} / ${total} targets met.`, met === total ? "success" : "info");

        // Track the best configuration seen so far.
        if (met > best.met) {
            best = { met, blueprint: JSON.parse(JSON.stringify(state.blueprint)) };
            stagnation = 0;
        } else {
            stagnation++;
        }

        if (met === total) { allMet = true; break; }
        if (stagnation >= stagnationLimit) {
            writeConsole(`No improvement for ${stagnationLimit} rounds, stopping early.`, "warning");
            break;
        }

        // Refine: numerical optimizer fits the parameters to the targets.
        writeConsole("Fitting parameters to targets (numerical optimizer)…", "info");
        try {
            const refineData = await (await fetch("/api/refine", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    blueprint: state.blueprint,
                    simulation_results: state.simulationResults,
                    targets: state.targets,
                    llm: getLlmConfig()
                })
            })).json();

            (refineData.logs || []).forEach(l =>
                writeConsole(l, l.startsWith("SUCCESS") ? "success" : (l.startsWith("Action") ? "warning" : "info")));

            state.blueprint = refineData.blueprint;
            await compileBlueprint();   // pull the tuned parameters into the sliders
        } catch (e) {
            writeConsole(`Optimization request failed: ${e.message}`, "error");
            break;
        }

        iteration++;
        await new Promise(r => setTimeout(r, 200));
    }

    // If we didn't fully converge, restore the best configuration we found.
    if (!allMet && best.blueprint) {
        state.blueprint = best.blueprint;
        await compileBlueprint();
        try { await simulateCurrent(); } catch (e) { /* ignore */ }
    }

    updateStatus("Ready", "green");
    if (allMet) {
        writeConsole("OPTIMIZATION COMPLETE: Model meets all target constraints!", "success");
        showToast("✓ All target conditions met.", "success");
    } else {
        writeConsole(`Stopped after ${iteration} round(s). Best result: ${best.met} / ${total} targets met.`, "error");
        showToast(`Best: ${best.met}/${total} targets met. Try widening target ranges or adding a feedback edge.`, "warn", 7000);
    }
}

// Bind slider inputs
function initEventListeners() {
    // ABM Events (only registered here)
    document.querySelectorAll(".abm-preset-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
            document.querySelectorAll(".abm-preset-btn").forEach(b => b.classList.remove("active-preset"));
            e.target.classList.add("active-preset");
            loadAbmPreset(e.target.getAttribute("data-preset"));
        });
    });
    document.getElementById("btn-run-abm").addEventListener("click", runAbmSimulation);
    document.getElementById("btn-abm-play").addEventListener("click", () => toggleAbmPlayback(true));
    document.getElementById("btn-abm-pause").addEventListener("click", () => toggleAbmPlayback(false));
    document.getElementById("abm-frame-slider").addEventListener("input", (e) => {
        state.abmPlaying = false;
        state.abmCurrentFrame = parseInt(e.target.value);
        renderAbmFrame(state.abmCurrentFrame);
    });

    // MAPLE Events (only registered here)
    document.getElementById("btn-maple-extract").addEventListener("click", extractMapleParameters);
    document.getElementById("btn-import-sbml").addEventListener("click", importBioModelSbml);
}

// ==========================================
// EXPANDED DATABASE RAG (OmniPath, SIGNOR, BioModels)
// ==========================================
async function searchDatabase() {
    const query = document.getElementById("db-query-input").value;
    if (!query) return;
    
    const container = document.getElementById("db-results-container");
    container.innerHTML = "Searching...";
    
    try {
        let endpoint = "";
        let body = null;
        let method = "GET";
        
        if (state.currentDb === "reactome") {
            endpoint = `/api/reactome/search?q=${encodeURIComponent(query)}`;
        } else if (state.currentDb === "string") {
            endpoint = `/api/string/network`;
            method = "POST";
            body = JSON.stringify({ proteins: query.split(',').map(s => s.trim()) });
        } else if (state.currentDb === "omnipath") {
            endpoint = `/api/omnipath/interactions`;
            method = "POST";
            body = JSON.stringify({ proteins: query.split(',').map(s => s.trim()) });
        } else if (state.currentDb === "signor") {
            endpoint = `/api/signor/search?q=${encodeURIComponent(query)}`;
        } else if (state.currentDb === "biomodels") {
            endpoint = `/api/biomodels/search?q=${encodeURIComponent(query)}`;
        }
        
        const response = await fetch(endpoint, {
            method: method,
            headers: { "Content-Type": "application/json" },
            body: body
        });
        
        const results = await response.json();
        container.innerHTML = "";
        
        if (results.length === 0) {
            container.innerHTML = "<p class='placeholder-text'>No results found.</p>";
            return;
        }
        
        results.forEach(res => {
            const item = document.createElement("div");
            item.className = "search-result-item";
            
            if (state.currentDb === "reactome") {
                item.innerHTML = `<strong>${res.id}</strong><br>${res.name}`;
            } else if (state.currentDb === "string" || state.currentDb === "omnipath") {
                const isOmni = state.currentDb === "omnipath";
                item.innerHTML = `<strong>${res.source} → ${res.target}</strong><br>
                                  Type: ${res.type} | Score: ${res.score.toFixed(2)}
                                  ${isOmni && res.references ? `<br>Refs: ${res.references.split(';')[0]}` : ''}`;
                item.addEventListener("click", () => {
                    const text = document.getElementById("bio-input");
                    const action = res.type === "inhibition" ? "inhibits" : "activates";
                    text.value += `\n${res.source} ${action} ${res.target}.`;
                });
            } else if (state.currentDb === "signor") {
                item.innerHTML = `<strong>${res.source} → ${res.target}</strong><br>
                                  Mech: ${res.mechanism} | Effect: ${res.effect}<br>
                                  PMID: ${res.pmid}`;
                item.addEventListener("click", () => {
                    const text = document.getElementById("bio-input");
                    const action = res.type === "inhibition" ? "inhibits" : "activates";
                    text.value += `\n${res.source} ${action} ${res.target}.`;
                });
            } else if (state.currentDb === "biomodels") {
                item.innerHTML = `<strong>${res.id}</strong>: ${res.name}<br>
                                  <span style="color:#6b7280;font-size:10px">${res.description.substring(0,80)}...</span>`;
                item.addEventListener("click", () => {
                    document.getElementById("biomodel-id-input").value = res.id;
                    document.querySelector("[data-tab='maple']").click();
                });
            }
            container.appendChild(item);
        });
        
    } catch(e) {
        container.innerHTML = `<p class="placeholder-text" style="color:var(--accent-red)">Error: ${e.message}</p>`;
    }
}

// ==========================================
// AGENT-BASED MODELING (ABM) LOGIC
// ==========================================
let abmResult = null;

async function loadAbmPreset(name) {
    try {
        const response = await fetch(`/api/abm/preset/${name}`);
        const bp = await response.json();
        state.abmBlueprint = bp;
        
        // Render info panel
        const panel = document.getElementById("abm-info-panel");
        let html = `<div class="abm-info-title">${bp.name}</div>
                    <div class="abm-info-desc">${bp.description}</div>
                    <div class="abm-cell-type-list">`;
        
        bp.cell_types.forEach(ct => {
            const color = `rgb(${ct.color[0]},${ct.color[1]},${ct.color[2]})`;
            html += `<div class="cell-type-chip">
                        <span class="cell-type-swatch" style="background:${color}"></span>
                        ${ct.name}
                     </div>`;
        });
        html += `</div>`;
        panel.innerHTML = html;
        
    } catch(e) {
        console.error("Failed to load ABM preset", e);
    }
}

async function runAbmSimulation() {
    if (!state.abmBlueprint) return alert("Select an ABM preset first!");
    
    document.getElementById("abm-canvas").style.display = "none";
    document.getElementById("btn-run-abm").textContent = "Simulating...";
    updateStatus("Running CPM...", "yellow");
    
    try {
        const response = await fetch("/api/abm/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ blueprint: state.abmBlueprint })
        });
        
        abmResult = await response.json();
        
        document.getElementById("abm-canvas").style.display = "block";
        document.getElementById("btn-run-abm").textContent = "Run ABM Simulation";
        updateStatus("Ready", "green");
        
        // Init player
        document.getElementById("abm-playback").style.display = "flex";
        const slider = document.getElementById("abm-frame-slider");
        slider.max = abmResult.lattice_frames.length - 1;
        slider.value = 0;
        
        state.abmCurrentFrame = 0;
        renderAbmFrame(0);
        toggleAbmPlayback(true);
        
    } catch(e) {
        alert("ABM Simulation failed: " + e.message);
        document.getElementById("btn-run-abm").textContent = "Run ABM Simulation";
        updateStatus("Error", "red");
    }
}

function renderAbmFrame(frameIdx) {
    if (!abmResult) return;
    
    const canvas = document.getElementById("abm-canvas");
    const ctx = canvas.getContext("2d");
    
    const width = abmResult.width;
    const height = abmResult.height;
    
    // Set canvas resolution
    canvas.width = width;
    canvas.height = height;
    
    const frameData = abmResult.lattice_frames[frameIdx];
    const imgData = ctx.createImageData(width, height);
    
    // Flat RGB array mapping
    for (let x = 0; x < width; x++) {
        for (let y = 0; y < height; y++) {
            const idx = (y * width + x) * 4;
            const rgb = frameData[x][y];
            imgData.data[idx] = rgb[0];
            imgData.data[idx+1] = rgb[1];
            imgData.data[idx+2] = rgb[2];
            imgData.data[idx+3] = 255; // Alpha
        }
    }
    ctx.putImageData(imgData, 0, 0);
    
    document.getElementById("abm-frame-label").textContent = 
        `MCS: ${abmResult.t[frameIdx]} / ${abmResult.total_mcs}`;
        
    // Update cell counts
    const counts = abmResult.cell_counts[frameIdx];
    const countPanel = document.getElementById("abm-cell-counts");
    let cHtml = '';
    for (const [name, val] of Object.entries(counts)) {
        if (name === "total") continue;
        cHtml += `<div class="cell-count-row">
                    <span class="cell-count-label">${name}</span>
                    <span class="cell-count-value">${val}</span>
                  </div>`;
    }
    countPanel.innerHTML = cHtml;
}

function toggleAbmPlayback(play) {
    state.abmPlaying = play;
    const playBtn = document.getElementById("btn-abm-play");
    const pauseBtn = document.getElementById("btn-abm-pause");
    
    if (play) {
        playBtn.style.background = "rgba(138, 43, 226, 0.4)";
        pauseBtn.style.background = "";
        
        if (state.abmPlayInterval) clearInterval(state.abmPlayInterval);
        state.abmPlayInterval = setInterval(() => {
            if (state.abmCurrentFrame >= abmResult.lattice_frames.length - 1) {
                toggleAbmPlayback(false);
                return;
            }
            state.abmCurrentFrame++;
            document.getElementById("abm-frame-slider").value = state.abmCurrentFrame;
            renderAbmFrame(state.abmCurrentFrame);
        }, 100);
    } else {
        pauseBtn.style.background = "rgba(255,255,255,0.2)";
        playBtn.style.background = "";
        if (state.abmPlayInterval) clearInterval(state.abmPlayInterval);
    }
}


// ==========================================
// MAPLE CALIBRATION LOGIC
// ==========================================
async function extractMapleParameters() {
    const paramName = document.getElementById("maple-param-name").value;
    if (!paramName) return alert("Parameter Name is required.");
    
    const units = document.getElementById("maple-param-units").value;
    const desc = document.getElementById("maple-param-desc").value;
    const context = document.getElementById("maple-param-context").value;
    
    const btn = document.getElementById("btn-maple-extract");
    btn.textContent = "Extracting...";
    btn.disabled = true;
    updateStatus("LLM Extracting...", "yellow");
    
    try {
        const response = await fetch("/api/maple/extract", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                param_name: paramName,
                param_units: units,
                param_description: desc,
                mechanistic_context: context,
                llm: getLlmConfig()
            })
        });
        
        const result = await response.json();
        
        // Render JSON viewer
        const resContainer = document.getElementById("maple-results-container");
        resContainer.innerHTML = `<pre class="maple-target-viewer">${JSON.stringify(result.target, null, 2)}</pre>`;
        
        // Render Validation Badges
        const valContainer = document.getElementById("maple-validation-container");
        valContainer.innerHTML = `<h4 class="subsection-title">Validation Checks</h4>`;
        
        if (result.validation && result.validation.results) {
            result.validation.results.forEach(val => {
                const div = document.createElement("div");
                div.className = "validation-result-item";
                
                let badgeClass = "pass";
                let badgeText = "✓";
                if (!val.passed) {
                    badgeClass = val.severity === "error" ? "fail" : "warn";
                    badgeText = val.severity === "error" ? "✗" : "!";
                }
                
                div.innerHTML = `
                    <div class="validation-badge ${badgeClass}">${badgeText}</div>
                    <div class="validation-msg">${val.message}</div>
                    ${val.field_path ? `<div class="validation-field">${val.field_path}</div>` : ''}
                `;
                valContainer.appendChild(div);
            });
        }
        
        // Render Logs
        const logContainer = document.getElementById("maple-logs-container");
        logContainer.innerHTML = result.logs.map(l => `<div class="log-line ${l.includes('fail')||l.includes('error') ? 'error' : 'info'}">${l}</div>`).join("");
        
    } catch(e) {
        alert("Extraction failed: " + e.message);
    } finally {
        btn.textContent = "Extract & Validate";
        btn.disabled = false;
        updateStatus("Ready", "green");
    }
}

async function importBioModelSbml() {
    const modelId = document.getElementById("biomodel-id-input").value;
    if (!modelId) return alert("Enter a BioModels ID (e.g., BIOMD0000000006)");
    
    const btn = document.getElementById("btn-import-sbml");
    btn.textContent = "Importing...";
    
    try {
        const response = await fetch("/api/biomodels/import", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_id: modelId })
        });
        
        if (!response.ok) throw new Error("Failed to import model");
        
        const result = await response.json();
        
        // Set blueprint
        state.blueprint = result.blueprint;
        document.getElementById("blueprint-json-viewer").textContent = JSON.stringify(state.blueprint, null, 2);
        
        // Switch to Blueprint tab
        document.querySelector("[data-tab='blueprint']").click();
        
        // Attempt compile and render
        await compileBlueprint();
        
        alert(`Successfully imported SBML model: ${modelId}`);
        
    } catch(e) {
        alert("SBML Import failed: " + e.message);
    } finally {
        btn.textContent = "Import SBML";
    }
}


// ==========================================================================
// BOUNDARY / INITIAL CONDITIONS PANEL  (ODE interface)
// ==========================================================================
function renderBoundaryConditions() {
    const content = document.getElementById("bc-content");
    if (!content) return;
    if (!state.blueprint) {
        content.innerHTML = `<p class="placeholder-text">Compile a model to view its conditions.</p>`;
        return;
    }
    const bp = state.blueprint;
    let html = "";

    if (bp.type === "PDE") {
        html += `<div class="bc-note"><span class="tag">PDE</span> Spatial boundary condition:
                 <b>Zero-flux (Neumann)</b> on all four domain edges.</div>`;
        html += `<div class="bc-note"><span class="tag">t = 0</span> Initial field: base value per species with small random noise.</div>`;
    } else {
        html += `<div class="bc-note"><span class="tag">ODE</span> Initial conditions
                 <span class="muted">(system state at t = 0; edit to change starting abundances)</span></div>`;
    }

    html += `<div class="bc-list">`;
    (bp.nodes || []).forEach(n => {
        const v = (n.initial_value !== undefined && n.initial_value !== null) ? n.initial_value : 0;
        html += `<div class="bc-row">
            <span class="bc-species">${n.id}</span>
            <span class="bc-eq">[${n.id}]<sub>0</sub></span>
            <input type="number" class="bc-input" data-node="${n.id}" value="${v}" step="0.1">
        </div>`;
    });
    html += `</div>`;
    content.innerHTML = html;

    content.querySelectorAll(".bc-input").forEach(inp => {
        inp.addEventListener("change", (e) => {
            const id = e.target.getAttribute("data-node");
            const v = parseFloat(e.target.value);
            if (isNaN(v)) return;
            const node = state.blueprint.nodes.find(x => x.id === id);
            if (!node) return;
            node.initial_value = v;
            document.getElementById("blueprint-json-viewer").textContent = JSON.stringify(state.blueprint, null, 2);
            if (cyInstance) {
                const el = cyInstance.getElementById(id);
                if (el) el.data("label", `${id}\n(${v})`);
            }
        });
    });
}

function populateExploreTargets() {
    const sel = document.getElementById("explore-target");
    if (!sel || !state.blueprint) return;
    const ids = (state.blueprint.nodes || []).map(n => n.id);
    const prev = sel.value;
    sel.innerHTML = ids.map(id => `<option value="${id}">${id}</option>`).join("");
    if (ids.includes(prev)) sel.value = prev;
    else if (ids.length) sel.value = ids[ids.length - 1]; // default to most-downstream node
}


// ==========================================================================
// PARAMETER-SPACE EXPLORATION  (Latin Hypercube sampling)
// ==========================================================================
let exploreSort = { key: 'id', dir: 1 };

async function runExploration() {
    if (!state.blueprint) return alert("Please compile a blueprint first!");
    if (state.blueprint.type === "PDE") {
        return alert("Parameter-space exploration is available for ODE models.");
    }

    // Build bounds payload from every reaction parameter's current range
    const bounds = {};
    Object.keys(state.customParams).forEach(p => {
        const b = state.paramBounds[p] || defaultBounds(p, state.customParams[p]);
        bounds[p] = { min: Number(b.min), max: Number(b.max) };
    });

    const method = document.getElementById("explore-method").value;
    const nSamples = parseInt(document.getElementById("explore-samples").value) || 48;
    const target = document.getElementById("explore-target").value;
    const tmax = parseFloat(document.getElementById("sim-tmax").value);
    if (state.blueprint.simulation_config) state.blueprint.simulation_config.t_max = tmax;

    const btn = document.getElementById("btn-explore");
    btn.textContent = "Sampling…";
    btn.disabled = true;
    updateStatus("Exploring parameter space...", "yellow");

    try {
        const resp = await fetch("/api/sample", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                blueprint: state.blueprint,
                param_bounds: bounds,
                n_samples: nSamples,
                method: method,
                target_species: target
            })
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        state.sampleResult = data;
        exploreSort = { key: 'id', dir: 1 };
        renderEnsembleChart(data);
        renderExploreResults(data);
        updateStatus("Ready", "green");
    } catch (e) {
        console.error(e);
        alert("Exploration failed: " + e.message);
        updateStatus("Exploration failed", "red");
    } finally {
        btn.textContent = "Explore Parameter Space";
        btn.disabled = false;
    }
}

function renderEnsembleChart(data) {
    stopPdeAnimation();
    document.getElementById("ode-chart").style.display = "block";
    document.getElementById("pde-canvas").style.display = "none";
    document.getElementById("pde-controls").style.display = "none";
    document.getElementById("vis-title").textContent =
        `Parameter Ensemble: ${data.target_species} (${data.n_evaluated} samples)`;

    const ctx = document.getElementById("ode-chart").getContext("2d");
    const t = data.t;
    const trajs = data.samples.filter(s => s.trajectory && s.trajectory.length);

    const datasets = trajs.map(s => ({
        label: `#${s.id}`,
        data: s.trajectory,
        borderColor: "rgba(0, 242, 254, 0.13)",
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.15,
        fill: false
    }));

    if (trajs.length) {
        const L = t.length;
        const mean = new Array(L).fill(0);
        trajs.forEach(s => { for (let i = 0; i < L; i++) mean[i] += (s.trajectory[i] || 0); });
        for (let i = 0; i < L; i++) mean[i] /= trajs.length;
        datasets.push({
            label: "ensemble mean",
            data: mean,
            borderColor: "#00f2fe",
            borderWidth: 2.5,
            pointRadius: 0,
            tension: 0.15,
            fill: false
        });
    }

    if (chartInstance) chartInstance.destroy();
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: t.map(v => v.toFixed(1)), datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', maxTicksLimit: 10 },
                    title: { display: true, text: 'Time (minutes)', color: '#9ca3af' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' },
                    title: { display: true, text: `[${data.target_species}]`, color: '#9ca3af' }
                }
            },
            plugins: { legend: { display: false }, tooltip: { enabled: false } }
        }
    });
}

function renderExploreResults(data) {
    const wrap = document.getElementById("explore-results");
    wrap.hidden = false;
    const mr = data.metric_ranges;
    document.getElementById("explore-summary").innerHTML = `
        <span class="chip chip-strong">${data.method.toUpperCase()}</span>
        <span class="chip">${data.n_evaluated} samples</span>
        ${data.n_failed ? `<span class="chip warn">${data.n_failed} failed</span>` : ''}
        <span class="chip">peak [${mr.peak_value.min.toFixed(2)} – ${mr.peak_value.max.toFixed(2)}]</span>
        <span class="chip">t<sub>peak</sub> [${mr.peak_time.min.toFixed(1)} – ${mr.peak_time.max.toFixed(1)}]</span>
        <span class="chip">steady [${mr.final_value.min.toFixed(2)} – ${mr.final_value.max.toFixed(2)}]</span>`;
    renderExploreTable(data);
}

function shortParam(p) {
    return p.replace('act_', '').replace('inh_', '').replace('_to_', '→');
}

function renderExploreTable(data) {
    const table = document.getElementById("explore-table");
    const pnames = data.param_names;
    const metricCols = [['peak_value', 'Peak'], ['peak_time', 'Peak t'], ['final_value', 'Steady'], ['auc', 'AUC']];

    const rows = [...data.samples];
    const key = exploreSort.key, dir = exploreSort.dir;
    rows.sort((a, b) => {
        let av, bv;
        if (key === 'id') { av = a.id; bv = b.id; }
        else if (pnames.includes(key)) { av = a.params[key]; bv = b.params[key]; }
        else { av = a.metrics[key]; bv = b.metrics[key]; }
        return (av - bv) * dir;
    });

    let head = `<thead><tr><th data-sort="id">#</th>`;
    pnames.forEach(p => head += `<th data-sort="${p}" title="${p}">${shortParam(p)}</th>`);
    metricCols.forEach(([k, l]) => head += `<th data-sort="${k}">${l}</th>`);
    head += `</tr></thead>`;

    let body = "<tbody>";
    rows.slice(0, 200).forEach(s => {
        body += `<tr class="${s.ok ? '' : 'row-fail'}"><td>${s.id}</td>`;
        pnames.forEach(p => body += `<td>${(s.params[p]).toFixed(3)}</td>`);
        body += `<td>${s.metrics.peak_value.toFixed(2)}</td><td>${s.metrics.peak_time.toFixed(1)}</td>` +
                `<td>${s.metrics.final_value.toFixed(2)}</td><td>${s.metrics.auc.toFixed(1)}</td></tr>`;
    });
    body += "</tbody>";
    table.innerHTML = head + body;

    table.querySelectorAll("th").forEach(th => {
        const k = th.getAttribute("data-sort");
        if (k === exploreSort.key) th.classList.add(exploreSort.dir > 0 ? 'sort-asc' : 'sort-desc');
        th.addEventListener("click", () => {
            if (exploreSort.key === k) exploreSort.dir *= -1;
            else { exploreSort.key = k; exploreSort.dir = 1; }
            renderExploreTable(data);
        });
    });
}


// ==========================================================================
// CONNECTION LIBRARY MODAL ("book" of protein interactions)
// ==========================================================================
const connState = { db: 'omnipath', results: [], selected: new Set() };

function openConnModal() {
    const m = document.getElementById("conn-modal");
    m.hidden = false;
    const q = document.getElementById("conn-query");
    if (!q.value && state.blueprint && state.blueprint.nodes) {
        q.value = state.blueprint.nodes.map(n => n.id).join(", ");
    }
    setTimeout(() => q.focus(), 50);
}

function closeConnModal() {
    document.getElementById("conn-modal").hidden = true;
}

// Open the Connection Library pre-set to a database + query and run the search.
function openConnModalWith(db, query) {
    connState.db = db;
    document.querySelectorAll("#conn-db-tabs .seg-btn").forEach(b =>
        b.classList.toggle("active", b.getAttribute("data-db") === db));
    document.getElementById("conn-query").value = query;
    document.getElementById("conn-modal").hidden = false;
    connSearch();
}

function normalizeConn(item) {
    const type = (item.type || 'association').toLowerCase();
    return {
        source: (item.source || '').trim(),
        target: (item.target || '').trim(),
        type: (type === 'activation' || type === 'inhibition') ? type : 'association',
        score: item.score,
        references: item.references,
        pmid: item.pmid,
        mechanism: item.mechanism,
        effect: item.effect
    };
}

function edgeExists(src, tgt, type) {
    if (!state.blueprint || !state.blueprint.edges) return false;
    const s = src.toUpperCase(), t = tgt.toUpperCase();
    return state.blueprint.edges.some(e =>
        (e.source || '').toUpperCase() === s &&
        (e.target || '').toUpperCase() === t &&
        (e.type || 'activation').toLowerCase() === type);
}

async function connSearch() {
    const raw = document.getElementById("conn-query").value.trim();
    const results = document.getElementById("conn-results");
    connState.selected.clear();
    updateConnCount();

    if (!raw) {
        results.innerHTML = `<p class="placeholder-text">Enter one or more proteins to search.</p>`;
        return;
    }
    results.innerHTML = `<p class="placeholder-text">Searching ${connState.db}…</p>`;
    const proteins = raw.split(/[\s,]+/).filter(Boolean);

    try {
        let data;
        if (connState.db === "omnipath") {
            data = await (await fetch("/api/omnipath/interactions", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ proteins })
            })).json();
        } else if (connState.db === "string") {
            data = await (await fetch("/api/string/network", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ proteins })
            })).json();
        } else { // signor
            data = await (await fetch(`/api/signor/search?q=${encodeURIComponent(proteins[0] || raw)}`)).json();
        }
        connState.results = (data || []).map(normalizeConn).filter(c => c.source && c.target);
        renderConnResults();
    } catch (e) {
        results.innerHTML = `<p class="placeholder-text err">Search failed: ${e.message}</p>`;
    }
}

function renderConnResults() {
    const container = document.getElementById("conn-results");
    if (!connState.results.length) {
        container.innerHTML = `<p class="placeholder-text">No connections found. Try different proteins or another database.</p>`;
        return;
    }

    const order = { activation: 0, inhibition: 1, association: 2 };
    const sorted = connState.results
        .map((c, i) => ({ c, i }))
        .sort((a, b) => order[a.c.type] - order[b.c.type]);

    // Count summary
    const counts = { activation: 0, inhibition: 0, association: 0, inModel: 0 };
    connState.results.forEach(c => {
        counts[c.type] = (counts[c.type] || 0) + 1;
        if (edgeExists(c.source, c.target, c.type)) counts.inModel++;
    });
    const newTotal = connState.results.length - counts.inModel;
    let html = `<div class="conn-summary">
        <span class="chip chip-strong">${connState.results.length} found</span>
        <span class="chip">${newTotal} new</span>
        <span class="chip">${counts.inModel} in model</span>
        <span class="conn-legend">
            <span class="conn-badge type-activation">${counts.activation} activation</span>
            <span class="conn-badge type-inhibition">${counts.inhibition} inhibition</span>
            <span class="conn-badge type-association">${counts.association} association</span>
        </span>
    </div>`;

    sorted.forEach(({ c, i }) => {
        const inModel = edgeExists(c.source, c.target, c.type);
        const verb = c.type === 'inhibition' ? 'inhibits' : (c.type === 'activation' ? 'activates' : 'associates');
        const arrow = c.type === 'inhibition' ? '⊣' : (c.type === 'activation' ? '→' : '-');

        const meta = [];
        if (c.score != null && !isNaN(c.score)) meta.push(`score ${Number(c.score).toFixed(2)}`);
        if (c.mechanism) meta.push(c.mechanism);
        if (c.effect) meta.push(c.effect);
        if (c.pmid) meta.push(`PMID:${c.pmid}`);
        if (c.references) meta.push(`ref ${String(c.references).split(';')[0]}`);

        html += `<label class="conn-entry ${inModel ? 'in-model' : ''}" data-type="${c.type}">
            <input type="checkbox" data-idx="${i}" ${inModel ? 'checked disabled' : ''}>
            <span class="conn-badge type-${c.type}">${verb}</span>
            <span class="conn-pair"><b>${c.source}</b> <span class="arrow">${arrow}</span> <b>${c.target}</b></span>
            <span class="conn-meta">${meta.join(' · ')}</span>
            ${inModel ? '<span class="in-model-tag">✓ in model</span>' : ''}
        </label>`;
    });
    container.innerHTML = html;

    container.querySelectorAll("input[type=checkbox]:not([disabled])").forEach(cb => {
        cb.addEventListener("change", (e) => {
            const idx = parseInt(e.target.getAttribute("data-idx"));
            if (e.target.checked) connState.selected.add(idx); else connState.selected.delete(idx);
            updateConnCount();
        });
    });
    applyConnFilters();
}

function applyConnFilters() {
    const active = {};
    document.querySelectorAll("#conn-filters input").forEach(inp => {
        active[inp.getAttribute("data-type")] = inp.checked;
    });
    document.querySelectorAll(".conn-entry").forEach(el => {
        el.style.display = active[el.getAttribute("data-type")] ? "" : "none";
    });
}

function updateConnCount() {
    const n = connState.selected.size;
    document.getElementById("conn-selected-count").textContent = `${n} selected`;
    document.getElementById("conn-add-btn").disabled = n === 0;
}

function connSelectAllNew() {
    document.querySelectorAll("#conn-results input[type=checkbox]:not([disabled])").forEach(cb => {
        const entry = cb.closest(".conn-entry");
        if (entry && entry.style.display === "none") return; // respect active filters
        cb.checked = true;
        connState.selected.add(parseInt(cb.getAttribute("data-idx")));
    });
    updateConnCount();
}

function connClear() {
    document.querySelectorAll("#conn-results input[type=checkbox]:not([disabled])").forEach(cb => cb.checked = false);
    connState.selected.clear();
    updateConnCount();
}

function addSelectedConnections() {
    if (!state.blueprint) { alert("Compile a model first."); return; }
    if (!state.blueprint.edges) state.blueprint.edges = [];
    if (!state.blueprint.nodes) state.blueprint.nodes = [];

    // Map of existing node ids (case-insensitive) → canonical id
    const nodeIndex = {};
    state.blueprint.nodes.forEach(n => nodeIndex[n.id.toUpperCase()] = n.id);

    let added = 0, skippedDup = 0, skippedNode = 0;

    const ensureNode = (name) => {
        const key = name.toUpperCase();
        if (nodeIndex[key]) return nodeIndex[key];
        // Slight pop-up: let the researcher decline biologically implausible species
        const ok = confirm(
            `"${name}" is not in your model yet.\n\n` +
            `Add it as a new species (initial value 0)?\n\n` +
            `Choose Cancel to skip any connection that uses it.`
        );
        if (!ok) return null;
        state.blueprint.nodes.push({ id: name, name: name, initial_value: 0.0 });
        nodeIndex[key] = name;
        return name;
    };

    Array.from(connState.selected).sort((a, b) => a - b).forEach(idx => {
        const c = connState.results[idx];
        if (!c) return;
        const s = ensureNode(c.source);
        if (s === null) { skippedNode++; return; }
        const t = ensureNode(c.target);
        if (t === null) { skippedNode++; return; }
        if (edgeExists(s, t, c.type)) { skippedDup++; return; }
        // Append only; existing edges are never modified
        state.blueprint.edges.push({
            source: s, target: t, type: c.type,
            parameters: { k: 0.5, K_d: 1.0, n: 2.0 }
        });
        added++;
    });

    // Refresh views; existing connections remain intact
    document.getElementById("blueprint-json-viewer").textContent = JSON.stringify(state.blueprint, null, 2);
    renderCytoscape();
    renderTargets();
    compileBlueprint();
    closeConnModal();
    updateStatus("Ready", "green");

    let msg = `Added ${added} connection${added !== 1 ? 's' : ''}.`;
    if (skippedDup) msg += ` ${skippedDup} already existed (unchanged).`;
    if (skippedNode) msg += ` ${skippedNode} skipped (species declined).`;
    showToast(msg, added > 0 ? 'success' : 'info');
}


// ==========================================================================
// TOAST NOTIFICATIONS (non-blocking)
// ==========================================================================
function showToast(message, type = 'info', ms = 4000) {
    let c = document.getElementById('toast-container');
    if (!c) {
        c = document.createElement('div');
        c.id = 'toast-container';
        document.body.appendChild(c);
    }
    const t = document.createElement('div');
    t.className = `toast toast-${type}`;
    t.textContent = message;
    c.appendChild(t);
    requestAnimationFrame(() => t.classList.add('show'));
    setTimeout(() => {
        t.classList.remove('show');
        setTimeout(() => t.remove(), 300);
    }, ms);
}


// ==========================================================================
// AI LANGUAGE MODEL SETTINGS
// ==========================================================================
let llmModelsInfo = null;
let llmDraft = null;                 // uncommitted edits while the modal is open
const activePolls = {};              // model key -> setInterval id (dedup + cleanup)

function loadLlmSettings() {
    let hadSaved = false;
    try {
        const raw = localStorage.getItem('biosim_llm');
        const saved = JSON.parse(raw || 'null');
        if (saved && typeof saved === 'object') { Object.assign(state.llm, saved); hadSaved = true; }
    } catch (e) { /* ignore */ }
    updateEngineLabel();
    // If the server has a pre-configured .env for Bedrock, adopt its model/region
    // and auto-select the Bedrock engine so no keys ever need typing in the app.
    fetch('/api/llm/env').then(r => r.ok ? r.json() : null).then(env => {
        if (!env) return;
        if (env.model) state.llm.bedrock_model = env.model;
        if (env.region) state.llm.bedrock_region = env.region;
        // Only adopt Bedrock when the environment actually has usable credentials
        // (placeholder .env values are reported as not-ready by the server).
        const wantBedrock = env.bedrock_env_ready &&
            (env.engine_default === 'bedrock' || !hadSaved);
        if (wantBedrock && state.llm.engine !== 'bedrock') {
            state.llm.engine = 'bedrock';
            saveLlmSettings();
        }
        updateEngineLabel();
    }).catch(() => { /* env endpoint optional */ });
}

function saveLlmSettings() {
    try {
        // Persist everything EXCEPT credentials (remote bearer token and ALL AWS key
        // material — access key id, secret key, session token), which are kept in memory
        // only so no credential ever touches disk.
        const toSave = Object.assign({}, state.llm);
        delete toSave.remote_key;
        delete toSave.bedrock_bearer_token;
        delete toSave.bedrock_access_key;
        delete toSave.bedrock_secret_key;
        delete toSave.bedrock_session_token;
        localStorage.setItem('biosim_llm', JSON.stringify(toSave));
    } catch (e) { /* ignore */ }
}

function engineDisplayName(cfg) {
    const l = cfg || state.llm || {};
    if (l.engine === 'local') {
        const m = llmModelsInfo && llmModelsInfo.models.find(x => x.key === l.model);
        return m ? m.label : (l.model || 'Local model');
    }
    if (l.engine === 'remote') return 'Remote: ' + (l.remote_model || 'endpoint');
    if (l.engine === 'bedrock') return 'AWS Bedrock: ' + (l.bedrock_model || 'model');
    return 'Rule-based';
}

function updateEngineLabel() {
    const el = document.getElementById('ai-engine-label');
    if (el) el.textContent = engineDisplayName(state.llm);
}

function openLlmModal() {
    // Edit a DRAFT copy; the live config only changes on Save.
    llmDraft = JSON.parse(JSON.stringify(state.llm));
    document.getElementById('llm-modal').hidden = false;
    selectEngineTab(llmDraft.engine || 'off');
    document.getElementById('llm-remote-url').value = llmDraft.remote_url || '';
    document.getElementById('llm-remote-model').value = llmDraft.remote_model || '';
    document.getElementById('llm-remote-key').value = llmDraft.remote_key || '';
    const bm = document.getElementById('llm-bedrock-model');
    const br = document.getElementById('llm-bedrock-region');
    if (bm) bm.value = llmDraft.bedrock_model || 'mistral.mistral-large-3-675b-instruct';
    if (br) br.value = llmDraft.bedrock_region || 'us-east-2';
    const bbt = document.getElementById('llm-bedrock-bearer-token');
    const bak = document.getElementById('llm-bedrock-access-key');
    const bsk = document.getElementById('llm-bedrock-secret-key');
    const bst = document.getElementById('llm-bedrock-session-token');
    if (bbt) bbt.value = llmDraft.bedrock_bearer_token || '';
    if (bak) bak.value = llmDraft.bedrock_access_key || '';
    if (bsk) bsk.value = llmDraft.bedrock_secret_key || '';
    if (bst) bst.value = llmDraft.bedrock_session_token || '';
    loadLlmModels();
    updateLlmNote();
}

function closeLlmModal() {
    document.getElementById('llm-modal').hidden = true;
    llmDraft = null;                 // discard uncommitted edits
    stopAllPolls();                  // stop background polling on dismissal
}

function stopAllPolls() {
    Object.keys(activePolls).forEach(k => { clearInterval(activePolls[k]); delete activePolls[k]; });
}

function selectEngineTab(engine) {
    document.querySelectorAll('#llm-engine-tabs .seg-btn').forEach(b =>
        b.classList.toggle('active', b.getAttribute('data-engine') === engine));
    document.getElementById('llm-panel-off').hidden = engine !== 'off';
    document.getElementById('llm-panel-local').hidden = engine !== 'local';
    document.getElementById('llm-panel-remote').hidden = engine !== 'remote';
    const bp = document.getElementById('llm-panel-bedrock');
    if (bp) bp.hidden = engine !== 'bedrock';
    if (llmDraft) llmDraft.engine = engine;
    updateLlmNote();
}

function updateLlmNote() {
    const note = document.getElementById('llm-current-note');
    if (note) note.textContent = 'Engine: ' + engineDisplayName(llmDraft || state.llm);
}

async function loadLlmModels() {
    const list = document.getElementById('llm-model-list');
    list.innerHTML = `<p class="placeholder-text">Loading models…</p>`;
    try {
        const info = await (await fetch('/api/llm/models')).json();
        llmModelsInfo = info;
        document.getElementById('llm-runtime-warning').hidden = !!info.runtime_available;
        renderLlmModels();
    } catch (e) {
        list.innerHTML = `<p class="placeholder-text err">Failed to load models: ${e.message}</p>`;
    }
}

function renderLlmModels() {
    const list = document.getElementById('llm-model-list');
    const info = llmModelsInfo;
    if (!info || !info.models.length) { list.innerHTML = `<p class="placeholder-text">No models available.</p>`; return; }
    const ready = !!info.runtime_available;

    list.innerHTML = info.models.map(m => {
        const tags = `${m.recommended ? '<span class="tag tag-accent">recommended</span>' : ''}${m.heavy ? '<span class="tag">heavy</span>' : ''}`;
        // Only offer Download when the runtime is actually installed.
        let action;
        if (m.downloaded) action = '<span class="llm-ready">✓ Ready</span>';
        else if (ready) action = `<button type="button" class="btn btn-outline btn-sm llm-dl-btn" data-key="${m.key}">Download</button>`;
        else action = '<span class="llm-model-meta">unavailable</span>';
        return `<label class="llm-model-row ${m.downloaded ? '' : 'not-downloaded'}" data-key="${m.key}">
            <input type="radio" name="llm-model" value="${m.key}" ${(llmDraft && llmDraft.model === m.key) ? 'checked' : ''} ${ready ? '' : 'disabled'}>
            <span class="llm-model-main">
                <span class="llm-model-name">${m.label} ${tags}</span>
                <span class="llm-model-meta">${m.params} · ~${m.size_gb} GB</span>
            </span>
            <span class="llm-model-action" data-key="${m.key}">${action}</span>
        </label>`;
    }).join('');

    list.querySelectorAll('.llm-dl-btn').forEach(b =>
        b.addEventListener('click', (e) => { e.preventDefault(); startModelDownload(b.getAttribute('data-key')); }));
    list.querySelectorAll('input[name="llm-model"]').forEach(r =>
        r.addEventListener('change', () => {
            if (llmDraft) llmDraft.model = r.value;
            updateLlmNote();
            const m = info.models.find(x => x.key === r.value);
            if (m && !m.downloaded) startModelDownload(r.value);
        }));
}

async function startModelDownload(key) {
    const action = document.querySelector(`.llm-model-action[data-key="${key}"]`);
    const spec = llmModelsInfo && llmModelsInfo.models.find(x => x.key === key);
    if (action) action.innerHTML = `<span class="llm-progress">Downloading${spec ? ` ~${spec.size_gb} GB` : ''}…</span>`;
    try {
        const resp = await fetch('/api/llm/download', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: key })
        });
        if (!resp.ok) {
            // Surface the server's reason (e.g. runtime missing, disk full, another download running)
            let detail = `HTTP ${resp.status}`;
            try { const j = await resp.json(); if (j.detail) detail = j.detail; } catch (e) { /* ignore */ }
            resetDownloadAction(key, 'Retry');
            showToast(`Cannot download: ${detail}`, 'error', 7000);
            return;
        }
        showToast(`Downloading ${spec ? spec.label : key}${spec ? ` (~${spec.size_gb} GB)` : ''}. This can take a few minutes.`, 'info', 5000);
        pollDownload(key);
    } catch (e) {
        resetDownloadAction(key, 'Retry');
        showToast(`Download request failed: ${e.message}`, 'error', 6000);
    }
}

function resetDownloadAction(key, label) {
    const action = document.querySelector(`.llm-model-action[data-key="${key}"]`);
    if (!action) return;
    action.innerHTML = `<button type="button" class="btn btn-outline btn-sm llm-dl-btn" data-key="${key}">${label}</button>`;
    const b = action.querySelector('.llm-dl-btn');
    if (b) b.addEventListener('click', (e) => { e.preventDefault(); startModelDownload(key); });
}

function pollDownload(key) {
    if (activePolls[key]) return;    // dedup: one poller per model
    activePolls[key] = setInterval(async () => {
        let st;
        try { st = await (await fetch(`/api/llm/status?model=${encodeURIComponent(key)}`)).json(); }
        catch (e) { return; /* transient; keep polling */ }

        const action = document.querySelector(`.llm-model-action[data-key="${key}"]`);
        if (st.downloaded || st.status === 'done') {
            clearInterval(activePolls[key]); delete activePolls[key];
            if (action) action.innerHTML = '<span class="llm-ready">✓ Ready</span>';
            const row = document.querySelector(`.llm-model-row[data-key="${key}"]`);
            if (row) row.classList.remove('not-downloaded');
            if (llmModelsInfo) { const m = llmModelsInfo.models.find(x => x.key === key); if (m) m.downloaded = true; }
            showToast(`Model ready: ${key}`, 'success');
        } else if (st.status === 'error') {
            clearInterval(activePolls[key]); delete activePolls[key];
            resetDownloadAction(key, 'Retry');
            showToast(`Download failed: ${st.error || 'unknown error'}`, 'error', 7000);
        }
    }, 2500);
}

function saveLlmFromModal() {
    if (!llmDraft) { closeLlmModal(); return; }
    if (llmDraft.engine === 'remote') {
        llmDraft.remote_url = document.getElementById('llm-remote-url').value.trim();
        llmDraft.remote_model = document.getElementById('llm-remote-model').value.trim();
        llmDraft.remote_key = document.getElementById('llm-remote-key').value.trim();
        if (!llmDraft.remote_url || !llmDraft.remote_model) {
            showToast('Enter both an endpoint URL and a model name for a remote engine.', 'warn');
            return;
        }
    }
    if (llmDraft.engine === 'bedrock') {
        llmDraft.bedrock_model = document.getElementById('llm-bedrock-model').value.trim();
        llmDraft.bedrock_region = document.getElementById('llm-bedrock-region').value.trim();
        llmDraft.bedrock_bearer_token = document.getElementById('llm-bedrock-bearer-token').value.trim();
        llmDraft.bedrock_access_key = document.getElementById('llm-bedrock-access-key').value.trim();
        llmDraft.bedrock_secret_key = document.getElementById('llm-bedrock-secret-key').value.trim();
        llmDraft.bedrock_session_token = document.getElementById('llm-bedrock-session-token').value.trim();
        if (!llmDraft.bedrock_model || !llmDraft.bedrock_region) {
            showToast('Enter both a Bedrock model ID and an AWS region.', 'warn');
            return;
        }
        if ((llmDraft.bedrock_access_key && !llmDraft.bedrock_secret_key) ||
            (!llmDraft.bedrock_access_key && llmDraft.bedrock_secret_key)) {
            showToast('Enter both an Access Key ID and a Secret Access Key, or leave both blank to use this machine’s AWS credentials.', 'warn', 6000);
            return;
        }
    }
    if (llmDraft.engine === 'local') {
        const m = llmModelsInfo && llmModelsInfo.models.find(x => x.key === llmDraft.model);
        if (m && !m.downloaded) {
            showToast(`${m.label} isn't downloaded yet; it will fall back to rule-based until the download finishes.`, 'warn', 6000);
        }
    }
    // Commit the draft to the live config.
    state.llm = JSON.parse(JSON.stringify(llmDraft));
    saveLlmSettings();
    updateEngineLabel();
    const committed = engineDisplayName(state.llm);
    closeLlmModal();
    showToast(`AI engine set to: ${committed}`, 'success');
}


// ==========================================================================
// ENHANCEMENT EVENT BINDINGS
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
    // --- Parameter bounds visibility toggle
    const boundsBtn = document.getElementById("btn-toggle-bounds");
    if (boundsBtn) {
        boundsBtn.addEventListener("click", () => {
            state.showBounds = !state.showBounds;
            document.getElementById("parameters-sliders-container")
                .classList.toggle("show-bounds", state.showBounds);
            boundsBtn.textContent = state.showBounds ? "Hide bounds" : "Edit bounds";
            boundsBtn.classList.toggle("active", state.showBounds);
        });
    }

    // --- Boundary conditions collapse
    const bcBtn = document.getElementById("btn-toggle-bc");
    if (bcBtn) {
        bcBtn.addEventListener("click", () => {
            const body = document.getElementById("bc-content");
            const open = bcBtn.getAttribute("aria-expanded") === "true";
            bcBtn.setAttribute("aria-expanded", String(!open));
            body.hidden = open;
        });
    }

    // --- AI language model settings
    loadLlmSettings();
    const aiBtn = document.getElementById("btn-ai-settings");
    if (aiBtn) aiBtn.addEventListener("click", openLlmModal);
    const llmClose = document.getElementById("llm-modal-close");
    if (llmClose) llmClose.addEventListener("click", closeLlmModal);
    const llmModal = document.getElementById("llm-modal");
    if (llmModal) llmModal.addEventListener("click", (e) => { if (e.target.id === "llm-modal") closeLlmModal(); });
    document.querySelectorAll("#llm-engine-tabs .seg-btn").forEach(b =>
        b.addEventListener("click", () => selectEngineTab(b.getAttribute("data-engine"))));
    const llmSave = document.getElementById("llm-save-btn");
    if (llmSave) llmSave.addEventListener("click", saveLlmFromModal);
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !document.getElementById("llm-modal").hidden) closeLlmModal();
    });

    // --- Equations: Symbols / Full-names toggle
    document.querySelectorAll("#eq-mode-toggle .seg-btn").forEach(b => {
        b.addEventListener("click", () => {
            document.querySelectorAll("#eq-mode-toggle .seg-btn").forEach(x => x.classList.remove("active"));
            b.classList.add("active");
            state.eqMode = b.getAttribute("data-mode");
            renderEquations();
        });
    });

    // --- Parameter-space exploration
    const exploreBtn = document.getElementById("btn-explore");
    if (exploreBtn) exploreBtn.addEventListener("click", runExploration);

    // --- Connection Library modal
    const openBtn = document.getElementById("btn-open-conn-library");
    if (openBtn) openBtn.addEventListener("click", openConnModal);
    const closeBtn = document.getElementById("conn-modal-close");
    if (closeBtn) closeBtn.addEventListener("click", closeConnModal);

    const modal = document.getElementById("conn-modal");
    if (modal) modal.addEventListener("click", (e) => { if (e.target.id === "conn-modal") closeConnModal(); });

    const searchBtn = document.getElementById("conn-search-btn");
    if (searchBtn) searchBtn.addEventListener("click", connSearch);
    const connQuery = document.getElementById("conn-query");
    if (connQuery) connQuery.addEventListener("keydown", (e) => { if (e.key === "Enter") connSearch(); });

    document.querySelectorAll("#conn-db-tabs .seg-btn").forEach(b => {
        b.addEventListener("click", () => {
            document.querySelectorAll("#conn-db-tabs .seg-btn").forEach(x => x.classList.remove("active"));
            b.classList.add("active");
            connState.db = b.getAttribute("data-db");
        });
    });
    document.querySelectorAll("#conn-filters input").forEach(inp =>
        inp.addEventListener("change", applyConnFilters));

    const selAll = document.getElementById("conn-select-all");
    if (selAll) selAll.addEventListener("click", connSelectAllNew);
    const clr = document.getElementById("conn-clear");
    if (clr) clr.addEventListener("click", connClear);
    const addBtn = document.getElementById("conn-add-btn");
    if (addBtn) addBtn.addEventListener("click", addSelectedConnections);

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !document.getElementById("conn-modal").hidden) closeConnModal();
    });
});

