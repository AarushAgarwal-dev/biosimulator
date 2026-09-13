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
        ],
        // Deterministic blueprint loaded directly (no LLM), so ERK/EGFR are the exact
        // species the targets reference.
        //
        // These rate laws are EXPLICIT because the generic Hill compiler could not
        // produce the behaviour the preset is named for. With nodes and edges alone,
        // ERK rose monotonically to 4.71 and its maximum WAS the final sample
        // (peak_time == t_max == 50), so the preset that auto-loads on page open failed
        // 2 of its own 3 targets and the target ring read 33% on first sight.
        //
        // Two ingredients were missing, and neither is a tuning knob:
        //   1. Every stage needs a dephosphorylation term. Without one a species can
        //      only accumulate, so no amount of upstream feedback makes it come down.
        //   2. The stimulus has to be transient. `ERK inhibits EGFR` was present, but
        //      EGF was held at 10.0 for ever, so the receptor was continuously
        //      re-driven and the cascade settled at a plateau instead of adapting.
        //      EGF is now consumed on binding (-kEGF*EGF*EGFR).
        // Species are fractional activations in [0, 1], which is what bounds the peak.
        //
        // Measured with these values: ERK peaks at t = 6.57 with value 0.607 and falls
        // to 46% of peak by t = 50; EGFR decays to 0.7% of its peak. An earlier
        // parameter set met all three shipped targets with only a 4% ERK overshoot
        // followed by a plateau -- rejected, because the targets as written do not
        // require ERK to come back down but adaptation does.
        blueprint: {
            type: "ODE",
            nodes: [
                { id: "EGF", initial_value: 10.0 }, { id: "EGFR", initial_value: 1.0 },
                { id: "RAS", initial_value: 1.0 }, { id: "RAF", initial_value: 1.0 },
                { id: "MEK", initial_value: 0.0 }, { id: "ERK", initial_value: 0.0 }
            ],
            edges: [
                { source: "EGF", target: "EGFR", type: "activation" },
                { source: "EGFR", target: "RAS", type: "activation" },
                { source: "RAS", target: "RAF", type: "activation" },
                { source: "RAF", target: "MEK", type: "activation" },
                { source: "MEK", target: "ERK", type: "activation" },
                { source: "ERK", target: "EGFR", type: "inhibition" }
            ],
            parameters: {
                kEGF: 0.8,                       // ligand consumed on binding
                kR: 1.8, KmR: 5.0,               // receptor activation by ligand
                KiR: 0.06, nR: 3.0,              // ERK negative feedback on the receptor
                dR: 0.25,                        // receptor turnover
                k1: 0.9, k2: 0.9, k3: 0.9, k4: 0.9,
                d1: 0.35, d2: 0.35, d3: 0.35, d4: 0.35   // dephosphorylation per stage
            },
            odes: {
                EGF: "-kEGF*EGF*EGFR",
                EGFR: "kR*EGF/(KmR + EGF)*(1 - EGFR)/(1 + (ERK/KiR)**nR) - dR*EGFR",
                RAS: "k1*EGFR*(1 - RAS) - d1*RAS",
                RAF: "k2*RAS*(1 - RAF) - d2*RAF",
                MEK: "k3*RAF*(1 - MEK) - d3*MEK",
                ERK: "k4*MEK*(1 - ERK) - d4*ERK"
            },
            simulation_config: { t_max: 50.0 }
        }
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
        // Goodwin (1965) three-stage negative-feedback loop: GENA -> GENB -> GENC -| GENA.
        // The three stages ARE the delay that a negative-feedback oscillator needs.
        // Griffith (1968) proved this loop has a limit cycle only for Hill n > 8 when
        // removal is first-order, so n is a structural requirement here, not a fitted
        // knob: n = 16 sits well past the Hopf bifurcation (measured at n ~ 10), giving
        // an amplitude constant to 1 part in 10^4 over 115 cycles. Initial values are a
        // point ON the limit cycle, so the very first cycle already looks like the rest.
        // Shipped tuned, so ONE "Run Simulation" oscillates with no refine round.
        text: `GENA activates GENB.
GENB activates GENC.
GENC inhibits GENA.
GENA starts at 2.851.
GENB starts at 1.495.
GENC starts at 1.024.`,
        targets: [
            { species: "GENA", type: "oscillation", min: 0.3 }
        ],
        t_max: 300.0,
        blueprint: {
            type: "ODE",
            name: "Goodwin three-stage negative-feedback oscillator",
            nodes: [
                { id: "GENA", name: "Gene A product", initial_value: 2.851 },
                { id: "GENB", name: "Gene B product", initial_value: 1.495 },
                { id: "GENC", name: "Gene C product (repressor)", initial_value: 1.024 }
            ],
            edges: [
                { id: "e1", source: "GENA", target: "GENB", type: "activation" },
                { id: "e2", source: "GENB", target: "GENC", type: "activation" },
                { id: "e3", source: "GENC", target: "GENA", type: "inhibition" }
            ],
            parameters: {
                v1: 1.0, K1: 1.0, n: 16.0,
                d1: 0.15, k3: 0.15, d2: 0.15, k5: 0.15, d3: 0.15
            },
            odes: {
                // GENC represses GENA's synthesis; each downstream step adds phase lag.
                GENA: "v1*K1**n/(K1**n + GENC**n) - d1*GENA",
                GENB: "k3*GENA - d2*GENB",
                GENC: "k5*GENB - d3*GENC"
            },
            plot_species: ["GENA", "GENB", "GENC"],
            simulation_config: { t_max: 300.0 }
        }
    },
    bistable: {
        // Cooperative positive autofeedback + first-order removal = a true two-attractor
        // switch (Griffith 1968 / Ferrell's ultrasensitive-feedback switch). CaMKII
        // autophosphorylation is a 12-subunit holoenzyme process, so Hill n = 4 is a
        // conservative cooperativity. The nullclines cross THREE times, so the OFF
        // (0.040) and ON (1.896) states are genuine fixed points: the gap is identical
        // at t = 100, 1000 and 10000, not a slow transient. STIM is a real knob, not
        // decoration - the OFF state is annihilated in a saddle-node at STIM ~ 9.75,
        // so STIM = 1.0 sits inside the bistable window where the switch remembers.
        text: `STIM activates CAMKII.
CAMKII activates itself.
STIM starts at 1.0.
CAMKII starts at 0.1.`,
        targets: [
            { species: "CAMKII", type: "bistability", min: 0.5 }
        ],
        t_max: 100.0,
        blueprint: {
            type: "ODE",
            name: "Cooperative positive-feedback bistable switch",
            nodes: [
                { id: "STIM", name: "Stimulus", initial_value: 1.0 },
                { id: "CAMKII", name: "Active CaMKII", initial_value: 0.1 }
            ],
            edges: [
                { id: "e1", source: "STIM", target: "CAMKII", type: "activation" },
                { id: "e2", source: "CAMKII", target: "CAMKII", type: "activation" }
            ],
            parameters: {
                ks: 1.0, Sset: 1.0, kbas: 0.02,
                kfb: 1.0, Kfb: 1.0, n: 4.0, kdeg: 0.5
            },
            odes: {
                // Stimulus is clamped at Sset (the experimenter's input level).
                STIM: "ks*(Sset - STIM)",
                // basal drive + cooperative autophosphorylation - dephosphorylation
                CAMKII: "kbas*STIM + kfb*CAMKII**n/(Kfb**n + CAMKII**n) - kdeg*CAMKII"
            },
            plot_species: ["STIM", "CAMKII"],
            simulation_config: { t_max: 100.0 }
        }
    },
    foldchange: {
        // Incoherent feed-forward loop (Goentoro & Alon 2009): EGF drives AKT directly
        // AND drives an adapted background BG that divides it, so AKT tracks the RATIO
        // EGF/BG - the input's fold-change - and not its absolute level. BG adapts at a
        // level-INDEPENDENT rate (Weber's law), which is what makes the peak invariant.
        // A saturating Hill cascade cannot do this: its response asymptotes, so equal
        // fold-changes give ever-smaller peaks as the level rises. Same structure as the
        // Lyashenko (2020) paper model below; here the fold-change step is applied by
        // the model at t_on while EGF sets the ambient level, so a level sweep measures
        // level-invariance. BG starts high (desensitised) and re-adapts to the ambient
        // level before the step, so no start-up transient contaminates the peak.
        text: `EGF activates AKT.
EGF activates BG.
BG inhibits AKT.
EGF starts at 1.0.
BG starts at 500.0.
AKT starts at 1.0.`,
        targets: [
            { species: "AKT", type: "fold_change", input: "EGF", output: "AKT", max: 0.15 }
        ],
        t_max: 140.0,
        blueprint: {
            type: "ODE",
            name: "Fold-change detection (incoherent feed-forward loop)",
            nodes: [
                { id: "EGF", name: "Ambient EGF level", initial_value: 1.0 },
                { id: "BG", name: "Adapted EGF background", initial_value: 500.0 },
                { id: "AKT", name: "Relative AKT response", initial_value: 1.0 }
            ],
            edges: [
                { id: "e1", source: "EGF", target: "BG", type: "activation" },
                { id: "e2", source: "EGF", target: "AKT", type: "activation" },
                { id: "e3", source: "BG", target: "AKT", type: "inhibition" }
            ],
            parameters: {
                a: 0.2, kf: 8.0, fold: 3.0, sr: 6.0, t_on: 70.0
            },
            fluxes: {
                // A clean `fold`-fold step applied to whatever the ambient EGF level is.
                step: "1 + (fold-1)/(1+exp(-sr*(t-t_on)))",
                Lig: "EGF*step"
            },
            odes: {
                // EGF is the clamped ambient bath level - the knob a level sweep varies.
                EGF: "0",
                BG: "a*(Lig - BG)",
                AKT: "kf*(Lig/BG - AKT)"
            },
            plot_species: ["EGF", "BG", "AKT"],
            simulation_config: { t_max: 140.0 }
        }
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

async function loadPreset(name) {
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

    // If the preset ships a ready blueprint, load it directly (no LLM). This keeps the
    // species exactly matching the targets, so the closed-loop optimizer works reliably.
    if (preset.blueprint) {
        state.blueprint = JSON.parse(JSON.stringify(preset.blueprint));
        document.getElementById("blueprint-json-viewer").textContent = JSON.stringify(state.blueprint, null, 2);
        renderCytoscape();
        renderTargets();
        await compileBlueprint();
        document.querySelector("[data-tab='blueprint']").click();
        updateStatus("Ready", "green");
    }
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
    if (dot) dot.className = `status-dot ${type}`;
    if (textLabel) textLabel.textContent = text;
}

// Every generated/imported model must have a usable simulation horizon. LLM and
// database payloads can legally omit this optional object, so normalize it once
// instead of letting individual workflows crash on `.simulation_config.t_max`.
function ensureSimulationConfig(blueprint) {
    if (!blueprint || typeof blueprint !== "object") return null;
    if (!blueprint.simulation_config || typeof blueprint.simulation_config !== "object") {
        blueprint.simulation_config = {};
    }
    let tMax = Number(blueprint.simulation_config.t_max);
    if (!Number.isFinite(tMax) || tMax <= 0) tMax = blueprint.type === "PDE" ? 100 : 50;
    blueprint.simulation_config.t_max = tMax;
    return blueprint.simulation_config;
}

// Parse JSON exactly once and surface FastAPI's useful `detail`/`error` message.
// This also turns true network failures into a clear connection error instead of
// secondary messages such as "results.map is not a function".
async function apiJson(url, options = {}) {
    let response;
    try {
        response = await fetch(url, options);
    } catch (error) {
        throw new Error(`Could not connect to the BioSimulateAI server: ${error.message || error}`);
    }

    const raw = await response.text();
    let data = null;
    if (raw) {
        try { data = JSON.parse(raw); }
        catch {
            if (response.ok) throw new Error("The server returned an invalid JSON response.");
        }
    }
    if (!response.ok) {
        const detail = data && (data.detail || data.error);
        const fallback = raw && !raw.trim().startsWith("<") ? raw.slice(0, 500) : "";
        throw new Error(String(detail || fallback || `Request failed with HTTP ${response.status}`));
    }
    if (data === null) throw new Error("The server returned an empty response.");
    return data;
}

// External database fields are untrusted. Escape values before interpolating them
// into the few rich result templates that cannot use textContent directly.
function escapeHtml(value) {
    return String(value == null ? "" : value).replace(/[&<>'"]/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
    })[char]);
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

// Shared: take a freshly parsed/extracted blueprint and load it into the whole UI.
async function loadBlueprintIntoUI(parsedData, { switchTab = true } = {}) {
    if (!parsedData || typeof parsedData !== "object") {
        updateStatus("Invalid server response", "red");
        throw new Error("The server did not return a blueprint object.");
    }

    // Surface any LLM fallback / transcription notice without storing transport
    // metadata in the model itself.
    if (parsedData._llm_notice) {
        showToast(String(parsedData._llm_notice), 'warn', 5000);
        delete parsedData._llm_notice;
    }
    if (parsedData._transcription) {
        const box = document.getElementById("bio-input");
        if (box && !box.value.trim()) box.value = String(parsedData._transcription).replace(/```/g, "").trim();
        delete parsedData._transcription;
    }

    const validationErrors = Array.isArray(parsedData.validation_errors)
        ? parsedData.validation_errors.filter(Boolean).map(String)
        : [];
    const hasNodes = Array.isArray(parsedData.nodes) && parsedData.nodes.length > 0;
    if (!hasNodes) {
        if (validationErrors.length) showToast("Model needs more detail: " + validationErrors.join(" "), "warn", 7000);
        updateStatus("Needs more detail", "yellow");
        return false;
    }
    if (validationErrors.length) {
        // Keep these errors on the blueprint. The backend simulation guard uses
        // them to prevent integrating a model known to be invalid.
        showToast("Model needs review: " + validationErrors.join(" "), "warn", 7000);
    }

    state.blueprint = parsedData;
    ensureSimulationConfig(state.blueprint);
    document.getElementById("blueprint-json-viewer").textContent = JSON.stringify(state.blueprint, null, 2);
    renderCytoscape();
    renderTargets();
    const compiled = await compileBlueprint();
    if (switchTab) document.querySelector("[data-tab='blueprint']").click();
    if (!compiled) return false;

    if (validationErrors.length) updateStatus("Needs review", "yellow");
    else updateStatus("Ready", "green");
    return true;
}

// 1. Text Parsing & Compilation
document.getElementById("btn-parse-text").addEventListener("click", async () => {
    const text = document.getElementById("bio-input").value.trim();
    if (!text) return alert("Please type a biological process description first!");

    updateStatus("Parsing text...", "yellow");

    try {
        const data = await apiJson("/api/blueprint", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text, llm: getLlmConfig() })
        });
        return await loadBlueprintIntoUI(data);
    } catch (e) {
        console.error("Blueprint parsing failed", e);
        updateStatus("Parsing failed", "red");
        if (typeof showToast === "function") showToast("Could not build blueprint: " + e.message, "error", 8000);
        return false;
    }
});

// 1b. Photo -> equations (click the drop-zone or drag an image onto it)
async function handleEquationImageFile(file) {
    const statusEl = document.getElementById("eq-image-status");
    if (!file) return;
    if (!/^image\//.test(file.type)) { statusEl.textContent = "Please choose an image file."; return; }
    const llm = getLlmConfig();
    if (llm.engine !== 'bedrock') {
        statusEl.textContent = "Photo reading needs the AWS Bedrock engine (set it in the model config).";
        return;
    }
    if (file.size > 12 * 1024 * 1024) { statusEl.textContent = "That image is too large (max 12 MB)."; return; }
    statusEl.textContent = "📷 " + file.name + " — reading equations…";
    updateStatus("Reading equations from image…", "yellow");
    try {
        const dataUrl = await new Promise((res, rej) => {
            const fr = new FileReader();
            fr.onload = () => res(fr.result);
            fr.onerror = () => rej(new Error("Could not read the file."));
            fr.readAsDataURL(file);
        });
        const data = await apiJson("/api/extract-equations", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataUrl, llm: llm })
        });
        const loaded = await loadBlueprintIntoUI(data);
        if (!loaded) {
            statusEl.textContent = "Equations were read, but the resulting model needs correction before it can run.";
            return false;
        }
        statusEl.textContent = "✓ Read equations from " + file.name + " — review & edit in the Model Summary.";
        return true;
    } catch (e) {
        console.error(e);
        statusEl.textContent = "Could not read equations: " + e.message;
        updateStatus("Image read failed", "red");
    }
}
(function wireDropzone() {
    const dz = document.getElementById("eq-dropzone");
    const input = document.getElementById("eq-image-input");
    if (!dz || !input) return;
    dz.addEventListener("click", () => input.click());
    dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    input.addEventListener("change", (ev) => {
        const f = ev.target.files && ev.target.files[0];
        ev.target.value = "";
        handleEquationImageFile(f);
    });
    ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
    ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
    dz.addEventListener("drop", (e) => {
        const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        handleEquationImageFile(f);
    });
})();

// 1c. Input-mode tabs (Describe / Write equations / Photo)
(function wireInputModes() {
    const tabs = document.getElementById("input-mode-tabs");
    if (!tabs) return;
    tabs.querySelectorAll(".imode-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const mode = btn.getAttribute("data-imode");
            tabs.querySelectorAll(".imode-btn").forEach(b => b.classList.toggle("active", b === btn));
            document.querySelectorAll(".imode-panel").forEach(p =>
                p.classList.toggle("active", p.getAttribute("data-imode-panel") === mode));
        });
    });
})();

// 1d. Write-your-own-equations editor: live preview + build (deterministic, no LLM)
const EQ_EXAMPLE =
`dZ/dt = v0 + v1*beta - v2 + v3 + kf*Y - k*Z
dY/dt = v2 - v3 - kf*Y
v2 = VM2*Z^2/(K2^2 + Z^2)
v3 = VM3*(Y^2/(KR^2+Y^2))*(Z^4/(KA^4+Z^4))
v0=1, v1=7.3, beta=0.5, VM2=65, VM3=500
K2=1, KR=2, KA=0.9, kf=1, k=10
Z starts at 0.1
Y starts at 0.1`;

function renderEqPreview() {
    const src = document.getElementById("eq-input");
    const box = document.getElementById("eq-preview");
    if (!src || !box) return;
    const lines = src.value.split(/\n+/).map(l => l.split("#")[0].trim()).filter(Boolean);
    if (!lines.length) { box.innerHTML = '<span class="placeholder-text">Your equations render here as you type.</span>'; return; }
    box.innerHTML = "";
    lines.forEach(line => {
        // turn "dX/dt = ..." into pretty LaTeX; leave assignments/initials as text
        let latex = null;
        let m = line.match(/^d\s*([A-Za-z_]\w*)\s*\/\s*d\s*t\s*=\s*(.+)$/i);
        if (m) latex = "\\frac{d" + m[1] + "}{dt} = " + toLatexRHS(m[2]);
        else if (/^[A-Za-z_]\w*\s*=\s*.+/.test(line) && !/starts?\s+at/i.test(line)) latex = toLatexRHS(line);
        const item = document.createElement("div");
        item.className = "eq-preview-item";
        if (latex) {
            try { katex.render(latex, item, { throwOnError: false, displayMode: false }); }
            catch { item.textContent = line; }
        } else { item.textContent = line; }
        box.appendChild(item);
    });
}
// light touch: superscripts for ^ and \cdot for *, so the preview reads like math
function toLatexRHS(s) {
    return String(s)
        .replace(/\*\*/g, "^")
        .replace(/\^(\w+)/g, "^{$1}")
        .replace(/\^\(([^)]+)\)/g, "^{$1}")
        .replace(/\*/g, " \\cdot ");
}
(function wireEqEditor() {
    const src = document.getElementById("eq-input");
    const buildBtn = document.getElementById("btn-build-eq");
    const exBtn = document.getElementById("btn-eq-example");
    if (!src) return;
    let t = null;
    src.addEventListener("input", () => { clearTimeout(t); t = setTimeout(renderEqPreview, 200); });
    if (exBtn) exBtn.addEventListener("click", () => { src.value = EQ_EXAMPLE; renderEqPreview(); });
    if (buildBtn) buildBtn.addEventListener("click", async () => {
        const equations = src.value.trim();
        const statusEl = document.getElementById("eq-build-status");
        if (!equations) { statusEl.textContent = "Write at least one equation, e.g. dX/dt = k - X."; return; }
        statusEl.textContent = "Building model from your equations…";
        updateStatus("Building model…", "yellow");
        try {
            const data = await apiJson("/api/equations-to-model", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ equations })
            });
            const loaded = await loadBlueprintIntoUI(data);
            if (!loaded) {
                statusEl.textContent = "The equations were parsed, but the model could not be compiled. Review the Model Summary.";
            } else if (Array.isArray(data.validation_errors) && data.validation_errors.length) {
                statusEl.textContent = "⚠ " + data.validation_errors.join("  ");
            } else {
                statusEl.textContent = "✓ Built model with " + (Array.isArray(data.nodes) ? data.nodes.length : 0) + " variable(s). Edit it in the Model Summary.";
            }
        } catch (e) {
            console.error(e);
            statusEl.textContent = "Could not build model: " + e.message;
            updateStatus("Build failed", "red");
        }
    });
})();

async function compileBlueprint() {
    if (!state.blueprint) return false;

    try {
        const data = await apiJson("/api/compile", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ blueprint: state.blueprint })
        });
        state.equations = data.equations;
        state.equationsVerbose = data.equations_verbose || data.equations;

        // Render Equations via KaTeX
        renderEquations();

        // Load default values into parameters state
        state.customParams = {};
        if (data.parameters) Object.assign(state.customParams, data.parameters);

        // Set simulation tmax (a preset may request a longer horizon, e.g. oscillations)
        const config = ensureSimulationConfig(state.blueprint);
        const requestedTmax = Number(state.tmaxOverride || config.t_max);
        config.t_max = Number.isFinite(requestedTmax) && requestedTmax > 0
            ? requestedTmax
            : (state.blueprint.type === "PDE" ? 100 : 50);
        document.getElementById("sim-tmax").value = config.t_max;
        state.tmaxOverride = null;

        renderParameterSliders();
        renderBoundaryConditions();
        populateExploreTargets();
        renderModelSummary();
        return true;
    } catch (e) {
        console.error("Compilation error", e);
        updateStatus("Compilation failed", "red");
        if (typeof showToast === "function") showToast("Could not compile: " + e.message, "error", 7000);
        // Keep the existing blueprint available for correction, but never claim it
        // compiled successfully or erase the last valid equation display.
        try { renderModelSummary(); } catch { /* summary is best-effort */ }
        return false;
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

// ==========================================================================
// EDITABLE MODEL SUMMARY TABLE — variables, parameters, and (for custom-kinetics
// models) the d/dt equations. Lets the user fine-tune what the AI generated:
// change initial values / parameters, hand-edit an equation, or add a new
// species+equation, then Apply to recompile and simulate.
// ==========================================================================
function _msEsc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;")
                    .replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderModelSummary() {
    const host = document.getElementById("model-summary-container");
    if (!host) return;
    const bp = state.blueprint;
    if (!bp || !Array.isArray(bp.nodes) || bp.nodes.length === 0) {
        host.innerHTML = '<p class="placeholder-text">Compile a model to see and edit its variables and parameters here.</p>';
        return;
    }
    const isCustom = !!(bp.odes && Object.keys(bp.odes).length);
    const odes = bp.odes || {};
    const params = isCustom ? (bp.parameters || {}) : (state.customParams || {});

    let h = '';
    // ---- Variables ----
    h += '<div class="ms-section"><div class="ms-head">Variables (state species) · ' + bp.nodes.length + '</div>';
    h += '<div class="ms-table-wrap"><table class="ms-table"><thead><tr><th>Species</th><th>Initial value</th>' +
         (isCustom ? '<th>d/dt equation</th>' : '') + '<th></th></tr></thead><tbody>';
    bp.nodes.forEach(n => {
        const id = n.id;
        const init = (n.initial_value !== undefined ? n.initial_value : 0);
        h += '<tr data-node="' + _msEsc(id) + '">';
        h += '<td class="ms-name">' + _msEsc(id) + '</td>';
        h += '<td><input class="ms-init" type="number" step="any" value="' + _msEsc(init) + '"></td>';
        if (isCustom) h += '<td><input class="ms-ode" type="text" spellcheck="false" value="' + _msEsc(odes[id] || "0") + '"></td>';
        h += '<td class="ms-x">' + (isCustom ? '<button class="ms-del ms-del-node" title="Remove species">✕</button>' : '') + '</td>';
        h += '</tr>';
    });
    h += '</tbody></table></div>';
    if (isCustom) {
        h += '<div class="ms-add-row"><input class="ms-new-species" placeholder="new species id, e.g. Ca_mito">' +
             '<input class="ms-new-init" type="number" step="any" placeholder="init" value="0">' +
             '<button class="ms-btn ms-add-species">+ Add species &amp; equation</button></div>';
    }
    h += '</div>';

    // ---- Parameters ----
    const pnames = Object.keys(params).sort();
    h += '<div class="ms-section"><div class="ms-head">Parameters · ' + pnames.length + '</div>';
    h += '<div class="ms-table-wrap"><table class="ms-table"><thead><tr><th>Parameter</th><th>Value</th><th></th></tr></thead><tbody>';
    pnames.forEach(p => {
        h += '<tr data-param="' + _msEsc(p) + '">';
        h += '<td class="ms-name">' + _msEsc(p) + '</td>';
        h += '<td><input class="ms-pval" type="number" step="any" value="' + _msEsc(params[p]) + '"></td>';
        h += '<td class="ms-x">' + (isCustom ? '<button class="ms-del ms-del-param" title="Remove parameter">✕</button>' : '') + '</td>';
        h += '</tr>';
    });
    h += '</tbody></table></div>';
    if (isCustom) {
        h += '<div class="ms-add-row"><input class="ms-new-param-name" placeholder="name, e.g. k_leak">' +
             '<input class="ms-new-param-val" type="number" step="any" placeholder="value" value="1">' +
             '<button class="ms-btn ms-add-param">+ Add parameter</button></div>';
    }
    h += '</div>';

    // ---- Actions ----
    h += '<div class="ms-actions"><button class="ms-btn ms-apply">Apply &amp; Recompile</button>' +
         '<span class="ms-note">' + (isCustom
            ? 'Edit initial values & equations, add a species, then Apply.'
            : 'Built from the interaction graph — edit initial values & parameter values here.') +
         '</span></div>';

    host.innerHTML = h;
}

// Read the current input values from the summary table back onto the blueprint /
// customParams (without recompiling).
function readSummaryEdits() {
    const host = document.getElementById("model-summary-container");
    const bp = state.blueprint;
    if (!host || !bp) return;
    const isCustom = !!(bp.odes && Object.keys(bp.odes).length);
    host.querySelectorAll("tr[data-node]").forEach(tr => {
        const id = tr.getAttribute("data-node");
        const node = (bp.nodes || []).find(n => n.id === id);
        if (!node) return;
        const initEl = tr.querySelector(".ms-init");
        if (initEl && initEl.value !== "") node.initial_value = parseFloat(initEl.value);
        if (isCustom) {
            const odeEl = tr.querySelector(".ms-ode");
            if (odeEl) { bp.odes = bp.odes || {}; bp.odes[id] = odeEl.value; }
        }
    });
    host.querySelectorAll("tr[data-param]").forEach(tr => {
        const p = tr.getAttribute("data-param");
        const el = tr.querySelector(".ms-pval");
        if (!el || el.value === "") return;
        const v = parseFloat(el.value);
        if (isCustom) { bp.parameters = bp.parameters || {}; bp.parameters[p] = v; }
        else { state.customParams = state.customParams || {}; state.customParams[p] = v; }
    });
}

async function applySummary() {
    const bp = state.blueprint;
    if (!bp) return;
    const isCustom = !!(bp.odes && Object.keys(bp.odes).length);
    readSummaryEdits();
    document.getElementById("blueprint-json-viewer").textContent = JSON.stringify(bp, null, 2);
    if (isCustom) {
        await compileBlueprint();   // recompiles equations, sliders, and this summary
        renderCytoscape();
    } else {
        renderParameterSliders();
        renderModelSummary();
    }
    if (typeof showToast === "function") showToast("Model updated.", "info", 2000);
}

// Delegated handling for the summary table's buttons (the table is re-rendered
// often, so listen on the stable container).
(function wireModelSummary() {
    const host = document.getElementById("model-summary-container");
    if (!host) return;
    host.addEventListener("click", async (e) => {
        const bp = state.blueprint;
        if (!bp) return;
        const t = e.target;
        if (t.classList.contains("ms-apply")) { await applySummary(); return; }
        if (t.classList.contains("ms-del-node")) {
            const id = t.closest("tr").getAttribute("data-node");
            readSummaryEdits();
            bp.nodes = bp.nodes.filter(n => n.id !== id);
            if (bp.odes) delete bp.odes[id];
            await applySummary();
            return;
        }
        if (t.classList.contains("ms-del-param")) {
            const p = t.closest("tr").getAttribute("data-param");
            readSummaryEdits();
            if (bp.parameters) delete bp.parameters[p];
            renderModelSummary();
            return;
        }
        if (t.classList.contains("ms-add-species")) {
            const nameEl = host.querySelector(".ms-new-species");
            const initEl = host.querySelector(".ms-new-init");
            const id = (nameEl.value || "").trim();
            if (!id) { nameEl.focus(); return; }
            if (!/^[A-Za-z_]\w*$/.test(id)) { alert("Species id must start with a letter/underscore and contain only letters, digits, or _."); return; }
            if ((bp.nodes || []).some(n => n.id === id)) { alert("A species named '" + id + "' already exists."); return; }
            readSummaryEdits();
            bp.nodes.push({ id: id, initial_value: parseFloat(initEl.value) || 0 });
            bp.odes = bp.odes || {}; bp.odes[id] = "0";
            await applySummary();
            return;
        }
        if (t.classList.contains("ms-add-param")) {
            const nameEl = host.querySelector(".ms-new-param-name");
            const valEl = host.querySelector(".ms-new-param-val");
            const name = (nameEl.value || "").trim();
            if (!name) { nameEl.focus(); return; }
            if (!/^[A-Za-z_]\w*$/.test(name)) { alert("Parameter name must start with a letter/underscore and contain only letters, digits, or _."); return; }
            readSummaryEdits();
            bp.parameters = bp.parameters || {}; bp.parameters[name] = parseFloat(valEl.value) || 0;
            renderModelSummary();
            return;
        }
    });
})();

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
    updateStatus("Searching database...", "yellow");

    try {
        const endpoint = db === 'reactome'
            ? `/api/reactome/search?q=${encodeURIComponent(query)}`
            : `/api/biomodels/search?q=${encodeURIComponent(query)}`;
        const data = await apiJson(endpoint);
        if (!Array.isArray(data)) throw new Error("The database returned an unexpected response.");

        container.innerHTML = "";
        if (data.length === 0) {
            container.innerHTML = `<p class="placeholder-text">No ${db === 'reactome' ? 'pathways' : 'models'} found.</p>`;
            updateStatus("Ready", "green");
            return;
        }

        data.forEach(rawItem => {
            const item = rawItem && typeof rawItem === "object" ? rawItem : {};
            const id = String(item.id || "");
            const name = String(item.name || id || "Untitled result");
            const div = document.createElement("div");
            div.className = "search-result-item";

            if (db === 'reactome') {
                div.innerHTML = `
                    <strong>${escapeHtml(name)}</strong><br>
                    <span style="font-size:10px; color:var(--text-secondary);">${escapeHtml(id)} | ${escapeHtml(item.species || "")}</span>
                `;
                if (id) div.addEventListener("click", () => loadReactomePathwayText(id, name));
            } else {
                div.innerHTML = `
                    <strong>${escapeHtml(id)}</strong><br>
                    <span style="font-size:11px; color:var(--text-primary);">${escapeHtml(name)}</span><br>
                    <span style="font-size:10px; color:var(--text-secondary);">${escapeHtml(String(item.description || '').substring(0, 90))}</span>
                `;
                if (id) div.addEventListener("click", () => {
                    document.getElementById("biomodel-id-input").value = id;
                    document.querySelector("[data-tab='maple']").click();
                });
            }
            container.appendChild(div);
        });
        updateStatus("Ready", "green");
    } catch (e) {
        console.error("Database search failed", e);
        container.innerHTML = `<p class="placeholder-text err">Search failed: ${escapeHtml(e.message)}</p>`;
        updateStatus("Database search failed", "red");
    }
});

async function loadReactomePathwayText(pathwayId, pathwayName) {
    updateStatus("Fetching pathway reactions...", "yellow");

    try {
        const reactions = await apiJson(
            `/api/reactome/reactions?pathway_id=${encodeURIComponent(String(pathwayId))}`
        );
        if (!Array.isArray(reactions)) throw new Error("Reactome returned an unexpected response.");

        const lines = [`# Pathway: ${pathwayName} (${pathwayId})`];

        // Derive protein pairs from reaction display names (uppercase gene-like tokens)
        const pairs = [];
        reactions.forEach(rawReaction => {
            const rxn = rawReaction && typeof rawReaction === "object" ? rawReaction : {};
            const tokens = String(rxn.name || '')
                .replace(/[(),:;]/g, ' ')
                .split(/\s+/)
                .filter(w => w.length >= 2 && /^[A-Z][A-Z0-9-]*[A-Z0-9]$/.test(w) && isNaN(w));
            const uniq = [...new Set(tokens)];
            if (uniq.length >= 2) pairs.push([uniq[0], uniq[1]]);
        });

        if (pairs.length > 0) {
            pairs.forEach(([a, b]) => lines.push(`${a} activates ${b}.`));
            lines.push(`${pairs[0][0]} starts at 10.0.`);
            document.getElementById("bio-input").value = lines.join('\n');
            showToast(`Loaded ${pairs.length} interaction${pairs.length !== 1 ? 's' : ''} from “${pathwayName}”. Review, then Compile.`, 'success');
        } else if (reactions.length > 0) {
            reactions.slice(0, 20).forEach(rxn => lines.push(`# ${String((rxn && rxn.name) || "Unnamed reaction")}`));
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
        return true;
    } catch (e) {
        console.error("Reactome pathway load failed", e);
        updateStatus("Pathway load failed", "red");
        showToast("Failed to load pathway details: " + e.message, 'error', 7000);
        return false;
    }
}

// 3. Simulation Solver
document.getElementById("btn-run-simulation").addEventListener("click", runSimulation);

async function runSimulation() {
    if (!state.blueprint) return alert("Please compile a blueprint first!");

    updateStatus("Simulating model...", "yellow");

    try {
        const config = ensureSimulationConfig(state.blueprint);
        const enteredTmax = Number(document.getElementById("sim-tmax").value);
        if (!Number.isFinite(enteredTmax) || enteredTmax <= 0) {
            throw new Error("Simulation time must be a positive number.");
        }
        config.t_max = enteredTmax;

        state.simulationResults = await apiJson("/api/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                blueprint: state.blueprint,
                custom_params: state.customParams
            })
        });

        renderVisualizer();
        await evaluateSimulationTargets();
        updateStatus("Ready", "green");
        return true;
    } catch (e) {
        console.error("Simulation failed", e);
        updateStatus("Simulation failed", "red");
        if (typeof showToast === "function") showToast("Simulation failed: " + e.message, "error", 8000);
        return false;
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

// Custom-kinetics ODEs and PDE reaction systems carry their wiring inside
// expressions rather than an edge list. Derive display-only edges from either
// blueprint.odes or blueprint.spatial.reactions so valid PDEs never render as
// disconnected nodes. This does not alter the equations used by the solvers.
function deriveEdgesFromOdes(bp) {
    const nodes = (bp.nodes || []).map(n => n.id).filter(Boolean);
    const odeExpressions = bp.odes || {};
    const pdeExpressions = (bp.spatial && bp.spatial.reactions) || {};
    const expressions = Object.keys(odeExpressions).length ? odeExpressions : pdeExpressions;
    const fluxes = bp.fluxes || {};
    if (!nodes.length || !Object.keys(expressions).length) return [];
    const esc = (str) => String(str).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const bySpeciesLen = [...nodes].sort((a, b) => b.length - a.length);
    const fluxNames = Object.keys(fluxes).sort((a, b) => b.length - a.length);

    function expandFluxes(expr) {
        let e = String(expr);
        for (let iter = 0; iter < 4; iter++) {
            let changed = false;
            for (const f of fluxNames) {
                const re = new RegExp("\\b" + esc(f) + "\\b", "g");
                if (re.test(e)) { e = e.replace(re, "(" + fluxes[f] + ")"); changed = true; }
            }
            if (!changed) break;
        }
        return e;
    }

    // Split into top-level additive terms, each tagged with its sign.
    function signedTerms(expr) {
        const s = String(expr), terms = [];
        let depth = 0, cur = "", sign = "+";
        for (let i = 0; i < s.length; i++) {
            const c = s[i];
            if (c === "(") { depth++; cur += c; }
            else if (c === ")") { depth--; cur += c; }
            else if ((c === "+" || c === "-") && depth === 0) {
                const prev = cur.replace(/\s+$/, "").slice(-1);
                if (cur.trim() === "" || "*/(^+-eE".includes(prev)) { cur += c; }
                else { terms.push({ sign, term: cur }); sign = c; cur = ""; }
            } else cur += c;
        }
        if (cur.trim() !== "") terms.push({ sign, term: cur });
        return terms;
    }

    const speciesIn = (term) => bySpeciesLen.filter(sp =>
        new RegExp("\\b" + esc(sp) + "\\b").test(term));

    // A species in the immediate denominator has the opposite qualitative effect:
    // e.g. U**2 / V means V inhibits U even though the whole term is positive.
    function isImmediateDenominator(term, species) {
        const text = String(term);
        const speciesRe = new RegExp("\\b" + esc(species) + "\\b");
        for (let i = 0; i < text.length; i++) {
            if (text[i] !== "/") continue;
            let j = i + 1;
            while (/\s/.test(text[j] || "")) j++;
            let segment = "";
            if (text[j] === "(") {
                let depth = 0;
                for (; j < text.length; j++) {
                    segment += text[j];
                    if (text[j] === "(") depth++;
                    if (text[j] === ")" && --depth === 0) break;
                }
            } else {
                while (j < text.length && /[A-Za-z0-9_.]/.test(text[j])) segment += text[j++];
            }
            if (speciesRe.test(segment)) return true;
        }
        return false;
    }

    const edges = [], seen = new Set();
    for (const target of nodes) {
        if (!(target in expressions)) continue;
        const pos = new Set(), neg = new Set();
        for (const { sign, term } of signedTerms(expandFluxes(expressions[target]))) {
            for (const regulator of speciesIn(term)) {
                // Ignore ordinary self-decay, but retain explicit nonlinear
                // autocatalysis such as U**2 or U*U as a self-activation loop.
                if (regulator === target) {
                    const id = esc(target);
                    const nonlinearSelf = new RegExp("\\b" + id + "\\b\\s*(?:\\*\\*|\\^)").test(term) ||
                        new RegExp("\\b" + id + "\\b\\s*\\*\\s*\\b" + id + "\\b").test(term);
                    if (!nonlinearSelf || sign === "-") continue;
                }
                const denominator = isImmediateDenominator(term, regulator);
                const inhibitory = (sign === "-") !== denominator;
                (inhibitory ? neg : pos).add(regulator);
            }
        }
        for (const regulator of new Set([...pos, ...neg])) {
            const type = (neg.has(regulator) && !pos.has(regulator)) ? "inhibition" : "activation";
            const key = regulator + "->" + target;
            if (seen.has(key)) continue;
            seen.add(key);
            edges.push({ source: regulator, target, type });
        }
    }
    return edges;
}

function renderCytoscape() {
    const container = document.getElementById("cy-container");
    if (cyInstance && typeof cyInstance.destroy === "function") {
        cyInstance.destroy();
        cyInstance = null;
    }
    if (!state.blueprint || !Array.isArray(state.blueprint.nodes)) return;

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

    // Explicit edges take precedence. Equation-driven ODE and PDE models carry
    // their wiring inside expressions, so derive display-only edges when needed.
    let edgesToDraw = (state.blueprint.edges || []).filter(
        e => nodeIds.has(e.source) && nodeIds.has(e.target));
    const hasImplicitEdges = state.blueprint.odes ||
        (state.blueprint.spatial && state.blueprint.spatial.reactions);
    if (edgesToDraw.length === 0 && hasImplicitEdges) {
        edgesToDraw = deriveEdgesFromOdes(state.blueprint).filter(
            e => nodeIds.has(e.source) && nodeIds.has(e.target));
    }
    // Defensive: only draw edges whose endpoints are declared nodes. Cytoscape
    // throws on a dangling edge, which would crash the whole compile.
    edgesToDraw.forEach((edge, index) => {
        const rawType = String(edge.type || "association").toLowerCase();
        const type = (rawType === "activation" || rawType === "inhibition")
            ? rawType
            : "association";
        elements.push({
            data: {
                id: `e${index}`,
                source: edge.source,
                target: edge.target,
                type
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
                    // Emphasis is carried by the border alone. Cytoscape draws to
                    // canvas with its own style vocabulary, and this build accepts
                    // neither CSS 'box-shadow' nor 'outline-*': both were parsed,
                    // warned about on every single render, and discarded. Border
                    // width/colour are supported here, so the cyan ring is the glow.
                    'border-width': '3px',
                    'border-color': '#00f2fe',
                    'label': 'data(label)',
                    'color': '#f3f4f6',
                    'font-family': 'Outfit',
                    'font-size': '11px',
                    'text-wrap': 'wrap',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'width': '65px',
                    'height': '65px'
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
            },
            {
                selector: 'edge[type="association"]',
                style: {
                    'width': 2,
                    'line-color': '#94a3b8',
                    'target-arrow-color': '#94a3b8',
                    'target-arrow-shape': 'none',
                    'line-style': 'dashed',
                    'curve-style': 'bezier'
                }
            }
        ],
        layout: {
            name: state.blueprint.nodes.length <= 2 ? 'circle' : 'cose',
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
        const data = await apiJson("/api/evaluate", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                blueprint: state.blueprint,
                targets: state.targets,
                custom_params: state.customParams
            })
        });
        if (!data || !Array.isArray(data.results)) {
            throw new Error("The target evaluator returned an unexpected response.");
        }
        results = data.results;
    } catch (e) {
        console.error("Target evaluation failed", e);
        container.innerHTML = `<p class="placeholder-text err">Target evaluation failed: ${escapeHtml(e.message)}</p>`;
        return 0;
    }

    container.innerHTML = "";
    let metCount = 0;
    results.forEach(rawResult => {
        const r = rawResult && typeof rawResult === "object" ? rawResult : {};
        if (r.met) metCount++;
        const item = document.createElement("div");
        item.className = `eval-item ${r.met ? 'met' : 'failed'}`;
        item.innerHTML = `
            <div class="eval-icon">${r.met ? '🟢' : '🔴'}</div>
            <div class="eval-details">
                <div class="target-title">${escapeHtml(r.species || '')} ${escapeHtml(String(r.type || '').replace(/_/g, ' '))}</div>
                <div class="target-status-msg">${escapeHtml(r.detail || '')}</div>
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
    const stagnationLimit = 2;     // stop quickly if two rounds bring no improvement
    const total = state.targets.length;
    const config = ensureSimulationConfig(state.blueprint);
    const tmax = Number(document.getElementById("sim-tmax").value);
    if (!Number.isFinite(tmax) || tmax <= 0) {
        updateStatus("Optimization failed", "red");
        showToast("Simulation time must be a positive number.", "error");
        return false;
    }
    config.t_max = tmax;

    let iteration = 1;
    let allMet = false;
    let stagnation = 0;
    let loopError = null;
    let best = { met: -1, blueprint: null };

    // Simulate the current blueprint, render, and return how many targets are met.
    const simulateCurrent = async () => {
        ensureSimulationConfig(state.blueprint).t_max = tmax;
        state.simulationResults = await apiJson("/api/simulate", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ blueprint: state.blueprint, custom_params: state.customParams })
        });
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
            loopError = e;
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
            const refineData = await apiJson("/api/refine", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    blueprint: state.blueprint,
                    simulation_results: state.simulationResults,
                    targets: state.targets,
                    llm: getLlmConfig()
                })
            });

            (Array.isArray(refineData.logs) ? refineData.logs : []).forEach(rawLog => {
                const log = String(rawLog);
                writeConsole(log, log.startsWith("SUCCESS") ? "success" : (log.startsWith("Action") ? "warning" : "info"));
            });

            // Only adopt the refined blueprint if it is actually usable; never nuke
            // state.blueprint with undefined (that used to crash the next iteration).
            if (refineData.blueprint && Array.isArray(refineData.blueprint.nodes) && refineData.blueprint.nodes.length) {
                state.blueprint = refineData.blueprint;
                ensureSimulationConfig(state.blueprint).t_max = tmax;
                const compiled = await compileBlueprint(); // pull tuned parameters into the sliders
                if (!compiled) throw new Error("The optimized model could not be compiled.");
            } else {
                writeConsole("Optimizer returned no usable model; keeping the current one.", "warning");
            }
        } catch (e) {
            loopError = e;
            writeConsole(`Optimization request failed: ${e.message}`, "error");
            break;
        }

        iteration++;
        await new Promise(r => setTimeout(r, 200));
    }

    // If we didn't fully converge, restore the best configuration we found.
    if (!allMet && best.blueprint) {
        state.blueprint = best.blueprint;
        const restored = await compileBlueprint();
        if (!restored && !loopError) loopError = new Error("The best model could not be restored.");
        try { await simulateCurrent(); }
        catch (e) { if (!loopError) loopError = e; }
    }

    if (loopError) {
        updateStatus("Optimization failed", "red");
        writeConsole(`Optimization stopped because of an error: ${loopError.message}`, "error");
        showToast("Optimization failed: " + loopError.message, "error", 8000);
        return false;
    }

    updateStatus("Ready", "green");
    if (allMet) {
        writeConsole("OPTIMIZATION COMPLETE: Model meets all target constraints!", "success");
        showToast("✓ All target conditions met.", "success");
    } else {
        writeConsole(`Stopped after ${iteration} round(s). Best result: ${best.met} / ${total} targets met.`, "error");
        showToast(`Best: ${best.met}/${total} targets met. Try widening target ranges or adding a feedback edge.`, "warn", 7000);
    }
    return true;
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
        
        const results = await apiJson(endpoint, {
            method: method,
            headers: { "Content-Type": "application/json" },
            body: body
        });
        if (!Array.isArray(results)) throw new Error("The database returned an unexpected response.");
        container.innerHTML = "";
        
        if (results.length === 0) {
            container.innerHTML = "<p class='placeholder-text'>No results found.</p>";
            return;
        }
        
        results.forEach(rawResult => {
            const res = rawResult && typeof rawResult === "object" ? rawResult : {};
            const id = String(res.id || "");
            const name = String(res.name || "");
            const source = String(res.source || "");
            const target = String(res.target || "");
            const type = String(res.type || "association").toLowerCase();
            const score = Number(res.score);
            const item = document.createElement("div");
            item.className = "search-result-item";

            if (state.currentDb === "reactome") {
                item.innerHTML = `<strong>${escapeHtml(id)}</strong><br>${escapeHtml(name)}`;
            } else if (state.currentDb === "string" || state.currentDb === "omnipath") {
                const isOmni = state.currentDb === "omnipath";
                const scoreText = Number.isFinite(score) ? score.toFixed(2) : "n/a";
                const reference = res.references == null ? "" : String(res.references).split(';')[0];
                item.innerHTML = `<strong>${escapeHtml(source)} → ${escapeHtml(target)}</strong><br>
                                  Type: ${escapeHtml(type)} | Score: ${scoreText}
                                  ${isOmni && reference ? `<br>Refs: ${escapeHtml(reference)}` : ''}`;
                item.addEventListener("click", () => {
                    const text = document.getElementById("bio-input");
                    const action = type === "inhibition" ? "inhibits" : "activates";
                    text.value += `\n${source} ${action} ${target}.`;
                });
            } else if (state.currentDb === "signor") {
                item.innerHTML = `<strong>${escapeHtml(source)} → ${escapeHtml(target)}</strong><br>
                                  Mech: ${escapeHtml(res.mechanism || "")} | Effect: ${escapeHtml(res.effect || "")}<br>
                                  PMID: ${escapeHtml(res.pmid || "")}`;
                item.addEventListener("click", () => {
                    const text = document.getElementById("bio-input");
                    const action = type === "inhibition" ? "inhibits" : "activates";
                    text.value += `\n${source} ${action} ${target}.`;
                });
            } else if (state.currentDb === "biomodels") {
                item.innerHTML = `<strong>${escapeHtml(id)}</strong>: ${escapeHtml(name)}<br>
                                  <span style="color:#6b7280;font-size:10px">${escapeHtml(String(res.description || "").substring(0, 80))}...</span>`;
                item.addEventListener("click", () => {
                    document.getElementById("biomodel-id-input").value = id;
                    document.querySelector("[data-tab='maple']").click();
                });
            }
            container.appendChild(item);
        });

    } catch(e) {
        container.innerHTML = `<p class="placeholder-text err">Error: ${escapeHtml(e.message)}</p>`;
    }
}

// ==========================================
// AGENT-BASED MODELING (ABM) LOGIC
// ==========================================
let abmResult = null;

async function loadAbmPreset(name) {
    try {
        const bp = await apiJson(`/api/abm/preset/${encodeURIComponent(String(name))}`);
        if (!bp || typeof bp !== "object" || !Array.isArray(bp.cell_types)) {
            throw new Error("The ABM preset returned an unexpected response.");
        }
        state.abmBlueprint = bp;

        // Render info panel. Clamp server-provided RGB channels before using them
        // in a style attribute and escape all textual metadata.
        const panel = document.getElementById("abm-info-panel");
        let html = `<div class="abm-info-title">${escapeHtml(bp.name || name)}</div>
                    <div class="abm-info-desc">${escapeHtml(bp.description || "")}</div>
                    <div class="abm-cell-type-list">`;

        bp.cell_types.forEach(rawCellType => {
            const ct = rawCellType && typeof rawCellType === "object" ? rawCellType : {};
            const channels = Array.isArray(ct.color) ? ct.color.slice(0, 3) : [];
            while (channels.length < 3) channels.push(128);
            const color = `rgb(${channels.map(v => Math.max(0, Math.min(255, Number(v) || 0))).join(',')})`;
            html += `<div class="cell-type-chip">
                        <span class="cell-type-swatch" style="background:${color}"></span>
                        ${escapeHtml(ct.name || "Unnamed cell type")}
                     </div>`;
        });
        html += `</div>`;
        panel.innerHTML = html;
        return true;
    } catch(e) {
        console.error("Failed to load ABM preset", e);
        updateStatus("ABM preset failed", "red");
        showToast("Failed to load ABM preset: " + e.message, "error", 7000);
        return false;
    }
}

async function runAbmSimulation() {
    if (!state.abmBlueprint) return alert("Select an ABM preset first!");
    
    document.getElementById("abm-canvas").style.display = "none";
    document.getElementById("btn-run-abm").textContent = "Simulating...";
    updateStatus("Running CPM...", "yellow");
    
    try {
        const result = await apiJson("/api/abm/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ blueprint: state.abmBlueprint })
        });
        if (!result || !Array.isArray(result.lattice_frames) || result.lattice_frames.length === 0 ||
            !Array.isArray(result.t) || !Array.isArray(result.cell_counts)) {
            throw new Error("The ABM simulator returned incomplete playback data.");
        }
        abmResult = result;

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
        return true;
    } catch(e) {
        console.error("ABM simulation failed", e);
        showToast("ABM simulation failed: " + e.message, "error", 8000);
        document.getElementById("btn-run-abm").textContent = "Run ABM Simulation";
        updateStatus("ABM simulation failed", "red");
        return false;
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
        const result = await apiJson("/api/maple/extract", {
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
        if (!result || typeof result !== "object") throw new Error("MAPLE returned an unexpected response.");

        const resContainer = document.getElementById("maple-results-container");
        resContainer.innerHTML = "";
        const targetView = document.createElement("pre");
        targetView.className = "maple-target-viewer";
        targetView.textContent = JSON.stringify(result.target || {}, null, 2);
        resContainer.appendChild(targetView);

        const valContainer = document.getElementById("maple-validation-container");
        valContainer.innerHTML = `<h4 class="subsection-title">Validation Checks</h4>`;
        const validations = result.validation && Array.isArray(result.validation.results)
            ? result.validation.results
            : [];
        validations.forEach(rawValidation => {
            const val = rawValidation && typeof rawValidation === "object" ? rawValidation : {};
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
                <div class="validation-msg">${escapeHtml(val.message || "")}</div>
                ${val.field_path ? `<div class="validation-field">${escapeHtml(val.field_path)}</div>` : ''}
            `;
            valContainer.appendChild(div);
        });

        const logContainer = document.getElementById("maple-logs-container");
        logContainer.innerHTML = "";
        (Array.isArray(result.logs) ? result.logs : []).forEach(rawLog => {
            const log = String(rawLog);
            const line = document.createElement("div");
            line.className = `log-line ${/fail|error/i.test(log) ? 'error' : 'info'}`;
            line.textContent = log;
            logContainer.appendChild(line);
        });
        updateStatus("Ready", "green");
        return true;
    } catch(e) {
        console.error("MAPLE extraction failed", e);
        updateStatus("MAPLE extraction failed", "red");
        showToast("Extraction failed: " + e.message, "error", 8000);
        return false;
    } finally {
        btn.textContent = "Extract & Validate";
        btn.disabled = false;
    }
}

async function importBioModelSbml() {
    const modelId = document.getElementById("biomodel-id-input").value.trim();
    if (!modelId) return alert("Enter a BioModels ID (e.g., BIOMD0000000006)");

    const btn = document.getElementById("btn-import-sbml");
    btn.textContent = "Importing...";
    btn.disabled = true;
    updateStatus("Importing SBML...", "yellow");

    try {
        const result = await apiJson("/api/biomodels/import", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_id: modelId })
        });
        if (!result || !result.blueprint || typeof result.blueprint !== "object") {
            throw new Error("BioModels returned no usable blueprint.");
        }

        const loaded = await loadBlueprintIntoUI(result.blueprint);
        if (!loaded) {
            showToast(`Imported ${modelId}, but the model needs correction before simulation.`, "warn", 8000);
            return false;
        }
        showToast(`Successfully imported SBML model: ${modelId}`, "success");
        return true;
    } catch(e) {
        console.error("SBML import failed", e);
        updateStatus("SBML import failed", "red");
        showToast("SBML import failed: " + e.message, "error", 8000);
        return false;
    } finally {
        btn.textContent = "Import SBML";
        btn.disabled = false;
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
    const tmax = Number(document.getElementById("sim-tmax").value);
    if (!Number.isFinite(tmax) || tmax <= 0) {
        showToast("Simulation time must be a positive number.", "error");
        return false;
    }
    ensureSimulationConfig(state.blueprint).t_max = tmax;

    const btn = document.getElementById("btn-explore");
    btn.textContent = "Sampling…";
    btn.disabled = true;
    updateStatus("Exploring parameter space...", "yellow");

    try {
        const data = await apiJson("/api/sample", {
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
        if (!data || !Array.isArray(data.t) || !Array.isArray(data.samples)) {
            throw new Error("The parameter sampler returned incomplete results.");
        }
        state.sampleResult = data;
        exploreSort = { key: 'id', dir: 1 };
        renderEnsembleChart(data);
        renderExploreResults(data);
        updateStatus("Ready", "green");
        return true;
    } catch (e) {
        console.error("Parameter exploration failed", e);
        showToast("Exploration failed: " + e.message, "error", 8000);
        updateStatus("Exploration failed", "red");
        return false;
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
    const raw = item && typeof item === "object" ? item : {};
    const rawType = String(raw.type || 'association').toLowerCase();
    const score = Number(raw.score);
    return {
        source: String(raw.source || '').trim(),
        target: String(raw.target || '').trim(),
        type: (rawType === 'activation' || rawType === 'inhibition') ? rawType : 'association',
        score: Number.isFinite(score) ? score : null,
        references: raw.references == null ? "" : String(raw.references),
        pmid: raw.pmid == null ? "" : String(raw.pmid),
        mechanism: raw.mechanism == null ? "" : String(raw.mechanism),
        effect: raw.effect == null ? "" : String(raw.effect)
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
        return false;
    }
    results.innerHTML = `<p class="placeholder-text">Searching ${escapeHtml(connState.db)}…</p>`;
    const proteins = raw.split(/[\s,]+/).filter(Boolean);

    try {
        let data;
        if (connState.db === "omnipath") {
            data = await apiJson("/api/omnipath/interactions", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ proteins })
            });
        } else if (connState.db === "string") {
            data = await apiJson("/api/string/network", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ proteins })
            });
        } else {
            data = await apiJson(`/api/signor/search?q=${encodeURIComponent(proteins[0] || raw)}`);
        }
        if (!Array.isArray(data)) throw new Error("The connection database returned an unexpected response.");
        connState.results = data.map(normalizeConn).filter(c => c.source && c.target);
        renderConnResults();
        return true;
    } catch (e) {
        console.error("Connection search failed", e);
        connState.results = [];
        results.innerHTML = `<p class="placeholder-text err">Search failed: ${escapeHtml(e.message)}</p>`;
        return false;
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
        if (c.score != null) meta.push(`score ${c.score.toFixed(2)}`);
        if (c.mechanism) meta.push(c.mechanism);
        if (c.effect) meta.push(c.effect);
        if (c.pmid) meta.push(`PMID:${c.pmid}`);
        if (c.references) meta.push(`ref ${c.references.split(';')[0]}`);

        html += `<label class="conn-entry ${inModel ? 'in-model' : ''}" data-type="${c.type}">
            <input type="checkbox" data-idx="${i}" ${inModel ? 'checked disabled' : ''}>
            <span class="conn-badge type-${c.type}">${verb}</span>
            <span class="conn-pair"><b>${escapeHtml(c.source)}</b> <span class="arrow">${arrow}</span> <b>${escapeHtml(c.target)}</b></span>
            <span class="conn-meta">${escapeHtml(meta.join(' · '))}</span>
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

async function addSelectedConnections() {
    if (!state.blueprint) { alert("Compile a model first."); return false; }

    const hasExplicitEquations = state.blueprint.odes && Object.keys(state.blueprint.odes).length > 0;
    const hasSpatialReactions = state.blueprint.spatial && state.blueprint.spatial.reactions &&
        Object.keys(state.blueprint.spatial.reactions).length > 0;
    if (state.blueprint.type === "PDE" || hasExplicitEquations || hasSpatialReactions) {
        showToast(
            "Connections cannot be added here because this model's dynamics are defined by explicit equations. Edit those equations in the Model Summary instead.",
            "warn", 8000
        );
        return false;
    }

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

    // Refresh views; existing connections remain intact.
    document.getElementById("blueprint-json-viewer").textContent = JSON.stringify(state.blueprint, null, 2);
    renderCytoscape();
    renderTargets();

    let msg = `Added ${added} connection${added !== 1 ? 's' : ''}.`;
    if (skippedDup) msg += ` ${skippedDup} already existed (unchanged).`;
    if (skippedNode) msg += ` ${skippedNode} skipped (species declined).`;

    if (added > 0) {
        const compiled = await compileBlueprint();
        if (!compiled) {
            showToast(msg + " Recompilation failed; review the model before simulating.", 'error', 8000);
            return false;
        }
    }

    closeConnModal();
    const validationErrors = Array.isArray(state.blueprint.validation_errors)
        ? state.blueprint.validation_errors.filter(Boolean)
        : [];
    updateStatus(validationErrors.length ? "Needs review" : "Ready", validationErrors.length ? "yellow" : "green");
    showToast(msg, added > 0 ? 'success' : 'info');
    return true;
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

async function loadLlmSettings() {
    let hadSaved = false;
    try {
        const raw = localStorage.getItem('biosim_llm');
        const saved = JSON.parse(raw || 'null');
        if (saved && typeof saved === 'object') { Object.assign(state.llm, saved); hadSaved = true; }
    } catch (e) { /* ignore */ }
    updateEngineLabel();

    // If the server has a pre-configured .env for Bedrock, adopt its model/region
    // and auto-select the Bedrock engine. Credential values remain server-side.
    try {
        const env = await apiJson('/api/llm/env');
        if (!env || typeof env !== "object") return;
        if (env.model) state.llm.bedrock_model = String(env.model);
        if (env.region) state.llm.bedrock_region = String(env.region);
        const wantBedrock = env.bedrock_env_ready &&
            (env.engine_default === 'bedrock' || !hadSaved);
        if (wantBedrock && state.llm.engine !== 'bedrock') {
            state.llm.engine = 'bedrock';
            saveLlmSettings();
        }
        updateEngineLabel();
    } catch (e) {
        // This metadata endpoint is optional; other engines remain usable.
        console.warn("Could not load server LLM environment metadata", e.message);
    }
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
        const info = await apiJson('/api/llm/models');
        if (!info || !Array.isArray(info.models)) throw new Error("The model list returned an unexpected response.");
        llmModelsInfo = info;
        document.getElementById('llm-runtime-warning').hidden = !!info.runtime_available;
        renderLlmModels();
    } catch (e) {
        list.innerHTML = `<p class="placeholder-text err">Failed to load models: ${escapeHtml(e.message)}</p>`;
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
        await apiJson('/api/llm/download', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: key })
        });
        showToast(`Downloading ${spec ? spec.label : key}${spec ? ` (~${spec.size_gb} GB)` : ''}. This can take a few minutes.`, 'info', 5000);
        pollDownload(key);
        return true;
    } catch (e) {
        resetDownloadAction(key, 'Retry');
        showToast(`Download request failed: ${e.message}`, 'error', 6000);
        return false;
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
        try {
            st = await apiJson(`/api/llm/status?model=${encodeURIComponent(key)}`);
            if (!st || typeof st !== "object") return;
        } catch (e) { return; /* transient; keep polling */ }

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

