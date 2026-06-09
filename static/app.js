// ==========================================================================
// STATE MANAGEMENT
// ==========================================================================
const state = {
    blueprint: null,
    simulationResults: null,
    equations: null,
    customParams: {},
    targets: [],
    currentDb: 'reactome',
    
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
}

function loadPreset(name) {
    const preset = Presets[name];
    document.getElementById("bio-input").value = preset.text;
    
    // Load targets
    state.targets = JSON.parse(JSON.stringify(preset.targets));
    renderTargets();
    
    // Toggle active classes on buttons
    document.getElementById("load-egfr-btn").classList.toggle("active", name === 'egfr');
    document.getElementById("load-turing-btn").classList.toggle("active", name === 'turing');
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
    } else if (e.target.classList.contains("tgt-type")) {
        tgt.type = e.target.value;
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

function getApiKey() {
    return document.getElementById("api-key-input").value.trim() || null;
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
                api_key: getApiKey()
            })
        });
        
        if (!response.ok) throw new Error(await response.text());
        
        const parsedData = await response.json();
        
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
        
        // Render Equations via KaTeX
        renderEquations();
        
        // Load default values into parameters state
        state.customParams = {};
        if (data.parameters) {
            Object.assign(state.customParams, data.parameters);
        }
        
        // Set simulation tmax
        const tmax = state.blueprint.simulation_config?.t_max || 50;
        document.getElementById("sim-tmax").value = tmax;
        
        // RenderSliders
        renderParameterSliders();
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
    
    sortedParamNames.forEach(pname => {
        const val = state.customParams[pname];
        const group = document.createElement("div");
        group.className = "slider-group";
        
        // Guess limits based on param type
        let min = 0.0, max = 5.0, step = 0.05;
        if (pname.startsWith("deg_")) { min = 0.0; max = 2.0; step = 0.01; }
        else if (pname.endsWith("_n")) { min = 1.0; max = 6.0; step = 0.5; }
        else if (pname.endsWith("_Kd")) { min = 0.1; max = 10.0; step = 0.1; }
        
        group.innerHTML = `
            <div class="slider-labels">
                <span class="slider-name">${pname}</span>
                <span class="slider-value" id="val-${pname}">${val.toFixed(2)}</span>
            </div>
            <input type="range" min="${min}" max="${max}" step="${step}" value="${val}" data-param="${pname}">
        `;
        
        container.appendChild(group);
    });
    
    // Slider event listener
    container.querySelectorAll("input[type='range']").forEach(slider => {
        slider.addEventListener("input", (e) => {
            const pname = e.target.getAttribute("data-param");
            const val = parseFloat(e.target.value);
            document.getElementById(`val-${pname}`).textContent = val.toFixed(2);
            state.customParams[pname] = val;
        });
    });
}

function renderEquations() {
    const container = document.getElementById("equations-container");
    container.innerHTML = "";
    
    if (!state.equations) return;
    
    Object.keys(state.equations).forEach(var_name => {
        const latex_str = state.equations[var_name];
        const item = document.createElement("div");
        item.className = "math-item";
        
        // Render using KaTeX
        katex.render(latex_str, item, { throwOnError: false });
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
    
    const container = document.getElementById("db-results-container");
    container.innerHTML = `<p class="placeholder-text">Searching database...</p>`;
    
    try {
        if (state.currentDb === 'reactome') {
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
        } else {
            // STRING DB Search (comma separated proteins)
            const list = query.split(/[\s,]+/).filter(Boolean);
            const response = await fetch("/api/string/network", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ proteins: list })
            });
            const data = await response.json();
            
            container.innerHTML = "";
            if (data.length === 0) {
                container.innerHTML = `<p class="placeholder-text">No protein interactions found.</p>`;
                return;
            }
            
            // Build a description list
            let descLines = [];
            data.forEach(edge => {
                descLines.push(`${edge.source} activates ${edge.target}.`);
            });
            
            const div = document.createElement("div");
            div.className = "search-result-item";
            div.innerHTML = `
                <strong>STRING Network (${list.join(', ')})</strong><br>
                <span style="font-size:10px; color:var(--text-secondary);">${data.length} connections found. Click to load descriptions.</span>
            `;
            div.addEventListener("click", () => {
                document.getElementById("bio-input").value = descLines.join('\n');
                state.targets = list.slice(0, 3).map(p => ({
                    species: p.toUpperCase(),
                    type: "peak_time",
                    min: 5,
                    max: 20
                }));
                renderTargets();
            });
            container.appendChild(div);
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
        
        if (reactions.length === 0) {
            alert("No reactions catalogued for this pathway.");
            updateStatus("Ready", "green");
            return;
        }
        
        // Assemble text description
        let lines = [`# Pathway: ${pathwayName} (${pathwayId})`];
        
        // Mocking interaction bindings: for simple demo, connect cascade sequentially
        let proteins = [];
        reactions.forEach((rxn, idx) => {
            const rxn_words = rxn.name.split(" ");
            const key_proteins = rxn_words.filter(w => w.length > 2 && w === w.toUpperCase() && isNaN(w));
            if (key_proteins.length >= 2) {
                lines.push(`${key_proteins[0]} activates ${key_proteins[1]}.`);
                proteins.push(key_proteins[0], key_proteins[1]);
            } else if (key_proteins.length === 1 && idx > 0 && proteins.length > 0) {
                lines.push(`${proteins[proteins.length-1]} activates ${key_proteins[0]}.`);
                proteins.push(key_proteins[0]);
            }
        });
        
        if (lines.length === 1) {
            // Default sequential link if mapping fails
            lines.push("EGF activates EGFR.");
            lines.push("EGFR activates RAS.");
            lines.push("RAS activates RAF.");
            lines.push("RAF activates MEK.");
            lines.push("MEK activates ERK.");
            lines.push("ERK inhibits EGFR.");
        }
        
        lines.push("EGF starts at 10.0.");
        lines.push("EGFR starts at 1.0.");
        
        document.getElementById("bio-input").value = lines.join('\n');
        
        // Update Targets
        state.targets = [
            { species: "EGFR", type: "peak_time", min: 2, max: 10 },
            { species: "EGFR", type: "decay_ratio", max: 0.15 }
        ];
        renderTargets();
        
        updateStatus("Ready", "green");
        alert(`Loaded template structure from Reactome: ${pathwayName}`);
    } catch(e) {
        updateStatus("Ready", "green");
        alert("Failed to load pathway details.");
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
        evaluateSimulationTargets();
        
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
    
    Object.keys(species).forEach((nid, index) => {
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
    
    state.blueprint.nodes.forEach(node => {
        elements.push({
            data: { 
                id: node.id, 
                label: `${node.id}\n(${node.initial_value || 0})`
            }
        });
    });
    
    state.blueprint.edges.forEach((edge, index) => {
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
function evaluateSimulationTargets() {
    if (!state.simulationResults || state.targets.length === 0) return;
    
    const t = state.simulationResults.t;
    const species = state.simulationResults.species;
    
    let metCount = 0;
    const container = document.getElementById("target-eval-list");
    container.innerHTML = "";
    
    state.targets.forEach(tgt => {
        const y = species[tgt.species] || [];
        const item = document.createElement("div");
        
        let met = false;
        let details = "";
        
        if (y.length > 0) {
            const t_arr = np_array(t);
            const y_arr = np_array(y);
            
            if (tgt.type === 'peak_time') {
                const max_idx = argmax(y_arr);
                const peak_t = t_arr[max_idx];
                
                const minMet = tgt.min === undefined || peak_t >= tgt.min;
                const maxMet = tgt.max === undefined || peak_t <= tgt.max;
                met = minMet && maxMet;
                details = `Peak at ${peak_t.toFixed(2)} min (Target: ${tgt.min || 0} - ${tgt.max || 'max'})`;
            } else if (tgt.type === 'peak_value') {
                const peak_v = Math.max(...y);
                const minMet = tgt.min === undefined || peak_v >= tgt.min;
                const maxMet = tgt.max === undefined || peak_v <= tgt.max;
                met = minMet && maxMet;
                details = `Peak value: ${peak_v.toFixed(2)} (Target: ${tgt.min || 0} - ${tgt.max || 'max'})`;
            } else if (tgt.type === 'decay_ratio') {
                const peak_v = Math.max(...y);
                const final_v = y[y.length - 1];
                const ratio = peak_v > 0 ? final_v / peak_v : 1.0;
                met = tgt.max === undefined || ratio <= tgt.max;
                details = `Decays to ${(ratio*100).toFixed(1)}% of peak (Target: <= ${(tgt.max*100).toFixed(0)}%)`;
            } else if (tgt.type === 'steady_state') {
                const final_v = y[y.length - 1];
                const diff = Math.abs(final_v - (tgt.value || 0));
                const tolerance = tgt.tolerance || 0.1;
                met = diff <= tolerance;
                details = `Steady state at ${final_v.toFixed(2)} (Target: ${tgt.value} +/- ${tolerance})`;
            }
        }
        
        if (met) metCount++;
        
        item.className = `eval-item ${met ? 'met' : 'failed'}`;
        item.innerHTML = `
            <div class="eval-icon">${met ? '🟢' : '🔴'}</div>
            <div class="eval-details">
                <div class="target-title">${tgt.species} ${tgt.type.replace('_', ' ')}</div>
                <div class="target-status-msg">${details}</div>
            </div>
        `;
        container.appendChild(item);
    });
    
    // Update Circle Progress
    const pct = Math.round((metCount / state.targets.length) * 100);
    document.getElementById("target-score-percentage").textContent = `${pct}%`;
    
    // Progress Ring offset
    const circle = document.getElementById("target-progress-bar");
    const radius = circle.r.baseVal.value;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (pct / 100) * circumference;
    circle.style.strokeDashoffset = offset;
}

// Helpers for array calculations in js
function np_array(arr) { return arr; }
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
    
    // Switch to Feedback Tab
    document.querySelector("[data-tab='feedback']").click();
    
    document.getElementById("feedback-log").innerHTML = "";
    writeConsole("Starting automated closed-loop feedback loop...", "info");
    updateStatus("Optimizing...", "yellow");
    
    let maxIterations = 5;
    let iteration = 1;
    let allMet = false;
    
    while (iteration <= maxIterations && !allMet) {
        writeConsole(`--- ITERATION ${iteration} ---`, "info");
        
        // 1. Run Simulation
        writeConsole("Solving numerical equations...", "info");
        
        const tmax = parseFloat(document.getElementById("sim-tmax").value);
        state.blueprint.simulation_config.t_max = tmax;
        
        let simResponse;
        try {
            simResponse = await fetch("/api/simulate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    blueprint: state.blueprint,
                    custom_params: state.customParams
                })
            });
            state.simulationResults = await simResponse.json();
            
            // Re-draw ODE/PDE charts
            renderOdeChart();
            evaluateSimulationTargets();
        } catch(e) {
            writeConsole(`Simulation failed: ${e.message}`, "error");
            break;
        }
        
        // 2. Evaluate and Refine
        writeConsole("Evaluating target constraints...", "info");
        
        try {
            const refineResponse = await fetch("/api/refine", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    blueprint: state.blueprint,
                    simulation_results: state.simulationResults,
                    targets: state.targets,
                    api_key: getApiKey()
                })
            });
            
            const refineData = await refineResponse.json();
            
            // Print refining logs
            refineData.logs.forEach(logLine => {
                if (logLine.startsWith("SUCCESS")) {
                    writeConsole(logLine, "success");
                } else if (logLine.startsWith("Action")) {
                    writeConsole(logLine, "warning");
                } else {
                    writeConsole(logLine, "info");
                }
            });
            
            allMet = refineData.success;
            if (allMet) {
                state.blueprint = refineData.blueprint;
                // Update sliders
                await compileBlueprint();
                break;
            }
            
            // Update current blueprint and params for next iteration
            state.blueprint = refineData.blueprint;
            
            // Pull the adjusted parameters out
            await compileBlueprint();
            
            iteration++;
            
            // Artificial delay to feel alive
            await new Promise(r => setTimeout(r, 1200));
            
        } catch (e) {
            writeConsole(`Optimization endpoint failed: ${e.message}`, "error");
            break;
        }
    }
    
    if (allMet) {
        writeConsole("OPTIMIZATION COMPLETE: Model meets all experimental constraints!", "success");
        updateStatus("Ready", "green");
        alert("Success! All target conditions met.");
    } else {
        writeConsole("OPTIMIZATION PAUSED: Maximum iterations reached. Adjust parameters manually or check network topology.", "error");
        updateStatus("Ready", "green");
        alert("Completed optimization loop. Some targets remain unmet.");
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
                api_key: getApiKey()
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

