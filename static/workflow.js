/* ===========================================================================
 * Simulation workflow UI.
 *
 * The backend owns validation and execution; this file owns presentation and the
 * project document. Every stage sends its state to the API and renders whatever
 * comes back, so the UI can never claim a stage is valid on its own authority.
 * =========================================================================== */
"use strict";

/* --------------------------------------------------------------------------
 * State
 * ----------------------------------------------------------------------- */
const STAGES = [
    { id: "domain", label: "Domain (CAD)" },
    { id: "topology", label: "Nodes / edges / cells" },
    { id: "mesh", label: "Mesh" },
    { id: "model", label: "PDE model" },
    { id: "conditions", label: "Conditions" },
    { id: "approach", label: "Approach" },
    { id: "run", label: "Validate & run" },
    { id: "results", label: "Results" },
];

const state = {
    project: null,
    stage: "domain",
    validation: null,
    view: "graph",
    tool: "select",
    selection: { nodes: [], edges: [], cells: [] },
    run: null,
    runPoll: null,
    results: null,
    frame: 0,
    selectedCellId: null,
    // Lattice draw transform, written by drawLattice and read by the canvas
    // click handler so a click maps back to a cell.
    resultsTransform: null,
    playing: false,
    playTimer: null,
    chart: null,
    // Signature of the rows currently in the results cell table. Playback calls
    // renderCellList 25x a second; this lets an unchanged frame skip the rebuild.
    cellListKey: null,
    // Canvas transform: world (domain units) -> screen (pixels).
    camera: { scale: 1, offsetX: 0, offsetY: 0, dragging: false, lastX: 0, lastY: 0 },
};

/* --------------------------------------------------------------------------
 * Small helpers
 * ----------------------------------------------------------------------- */
const $ = (id) => document.getElementById(id);
/**
 * Guarded event binding. Every control was bound with $("id").addEventListener(...),
 * so ONE renamed or removed id threw a TypeError partway through bindControls and
 * left every later control in the page dead -- with no error the user could see.
 * A missing id is now a console warning and the rest of the page still binds.
 */
const on = (id, event, handler, options) => {
    const node = $(id);
    if (!node) {
        console.warn(`[workflow] no element #${id} to bind "${event}" to`);
        return false;
    }
    node.addEventListener(event, handler, options);
    return true;
};
const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
};

function escapeHtml(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, (c) => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
    ));
}

function fmt(value, digits = 3) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    if (number !== 0 && (Math.abs(number) < 1e-3 || Math.abs(number) >= 1e5)) {
        return number.toExponential(2);
    }
    return String(Number(number.toFixed(digits)));
}

/**
 * Min/max over one or more numeric series in a SINGLE linear pass.
 *
 * Math.min(...array) / Math.max(...array) spread every element onto the call
 * stack and throw RangeError past roughly 65k arguments. Both call sites were
 * inside a draw or a readout with no catch, so a fine mesh or a long run did not
 * report an error -- the plot simply went blank. This also avoids the temporary
 * array that `output.concat(target)` allocated just to be scanned once.
 *
 * `count` is the number of FINITE values seen, so callers can tell "the range is
 * 0…0" apart from "there was nothing to measure" instead of drawing at NaN.
 */
function extent(...series) {
    let low = Infinity, high = -Infinity, count = 0;
    for (let s = 0; s < series.length; s += 1) {
        const values = series[s] || [];
        for (let i = 0; i < values.length; i += 1) {
            const value = Number(values[i]);
            if (!Number.isFinite(value)) continue;
            if (value < low) low = value;
            if (value > high) high = value;
            count += 1;
        }
    }
    return count ? { low, high, count } : { low: 0, high: 0, count: 0 };
}

function toast(message, kind = "info", ms = 5000) {
    const host = $("wf-toasts");
    const node = el("div", `wf-toast ${kind}`, message);
    // A toast is sometimes the ONLY place a refusal is explained (an unavailable
    // engine, invalid JSON), so it has to reach assistive technology rather than
    // being a decorative div. Errors and warnings interrupt; info does not.
    node.setAttribute("role", kind === "error" || kind === "warn" ? "alert" : "status");
    node.setAttribute("aria-live", kind === "error" || kind === "warn" ? "assertive" : "polite");
    host.appendChild(node);
    setTimeout(() => node.remove(), ms);
}

function setStatus(text, colour = "green") {
    $("wf-status-text").textContent = text;
    $("wf-status-dot").className = `status-dot ${colour}`;
}

/** Single network path, so every failure is surfaced the same way. */
async function api(path, options) {
    let response;
    try {
        response = await fetch(path, options);
    } catch (error) {
        throw new Error(`Could not reach the server: ${error.message}`);
    }
    let payload = null;
    const text = await response.text();
    if (text) {
        try { payload = JSON.parse(text); } catch (error) { payload = { detail: text }; }
    }
    if (!response.ok) {
        const detail = payload && payload.detail !== undefined ? payload.detail : `HTTP ${response.status}`;
        const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
        error.status = response.status;
        error.payload = payload;
        throw error;
    }
    return payload;
}

const post = (path, body) => api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
});

function download(filename, contents, type = "application/json") {
    const blob = new Blob([contents], { type });
    const url = URL.createObjectURL(blob);
    const link = el("a");
    link.href = url;
    link.download = filename;
    // The anchor must be IN the document: click() on a detached node is a no-op in
    // Firefox. And revoking synchronously can abort a download that has only just
    // been handed to the browser, so it is deferred. Five call sites report success
    // after this returns, so a silent failure here was a false success.
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* --------------------------------------------------------------------------
 * Issue rendering: one shape for every stage
 * ----------------------------------------------------------------------- */
function renderIssues(containerId, payload, okMessage) {
    const host = $(containerId);
    if (!host) return;
    host.innerHTML = "";
    const issues = (payload && payload.issues) || [];
    if (!issues.length) {
        if (okMessage) {
            host.appendChild(el("div", "wf-issue-ok", okMessage));
        }
        return;
    }
    issues.forEach((issue) => {
        const node = el("div", `wf-issue ${issue.severity === "warning" ? "warning" : ""}`);
        node.appendChild(el("span", null, issue.message));
        if (issue.path) {
            node.appendChild(document.createTextNode(" "));
            node.appendChild(el("code", null, issue.path));
        }
        host.appendChild(node);
    });
}

/* --------------------------------------------------------------------------
 * Stage rail
 * ----------------------------------------------------------------------- */
function buildStageRail() {
    const list = $("wf-stage-list");
    list.innerHTML = "";
    STAGES.forEach((stage, index) => {
        const item = el("li");
        item.dataset.stage = stage.id;
        item.tabIndex = 0;
        item.setAttribute("role", "button");
        item.appendChild(el("span", "wf-stage-index", String(index + 1)));
        item.appendChild(el("span", null, stage.label));
        item.appendChild(el("span", "wf-badge skipped", "–"));
        const go = () => showStage(stage.id);
        item.addEventListener("click", go);
        item.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") { event.preventDefault(); go(); }
        });
        list.appendChild(item);
    });
}

function showStage(id) {
    state.stage = id;
    document.querySelectorAll(".wf-stage").forEach((section) => {
        section.hidden = section.dataset.stage !== id;
    });
    document.querySelectorAll("#wf-stage-list li").forEach((item) => {
        item.classList.toggle("active", item.dataset.stage === id);
    });
    // Relocate the ONE shared canvas into this stage's slot. Stages 1-3 all work on
    // the same geometry, so they share a single canvas rather than three; the canvas
    // previously lived inside stage 1, which hid it on exactly the stages whose
    // instructions say to click it.
    const viewport = $("wf-viewport");
    const slot = document.querySelector(`.wf-canvas-slot[data-slot="${id}"]`);
    if (viewport) {
        if (slot) {
            if (viewport.parentElement !== slot) slot.insertBefore(viewport, slot.firstChild);
            viewport.hidden = false;
        } else {
            viewport.hidden = true;
        }
    }
    if (id === "domain" || id === "topology" || id === "mesh") drawCanvas();
    if (id === "results") drawResults();
}

function paintStageBadges(validation) {
    const map = {
        domain: "domain", topology: "topology", mesh: "mesh",
        model: "model", conditions: "conditions", approach: "approach",
    };
    // The run and results stages are NOT part of the backend validation payload --
    // their status is the live run's own state. Deriving them here is what stops a
    // completed run from leaving both badges reading "not done yet", which made a
    // successful simulation look like it had never happened.
    const runBadge = () => {
        const run = state.run;
        if (!run) return { status: "skipped", note: "no run yet" };
        if (["failed", "invalid", "cancelled"].includes(run.state)) {
            return {
                status: "invalid",
                note: run.error || `run ${run.state}`,
                errors: (run.issues || []).length,
            };
        }
        if (run.state === "completed") return { status: "valid", note: "run completed" };
        return { status: "skipped", note: run.state };
    };
    const resultsBadge = () => {
        const run = state.run;
        if (state.results) return { status: "valid", note: "results loaded" };
        if (run && ["failed", "invalid", "cancelled"].includes(run.state)) {
            return { status: "invalid", note: "no results: the run did not complete" };
        }
        return { status: "skipped", note: "no results yet" };
    };

    document.querySelectorAll("#wf-stage-list li").forEach((item) => {
        const badge = item.querySelector(".wf-badge");
        if (!badge) return;
        let stage;
        if (item.dataset.stage === "run") {
            stage = runBadge();
        } else if (item.dataset.stage === "results") {
            stage = resultsBadge();
        } else {
            const key = map[item.dataset.stage];
            if (!key || !validation || !validation.stages) return;
            stage = validation.stages[key];
        }
        if (!stage) return;
        badge.className = `wf-badge ${stage.status === "valid" ? "valid"
            : stage.status === "invalid" ? "invalid" : "skipped"}`;
        badge.textContent = stage.status === "valid" ? "✓" : stage.status === "invalid" ? "!" : "–";
        badge.title = stage.status === "invalid" && stage.errors
            ? `${stage.errors} error(s)` : stage.note || stage.status;
    });
}

/* --------------------------------------------------------------------------
 * Canvas camera: world (domain units) <-> screen (pixels)
 * ----------------------------------------------------------------------- */
function worldToScreen(x, y) {
    const camera = state.camera;
    return {
        x: x * camera.scale + camera.offsetX,
        // Screen y grows downward; world y grows upward, so it is flipped here.
        y: -y * camera.scale + camera.offsetY,
    };
}

function screenToWorld(x, y) {
    const camera = state.camera;
    return {
        x: (x - camera.offsetX) / camera.scale,
        y: -(y - camera.offsetY) / camera.scale,
    };
}

function domainBounds() {
    const domain = state.project && state.project.domain;
    if (!domain) return [0, 0, 1, 1];
    const kind = domain.kind;
    if (kind === "interval") {
        const spec = domain.interval || {};
        const x0 = Number(spec.x_min || 0);
        return [x0, -0.1, x0 + Number(spec.length || 1), 0.1];
    }
    if (kind === "rectangle") {
        const spec = domain.rectangle || {};
        const x0 = Number(spec.x_min || 0), y0 = Number(spec.y_min || 0);
        return [x0, y0, x0 + Number(spec.width || 1), y0 + Number(spec.height || 1)];
    }
    if (kind === "circle") {
        const spec = domain.circle || {};
        const cx = Number(spec.cx || 0), cy = Number(spec.cy || 0), r = Number(spec.radius || 1);
        return [cx - r, cy - r, cx + r, cy + r];
    }
    if (kind === "polygon") {
        const points = (domain.polygon && domain.polygon.points) || [];
        if (!points.length) return [0, 0, 1, 1];
        // Single pass, no spread: a traced outline can carry tens of thousands of
        // points, and Math.min(...xs) throws RangeError once it does.
        const xs = extent(points.map((p) => Number(p[0])));
        const ys = extent(points.map((p) => Number(p[1])));
        if (!xs.count || !ys.count) return [0, 0, 1, 1];
        return [xs.low, ys.low, xs.high, ys.high];
    }
    return [0, 0, 1, 1];
}

function fitView() {
    const canvas = $("wf-canvas");
    if (!canvas) return;
    const [x0, y0, x1, y1] = domainBounds();
    const width = Math.max(x1 - x0, 1e-9);
    const height = Math.max(y1 - y0, 1e-9);
    const padding = 60;
    const scale = Math.min(
        (canvas.width - padding * 2) / width,
        (canvas.height - padding * 2) / height
    );
    state.camera.scale = Number.isFinite(scale) && scale > 0 ? scale : 1;
    state.camera.offsetX = canvas.width / 2 - ((x0 + x1) / 2) * state.camera.scale;
    state.camera.offsetY = canvas.height / 2 + ((y0 + y1) / 2) * state.camera.scale;
    drawCanvas();
}

function bindCanvasCamera() {
    const canvas = $("wf-canvas");
    if (!canvas) return;

    canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const px = (event.clientX - rect.left) * (canvas.width / rect.width);
        const py = (event.clientY - rect.top) * (canvas.height / rect.height);
        const before = screenToWorld(px, py);
        const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
        state.camera.scale = Math.max(1e-6, Math.min(1e7, state.camera.scale * factor));
        // Keep the point under the cursor fixed, which is what makes zoom feel right.
        const after = screenToWorld(px, py);
        state.camera.offsetX += (after.x - before.x) * state.camera.scale;
        state.camera.offsetY -= (after.y - before.y) * state.camera.scale;
        drawCanvas();
    }, { passive: false });

    canvas.addEventListener("mousedown", (event) => {
        const point = canvasPoint(event);
        if (event.button === 1 || event.shiftKey || state.tool === "select" || state.tool === "move") {
            state.camera.dragging = state.tool !== "move";
            state.camera.lastX = point.px;
            state.camera.lastY = point.py;
        }
        handleCanvasMouseDown(point, event);
    });

    canvas.addEventListener("mousemove", (event) => {
        const point = canvasPoint(event);
        $("canvas-readout").textContent =
            `x = ${fmt(point.world.x)}   y = ${fmt(point.world.y)}   ${units()}`;
        if (state.camera.dragging) {
            state.camera.offsetX += point.px - state.camera.lastX;
            state.camera.offsetY += point.py - state.camera.lastY;
            state.camera.lastX = point.px;
            state.camera.lastY = point.py;
            drawCanvas();
            return;
        }
        handleCanvasMouseMove(point, event);
    });

    const endDrag = () => { state.camera.dragging = false; handleCanvasMouseUp(); };
    canvas.addEventListener("mouseup", endDrag);
    canvas.addEventListener("mouseleave", endDrag);
}

function canvasPoint(event) {
    const canvas = $("wf-canvas");
    const rect = canvas.getBoundingClientRect();
    const px = (event.clientX - rect.left) * (canvas.width / rect.width);
    const py = (event.clientY - rect.top) * (canvas.height / rect.height);
    return { px, py, world: screenToWorld(px, py) };
}

/* --------------------------------------------------------------------------
 * Units
 *
 * There are exactly TWO stored base units -- length and time -- both living on
 * project.domain, so they survive export/import with the rest of the document.
 * Everything else is DERIVED: from those two, or from the selected field's own
 * declared units. Nothing is asked for twice, and anything that cannot be
 * derived is left unlabelled rather than given an invented unit.
 * ----------------------------------------------------------------------- */

/**
 * Sanitised at the SOURCE rather than at each call site: these values come from
 * an imported project file -- this app's untrusted entry point -- and are
 * interpolated into innerHTML in several places, so one missed call site would
 * be an injection. A unit symbol never legitimately contains markup characters.
 *
 * Compose derived units FROM sanitised parts; never sanitise a composed string,
 * which would strip the "²" and "·" the composition just added.
 */
function sanitiseUnit(raw) {
    return String(raw == null ? "" : raw).replace(/[^\w°µ/^.\-]/g, "").slice(0, 12);
}

/** The stored LENGTH unit, e.g. "um". */
function units() {
    return sanitiseUnit((state.project && state.project.domain && state.project.domain.units) || "");
}

/** The stored TIME unit, e.g. "s". Defaults to seconds so no label reads blank. */
function timeUnits() {
    return sanitiseUnit(
        (state.project && state.project.domain && state.project.domain.time_units) || "") || "s";
}

/** "" -> "", "um" -> " (um)". Keeps an underivable unit from rendering "()". */
function unitSuffix(unit) {
    return unit ? ` (${unit})` : "";
}

/** A field's OWN declared units, e.g. "mM". Empty when the field declares none. */
function fieldUnit(field) {
    return sanitiseUnit((field && field.units) || "");
}

/** L² -- an area: cell area, and a 2D element size. */
function areaUnit() {
    const length = units();
    return length ? `${length}²` : "";
}

/** L²/T -- a diffusion coefficient. */
function diffusionUnit() {
    const length = units(), time = timeUnits();
    return length && time ? `${length}²/${time}` : "";
}

/** L/T -- an advection velocity, and a Robin transfer coefficient. */
function velocityUnit() {
    const length = units(), time = timeUnits();
    return length && time ? `${length}/${time}` : "";
}

/** [u]/T -- a reaction or source term, which is a rate of change of the field. */
function rateUnit(field) {
    const value = fieldUnit(field), time = timeUnits();
    return value && time ? `${value}/${time}` : "";
}

/**
 * [u]·L/T -- a diffusive flux. From the flux term D·∂u/∂n:
 * (L²/T) · ([u]/L) = [u]·L/T. Derived, never asked for.
 */
function fluxUnit(field) {
    const value = fieldUnit(field), length = units(), time = timeUnits();
    return value && length && time ? `${value}·${length}/${time}` : "";
}

/** The field a stage-5 condition is being written against, for its units. */
function conditionField(name) {
    const wanted = name !== undefined ? name : ($("bc-field") || {}).value;
    return (((state.project || {}).model || {}).fields || [])
        .find((field) => field.name === wanted) || null;
}

/**
 * A mesh element_size is a LENGTH in 1D and an AREA in 2D -- meshing.py stores
 * segment lengths for line2 and element areas for quad4/tri3. Reading the wrong
 * one off the screen is a squared error, so the dimension decides the label.
 */
function elementSizeUnit(mesh) {
    const dimension = Number((mesh || {}).dimension);
    if (dimension === 1) return units();
    if (dimension === 2) return areaUnit();
    // Fall back on the element kind when an older document carries no dimension.
    const kind = String((mesh || {}).element_kind || ((mesh || {}).stats || {}).element_kind || "");
    if (kind === "line2") return units();
    if (kind === "quad4" || kind === "tri3") return areaUnit();
    return "";
}


/* --------------------------------------------------------------------------
 * Canvas rendering
 *
 * One canvas serves the domain, topology and mesh stages. Draw order is
 * background -> mesh -> domain outline -> boundary markers -> cells -> edges ->
 * nodes, so the things a user clicks are always on top of the things they do not.
 * ----------------------------------------------------------------------- */
const BC_COLOURS = {
    dirichlet: "#00f2fe",
    neumann: "#b180ff",
    no_flux: "#6b7280",
    robin: "#ffb020",
    periodic: "#00ff87",
};

function drawCanvas() {
    const canvas = $("wf-canvas");
    if (!canvas || !state.project) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(7, 9, 19, 0.9)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (state.view === "mesh" || state.stage === "mesh") drawMesh(ctx);
    drawDomain(ctx);
    drawBoundaryMarkers(ctx);
    if (state.view !== "mesh") drawTopology(ctx);

    $("canvas-mode-label").textContent =
        `${state.project.domain.kind} · ${state.view} view · ${state.tool} tool`;
}

function drawDomain(ctx) {
    const domain = state.project.domain;
    ctx.save();
    ctx.strokeStyle = "#4facfe";
    ctx.lineWidth = 2;
    ctx.setLineDash([]);

    if (domain.kind === "interval") {
        const spec = domain.interval || {};
        const x0 = Number(spec.x_min || 0);
        const x1 = x0 + Number(spec.length || 0);
        const a = worldToScreen(x0, 0), b = worldToScreen(x1, 0);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        // End caps make the extent unambiguous at any zoom.
        [a, b].forEach((point) => {
            ctx.beginPath();
            ctx.moveTo(point.x, point.y - 10);
            ctx.lineTo(point.x, point.y + 10);
            ctx.stroke();
        });
        annotate(ctx, a, b, `L = ${fmt(spec.length)} ${domain.units}`);
    } else if (domain.kind === "rectangle") {
        const spec = domain.rectangle || {};
        const x0 = Number(spec.x_min || 0), y0 = Number(spec.y_min || 0);
        const x1 = x0 + Number(spec.width || 0), y1 = y0 + Number(spec.height || 0);
        const tl = worldToScreen(x0, y1), br = worldToScreen(x1, y0);
        ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
        annotate(ctx, worldToScreen(x0, y0), worldToScreen(x1, y0),
            `${fmt(spec.width)} ${domain.units}`);
        annotate(ctx, worldToScreen(x0, y0), worldToScreen(x0, y1),
            `${fmt(spec.height)} ${domain.units}`, true);
    } else if (domain.kind === "circle") {
        const spec = domain.circle || {};
        const centre = worldToScreen(Number(spec.cx || 0), Number(spec.cy || 0));
        const radius = Number(spec.radius || 0) * state.camera.scale;
        ctx.beginPath();
        ctx.arc(centre.x, centre.y, Math.max(radius, 1), 0, Math.PI * 2);
        ctx.stroke();
        // Radius leader line, so the annotated dimension is visibly what it measures.
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(centre.x, centre.y);
        ctx.lineTo(centre.x + radius, centre.y);
        ctx.stroke();
        ctx.setLineDash([]);
        label(ctx, centre.x + radius / 2, centre.y - 8, `r = ${fmt(spec.radius)} ${domain.units}`);
    } else if (domain.kind === "polygon") {
        const points = (domain.polygon && domain.polygon.points) || [];
        if (points.length >= 2) {
            ctx.beginPath();
            points.forEach((point, index) => {
                const screen = worldToScreen(Number(point[0]), Number(point[1]));
                if (index === 0) ctx.moveTo(screen.x, screen.y);
                else ctx.lineTo(screen.x, screen.y);
            });
            ctx.closePath();
            ctx.stroke();
            ctx.fillStyle = "#4facfe";
            points.forEach((point, index) => {
                const screen = worldToScreen(Number(point[0]), Number(point[1]));
                ctx.beginPath();
                ctx.arc(screen.x, screen.y, 3.5, 0, Math.PI * 2);
                ctx.fill();
                label(ctx, screen.x + 7, screen.y - 7, String(index + 1), "#6b7280");
            });
        }
    }
    ctx.restore();
}

function annotate(ctx, from, to, text, vertical = false) {
    ctx.save();
    ctx.strokeStyle = "rgba(156, 163, 175, 0.5)";
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    const offset = vertical ? -18 : 20;
    const a = vertical ? { x: from.x + offset, y: from.y } : { x: from.x, y: from.y + offset };
    const b = vertical ? { x: to.x + offset, y: to.y } : { x: to.x, y: to.y + offset };
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    ctx.restore();
    label(ctx, (a.x + b.x) / 2, (a.y + b.y) / 2 - 5, text, "#9ca3af", "center");
}

function label(ctx, x, y, text, colour = "#f3f4f6", align = "left") {
    ctx.save();
    ctx.fillStyle = colour;
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.textAlign = align;
    ctx.fillText(text, x, y);
    ctx.restore();
}

function drawMesh(ctx) {
    const mesh = state.project.mesh;
    if (!mesh || !mesh.nodes) return;
    ctx.save();
    ctx.strokeStyle = "rgba(0, 242, 254, 0.22)";
    ctx.lineWidth = 1;
    (mesh.elements || []).forEach((element) => {
        if (!element || element.length < 2) return;
        ctx.beginPath();
        element.forEach((index, position) => {
            const node = mesh.nodes[index];
            if (!node) return;
            const screen = worldToScreen(Number(node[0]), Number(node[1]));
            if (position === 0) ctx.moveTo(screen.x, screen.y);
            else ctx.lineTo(screen.x, screen.y);
        });
        if (element.length > 2) ctx.closePath();
        ctx.stroke();
    });
    // Only draw mesh vertices when they are far enough apart to be readable.
    if (mesh.nodes.length <= 3000 && state.camera.scale > 3) {
        ctx.fillStyle = "rgba(0, 242, 254, 0.5)";
        mesh.nodes.forEach((node) => {
            const screen = worldToScreen(Number(node[0]), Number(node[1]));
            ctx.fillRect(screen.x - 1, screen.y - 1, 2, 2);
        });
    }
    ctx.restore();
}

function drawBoundaryMarkers(ctx) {
    const mesh = state.project.mesh;
    const conditions = state.project.conditions || [];
    if (!mesh || !mesh.boundaries) return;
    ctx.save();
    Object.entries(mesh.boundaries).forEach(([boundaryId, info]) => {
        const applied = conditions.filter(
            (c) => c.boundary === boundaryId && c.enabled !== false);
        const kind = applied.length ? applied[applied.length - 1].type : "no_flux";
        ctx.fillStyle = BC_COLOURS[kind] || "#6b7280";
        (info.nodes || []).forEach((index) => {
            const node = mesh.nodes[index];
            if (!node) return;
            const screen = worldToScreen(Number(node[0]), Number(node[1]));
            ctx.beginPath();
            ctx.arc(screen.x, screen.y, 3, 0, Math.PI * 2);
            ctx.fill();
        });
        // Flux direction arrow, drawn only where direction is meaningful.
        if ((kind === "neumann" || kind === "robin") && (info.nodes || []).length) {
            const node = mesh.nodes[info.nodes[Math.floor(info.nodes.length / 2)]];
            if (node) {
                const screen = worldToScreen(Number(node[0]), Number(node[1]));
                arrow(ctx, screen.x, screen.y, ctx.fillStyle);
            }
        }
    });
    ctx.restore();
}

function arrow(ctx, x, y, colour) {
    ctx.save();
    ctx.strokeStyle = colour;
    ctx.fillStyle = colour;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 14, y); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x + 14, y); ctx.lineTo(x + 9, y - 3.5); ctx.lineTo(x + 9, y + 3.5);
    ctx.closePath(); ctx.fill();
    ctx.restore();
}

function drawTopology(ctx) {
    const topology = state.project.topology || { nodes: [], edges: [], cells: [] };
    const nodeById = {};
    (topology.nodes || []).forEach((node) => { nodeById[node.id] = node; });

    // Cells first, so edges and nodes stay clickable on top of the fill.
    const showFill = state.view === "tissue" || state.view === "clustered" || state.view === "graph";
    if (showFill) {
        (topology.cells || []).forEach((cell, index) => {
            const ring = (cell.nodes || []).map((id) => nodeById[id]).filter(Boolean);
            if (ring.length < 3) return;
            ctx.save();
            const selected = state.selection.cells.includes(cell.id);
            const hue = state.view === "clustered" ? (index * 47) % 360 : 265;
            ctx.fillStyle = selected
                ? "rgba(0, 242, 254, 0.3)"
                : `hsla(${hue}, 60%, 60%, ${state.view === "tissue" ? 0.32 : 0.18})`;
            ctx.strokeStyle = selected ? "#00f2fe" : "rgba(177, 128, 255, 0.75)";
            ctx.lineWidth = selected ? 2.5 : 1.4;
            ctx.beginPath();
            ring.forEach((node, position) => {
                const screen = worldToScreen(Number(node.x), Number(node.y));
                if (position === 0) ctx.moveTo(screen.x, screen.y);
                else ctx.lineTo(screen.x, screen.y);
            });
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            if (cell.centroid) {
                const centre = worldToScreen(Number(cell.centroid[0]), Number(cell.centroid[1]));
                label(ctx, centre.x, centre.y + 4, cell.id, "#f3f4f6", "center");
            }
            ctx.restore();
        });
    }

    (topology.edges || []).forEach((edge) => {
        const a = nodeById[edge.source], b = nodeById[edge.target];
        if (!a || !b) return;
        const from = worldToScreen(Number(a.x), Number(a.y));
        const to = worldToScreen(Number(b.x), Number(b.y));
        const selected = state.selection.edges.includes(edge.id);
        ctx.save();
        ctx.strokeStyle = selected ? "#00f2fe" : "rgba(243, 244, 246, 0.55)";
        ctx.lineWidth = selected ? 3 : 1.6;
        ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke();
        ctx.restore();
    });

    const showNodes = state.view !== "tissue";
    if (showNodes) {
        (topology.nodes || []).forEach((node) => {
            const screen = worldToScreen(Number(node.x), Number(node.y));
            const selected = state.selection.nodes.includes(node.id);
            ctx.save();
            ctx.fillStyle = selected ? "#00f2fe" : "#00ff87";
            ctx.beginPath();
            ctx.arc(screen.x, screen.y, selected ? 6 : 4.5, 0, Math.PI * 2);
            ctx.fill();
            if (state.view === "vertex" || selected) {
                label(ctx, screen.x + 8, screen.y - 8, node.id, "#9ca3af");
            }
            ctx.restore();
        });
    }
}

/** Nearest node within a pixel radius, used for click selection. */
function nodeAt(world, pixelRadius = 12) {
    const topology = state.project.topology || { nodes: [] };
    const tolerance = pixelRadius / state.camera.scale;
    let best = null, bestDistance = Infinity;
    (topology.nodes || []).forEach((node) => {
        const dx = Number(node.x) - world.x, dy = Number(node.y) - world.y;
        const distance = Math.hypot(dx, dy);
        if (distance < tolerance && distance < bestDistance) {
            best = node; bestDistance = distance;
        }
    });
    return best;
}

function cellAt(world) {
    const topology = state.project.topology || { cells: [] };
    const nodeById = {};
    (topology.nodes || []).forEach((node) => { nodeById[node.id] = node; });
    return (topology.cells || []).find((cell) => {
        const ring = (cell.nodes || []).map((id) => nodeById[id]).filter(Boolean);
        if (ring.length < 3) return false;
        let inside = false;
        for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
            const xi = Number(ring[i].x), yi = Number(ring[i].y);
            const xj = Number(ring[j].x), yj = Number(ring[j].y);
            if ((yi > world.y) !== (yj > world.y) &&
                world.x < ((xj - xi) * (world.y - yi)) / (yj - yi) + xi) {
                inside = !inside;
            }
        }
        return inside;
    }) || null;
}


/* --------------------------------------------------------------------------
 * Canvas interaction
 * ----------------------------------------------------------------------- */
let dragNode = null;

function handleCanvasMouseDown(point, event) {
    if (!state.project) return;
    if (state.tool === "node" && state.stage === "topology") {
        addNode(point.world);
        return;
    }
    const node = nodeAt(point.world);
    if (state.tool === "move" && node) { dragNode = node; return; }

    if (state.tool === "edge" && node) {
        const chosen = state.selection.nodes;
        if (chosen.length && chosen[chosen.length - 1] !== node.id) {
            connectNodes(chosen[chosen.length - 1], node.id);
            state.selection.nodes = [node.id];
        } else {
            state.selection.nodes = [node.id];
        }
        // connectNodes() repaints via validateTopology() when an edge was actually
        // added; this only has to show the new node highlight, which is a class
        // toggle rather than a rebuild of ~1800 rows.
        paintSelection();
        return;
    }

    // Select tool: additive with shift, so a loop can be built up for cell creation.
    if (node) {
        if (event.shiftKey) {
            const at = state.selection.nodes.indexOf(node.id);
            if (at >= 0) state.selection.nodes.splice(at, 1);
            else state.selection.nodes.push(node.id);
        } else {
            state.selection.nodes = [node.id];
            state.selection.edges = [];
            state.selection.cells = [];
        }
        paintSelection();
        return;
    }
    const cell = cellAt(point.world);
    if (cell) {
        state.selection.cells = [cell.id];
        state.selection.nodes = [];
        paintSelection();
        return;
    }
    if (!event.shiftKey) {
        state.selection = { nodes: [], edges: [], cells: [] };
        paintSelection();
    }
}

function handleCanvasMouseMove(point) {
    if (dragNode) {
        dragNode.x = point.world.x;
        dragNode.y = point.world.y;
        drawCanvas();
    }
}

function handleCanvasMouseUp() {
    if (dragNode) {
        dragNode = null;
        // Areas and centroids are derived server-side; re-validating refreshes them.
        validateTopology();
    }
}

/* --------------------------------------------------------------------------
 * Stage 1: domain
 * ----------------------------------------------------------------------- */
const DOMAIN_FIELDS = {
    interval: [["x_min", "Start x", 0], ["length", "Length", 100]],
    rectangle: [["x_min", "Min x", 0], ["y_min", "Min y", 0],
                ["width", "Width", 40], ["height", "Height", 25]],
    circle: [["cx", "Centre x", 0], ["cy", "Centre y", 0], ["radius", "Radius", 12]],
    polygon: [],
};

function renderDomainControls() {
    const kind = $("domain-kind").value;
    const host = $("domain-params");
    host.innerHTML = "";
    $("domain-polygon-editor").hidden = kind !== "polygon";

    const current = (state.project && state.project.domain) || {};
    const spec = current[kind] || {};
    DOMAIN_FIELDS[kind].forEach(([key, label, fallback]) => {
        const field = el("label", "wf-field");
        field.appendChild(el("span", null, `${label} (${$("domain-units").value})`));
        const input = el("input");
        input.type = "number";
        input.step = "any";
        input.id = `domain-${key}`;
        input.value = spec[key] !== undefined ? spec[key] : fallback;
        field.appendChild(input);
        host.appendChild(field);
    });
    if (kind === "polygon" && current.polygon && current.polygon.points) {
        $("domain-polygon-points").value =
            current.polygon.points.map((p) => `${p[0]}, ${p[1]}`).join("\n");
    }
}

function readDomainFromControls() {
    const kind = $("domain-kind").value;
    const unitsValue = $("domain-units").value;
    const domain = {
        kind,
        name: "Domain",
        units: unitsValue,
        // Carried on the domain alongside `units` so the time unit survives
        // export/import with the rest of the document rather than being a
        // browser-only setting that a reopened project silently loses.
        time_units: $("domain-time-units").value,
        regions: (state.project && state.project.domain && state.project.domain.regions) || [],
    };

    if (kind === "polygon") {
        const points = [];
        $("domain-polygon-points").value.split(/\r?\n/).forEach((line) => {
            const trimmed = line.trim();
            if (!trimmed) return;
            const parts = trimmed.split(/[,\s]+/).map(Number);
            if (parts.length >= 2 && parts.every(Number.isFinite)) {
                points.push([parts[0], parts[1]]);
            } else {
                // Keep the malformed entry so the backend reports it by position
                // rather than the UI silently dropping a line the user typed.
                points.push([NaN, NaN]);
            }
        });
        domain.polygon = { points };
        domain.boundaries = [{ id: "perimeter", name: "Perimeter",
                               selector: { type: "polygon_perimeter" } }];
    } else {
        const spec = {};
        DOMAIN_FIELDS[kind].forEach(([key]) => {
            spec[key] = Number($(`domain-${key}`).value);
        });
        domain[kind] = spec;
        domain.boundaries = defaultBoundaries(kind);
    }
    return domain;
}

function defaultBoundaries(kind) {
    if (kind === "interval") {
        return [
            { id: "left", name: "Left end", selector: { type: "interval_end", end: "min" } },
            { id: "right", name: "Right end", selector: { type: "interval_end", end: "max" } },
        ];
    }
    if (kind === "rectangle") {
        return ["left", "right", "bottom", "top"].map((side) => ({
            id: side,
            name: `${side[0].toUpperCase()}${side.slice(1)} edge`,
            selector: { type: "rect_side", side },
        }));
    }
    return [{ id: "perimeter", name: "Perimeter", selector: { type: "circle_perimeter" } }];
}

async function applyDomain() {
    const domain = readDomainFromControls();
    let payload;
    try {
        payload = await post("/api/geometry/validate", { domain });
    } catch (error) {
        toast(`Could not validate the domain: ${error.message}`, "error", 8000);
        return false;
    }
    renderIssues("domain-issues", payload, "Domain is valid.");
    if (!payload.valid) {
        setStatus("Domain invalid", "red");
        $("domain-summary").textContent = "Fix the problems below before continuing.";
        return false;
    }

    if (!state.project) {
        try {
            state.project = await post("/api/project/new", { domain });
        } catch (error) {
            toast(`Could not create the project: ${error.message}`, "error", 8000);
            return false;
        }
    } else {
        state.project.domain = domain;
        // The mesh described the previous geometry, so it is no longer meaningful.
        if (state.project.mesh) {
            state.project.mesh = null;
            toast("The mesh was cleared because the domain changed. Regenerate it in stage 3.",
                  "warn", 7000);
        }
    }

    $("domain-summary").textContent = payload.summary || "";
    renderBoundaryChips();
    renderRegionChips();
    updateMeshControlsForDomain();
    refreshUnitLabels();
    // The domain is the other half of the lattice mapping, so a new extent or a new
    // length unit has to move the "1 site = ..." line with it.
    refreshLatticeMapping();
    fitView();
    setStatus("Ready", "green");
    await validateProject();
    return true;
}

function renderBoundaryChips() {
    const host = $("domain-boundaries");
    host.innerHTML = "";
    const boundaries = (state.project.domain.boundaries) || [];
    if (!boundaries.length) {
        host.appendChild(el("li", null, "none"));
        return;
    }
    boundaries.forEach((boundary) => {
        const item = el("li");
        const swatch = el("i", "wf-swatch bc-no_flux");
        const applied = (state.project.conditions || []).filter(
            (c) => c.boundary === boundary.id && c.enabled !== false);
        if (applied.length) {
            swatch.className = `wf-swatch bc-${applied[applied.length - 1].type}`;
        }
        item.appendChild(swatch);
        item.appendChild(el("span", null, `${boundary.name} (${boundary.id})`));
        host.appendChild(item);
    });
}

function renderRegionChips() {
    const host = $("domain-regions");
    host.innerHTML = "";
    const regions = state.project.domain.regions || [];
    if (!regions.length) {
        host.appendChild(el("li", null, "none"));
        return;
    }
    regions.forEach((region, index) => {
        const item = el("li");
        item.appendChild(el("span", null, region.name || region.id));
        const remove = el("button", null, "×");
        remove.title = `Remove ${region.name || region.id}`;
        remove.addEventListener("click", () => {
            regions.splice(index, 1);
            renderRegionChips();
            validateProject();
        });
        item.appendChild(remove);
        host.appendChild(item);
    });
}

function addRegion() {
    const name = $("region-name").value.trim();
    if (!name) { toast("Give the region a name first.", "warn"); return; }
    const regions = state.project.domain.regions || (state.project.domain.regions = []);
    const id = name.toLowerCase().replace(/[^a-z0-9]+/g, "_");
    if (regions.some((r) => r.id === id)) {
        toast(`A region called ${name} already exists.`, "warn");
        return;
    }
    regions.push({ id, name, kind: "all" });
    $("region-name").value = "";
    renderRegionChips();
    validateProject();
}


/* --------------------------------------------------------------------------
 * Stage 2: topology
 *
 * Ids are minted client-side but are only ever provisional: every mutation is
 * re-validated by the backend, which is the authority on whether the topology is
 * consistent. Selection is shared by the canvas and all three tables, so
 * highlighting an entity anywhere highlights it everywhere.
 * ----------------------------------------------------------------------- */
/**
 * Normalised key for an undirected node pair, so `a-b` and `b-a` are one entry.
 * NUL is the separator because it cannot occur in a node id, whereas "-" can.
 */
function edgeKey(a, b) {
    const x = String(a), y = String(b);
    return x < y ? `${x}\u0000${y}` : `${y}\u0000${x}`;
}

/**
 * Pair -> edge index for the current topology, built once and then reused.
 *
 * Every closure and duplicate test used to be a linear `.find()` / `.some()` over
 * topology.edges: proving a 1000-node ring closed against 800 edges scanned the
 * edge list 1000 times to check closure and 1000 times again to collect the ids,
 * about 1.6M comparisons each. The index makes both O(1) per ring edge.
 *
 * The cache is keyed on the ARRAY IDENTITY plus its length, which covers every
 * mutation this file performs: deleteSelected/clearTopology/importProject replace
 * the array (identity changes) and connectNodes pushes (length changes). Nothing
 * rewrites an existing edge's source or target in place.
 */
let edgeIndexCache = { edges: null, size: -1, map: null };

function edgeIndex(topology) {
    const edges = (topology && topology.edges) || [];
    if (edgeIndexCache.edges === edges && edgeIndexCache.size === edges.length) {
        return edgeIndexCache.map;
    }
    const map = new Map();
    for (let i = 0; i < edges.length; i += 1) {
        map.set(edgeKey(edges[i].source, edges[i].target), edges[i]);
    }
    edgeIndexCache = { edges, size: edges.length, map };
    return map;
}

/**
 * Monotonic id counter, kept per prefix on the project document (`next_ids`, which
 * the document already carries).
 *
 * The previous implementation built a Set of every existing id on EVERY insert, so
 * placing n nodes cost O(n^2) -- about 500k id comparisons for the 1000-node
 * target, on the click path. The counter is O(1). It is seeded once from the
 * highest numeric suffix actually present, so an imported project cannot collide,
 * and it is verified against the live collection before being handed out, so a
 * stale counter still cannot mint a duplicate.
 */
const ID_FIELDS = { n: "node", e: "edge", c: "cell" };

/**
 * Seeding is per DOCUMENT, not per insert -- scanning the collection on every call
 * would put the O(n) back and leave placing n nodes at O(n^2). A replaced topology
 * (new project, import, clear) changes object identity and is reseeded; a filtered
 * array is not, and must not be: the counter only ever moves forward, so deleting
 * n5 never lets a later node take the id a stale cell might still reference.
 */
let idCounterState = { topology: null, seeded: {} };

function nextId(collection, prefix) {
    const topology = (state.project && state.project.topology) || null;
    const field = ID_FIELDS[prefix];
    if (!topology || !field) {
        // No document to hold the counter: fall back to a scan of this collection.
        const used = new Set(collection.map((item) => item.id));
        let fallback = collection.length + 1;
        while (used.has(`${prefix}${fallback}`)) fallback += 1;
        return `${prefix}${fallback}`;
    }
    if (!topology.next_ids || typeof topology.next_ids !== "object") {
        topology.next_ids = { node: 1, edge: 1, cell: 1 };
    }
    if (idCounterState.topology !== topology) {
        idCounterState = { topology, seeded: {} };
    }
    const counters = topology.next_ids;
    let n = Number(counters[field]);
    if (!Number.isFinite(n) || n < 1) n = 1;
    if (!idCounterState.seeded[field]) {
        // One scan for this prefix, once, so an imported project whose counter is
        // behind its own ids still cannot mint a duplicate.
        const pattern = new RegExp(`^${prefix}(\\d+)$`);
        for (let i = 0; i < collection.length; i += 1) {
            const match = pattern.exec(String(collection[i].id));
            if (match) {
                const value = Number(match[1]);
                if (value >= n) n = value + 1;
            }
        }
        idCounterState.seeded[field] = true;
    }
    counters[field] = n + 1;
    return `${prefix}${n}`;
}

function addNode(world) {
    const topology = state.project.topology;
    const node = {
        id: nextId(topology.nodes, "n"),
        x: Number(world.x.toFixed(6)),
        y: Number(world.y.toFixed(6)),
        region: "",
        metadata: {},
    };
    topology.nodes.push(node);
    state.selection.nodes = [node.id];
    // validateTopology() rebuilds the tables and redraws the canvas on BOTH its
    // paths, so calling refreshTopologyViews() here as well rebuilt all three
    // tables and redrew the canvas twice for every single node placed.
    validateTopology();
}

function connectNodes(sourceId, targetId) {
    const topology = state.project.topology;
    if (sourceId === targetId) {
        toast("An edge cannot start and end at the same node.", "warn");
        return;
    }
    // O(1) index lookup instead of edges.some() scanning the whole edge list on
    // every insert, which made connecting n edges O(n * E).
    const index = edgeIndex(topology);
    if (index.has(edgeKey(sourceId, targetId))) {
        toast("Those nodes are already connected.", "warn");
        return;
    }
    const edge = {
        id: nextId(topology.edges, "e"),
        source: sourceId, target: targetId, directed: false, boundary: "",
    };
    topology.edges.push(edge);
    // Keep the index live across the push rather than throwing it away, so a run of
    // inserts stays O(1) each instead of rebuilding the map every time.
    index.set(edgeKey(sourceId, targetId), edge);
    edgeIndexCache.size = topology.edges.length;
    validateTopology();
}

async function makeCellFromSelection() {
    const ring = state.selection.nodes.slice();
    if (ring.length < 3) {
        toast("Select at least 3 nodes (shift-click) that form a closed loop.", "warn");
        return;
    }
    const topology = state.project.topology;
    // ONE indexed pass: prove the loop closes AND collect the edge ids together.
    // Previously each ring edge ran a linear .find() over topology.edges to test
    // closure and a second identical .find() to read the id back -- for a 1000-node
    // ring against 800 edges that is ~1.6M comparisons, done twice.
    const index = edgeIndex(topology);
    const edges = [];
    for (let i = 0; i < ring.length; i += 1) {
        const a = ring[i], b = ring[(i + 1) % ring.length];
        const found = index.get(edgeKey(a, b));
        if (!found) {
            toast(`The loop is open: no edge connects ${a} and ${b}.`, "error", 7000);
            return;
        }
        edges.push(found.id);
    }
    topology.cells.push({
        id: nextId(topology.cells, "c"),
        nodes: ring, edges, type: "generic", region: "",
        centroid: [0, 0], area: 0, neighbors: [],
    });
    const payload = await validateTopology();
    if (!payload) {
        // validateTopology returns null when the REQUEST failed, which is not the
        // same as "valid". Falling through kept an unverified -- possibly
        // self-intersecting or degenerate -- cell in the project, where it would
        // poison export and the run later, far from this cause.
        topology.cells.pop();
        toast("Could not verify the cell with the server, so it was not created. "
              + "Check the connection and try again.", "error", 9000);
        refreshTopologyViews();
        return;
    }
    if (!payload.valid) {
        // The backend rejected it (self-intersecting or degenerate); undo rather
        // than leave an invalid cell in the project.
        topology.cells.pop();
        await validateTopology();
        return;
    }
    state.selection.nodes = [];
    // validateTopology() above already rebuilt the tables with the new cell in them;
    // clearing the ring selection is a class change, not a structural one.
    paintSelection();
}

async function findLoops() {
    try {
        const payload = await post("/api/topology/cycles", { topology: state.project.topology });
        if (!payload.count) {
            toast("No closed loops found. Connect nodes into a ring first.", "warn");
            return;
        }
        const preview = payload.cycles.slice(0, 6).map((cycle) => cycle.join("→")).join("  ·  ");
        toast(`Found ${payload.count} loop(s)${payload.capped ? " (capped)" : ""}: ${preview}`,
              "info", 9000);
        state.selection.nodes = payload.cycles[0].slice();
        // Only the selection changed, so a full three-table rebuild was wasted work.
        paintSelection();
    } catch (error) {
        toast(`Loop detection failed: ${error.message}`, "error");
    }
}

function deleteSelected() {
    const topology = state.project.topology;
    const nodes = new Set(state.selection.nodes);
    const cells = new Set(state.selection.cells);
    if (!nodes.size && !cells.size) {
        toast("Nothing selected.", "warn");
        return;
    }
    if (cells.size) {
        topology.cells = topology.cells.filter((cell) => !cells.has(cell.id));
    }
    if (nodes.size) {
        topology.nodes = topology.nodes.filter((node) => !nodes.has(node.id));
        const deadEdges = new Set();
        topology.edges = topology.edges.filter((edge) => {
            const dead = nodes.has(edge.source) || nodes.has(edge.target);
            if (dead) deadEdges.add(edge.id);
            return !dead;
        });
        // Cascade: a cell missing a vertex or an edge is degenerate, not repairable.
        topology.cells = topology.cells.filter((cell) =>
            !(cell.nodes || []).some((id) => nodes.has(id)) &&
            !(cell.edges || []).some((id) => deadEdges.has(id)));
    }
    state.selection = { nodes: [], edges: [], cells: [] };
    // validateTopology() repaints on both its paths; the extra call here rebuilt
    // all three tables and redrew the canvas a second time for one delete.
    validateTopology();
}

function clearTopology() {
    state.project.topology = { nodes: [], edges: [], cells: [], next_ids: { node: 1, edge: 1, cell: 1 } };
    state.selection = { nodes: [], edges: [], cells: [] };
    validateTopology();
}

async function validateTopology() {
    try {
        const payload = await post("/api/topology/validate", {
            topology: state.project.topology,
            domain: state.project.domain,
        });
        renderIssues("topology-issues", payload,
                     "Topology is consistent with the domain.");
        renderTopologyStats(payload.stats);
        refreshTopologyViews();
        paintStageBadges(state.validation);
        return payload;
    } catch (error) {
        // The local mutation has already happened, so the tables must still be
        // rebuilt from it when the request fails -- otherwise dropping the callers'
        // eager refreshTopologyViews() would leave deleted rows on screen. This is
        // now the ONE place a topology mutation repaints, on both paths.
        refreshTopologyViews();
        toast(`Topology validation failed: ${error.message}`, "error");
        return null;
    }
}

function renderTopologyStats(stats) {
    if (!stats) return;
    const host = $("topo-stats");
    host.innerHTML = "";
    const lines = [
        `nodes <b>${stats.node_count}</b>   edges <b>${stats.edge_count}</b>   cells <b>${stats.cell_count}</b>`,
    ];
    if (stats.edge_length) {
        lines.push(`edge length  min <b>${fmt(stats.edge_length.min)}</b>  ` +
                   `max <b>${fmt(stats.edge_length.max)}</b>  mean <b>${fmt(stats.edge_length.mean)}</b> ${units()}`);
    }
    if (stats.cell_area) {
        lines.push(`cell area  min <b>${fmt(stats.cell_area.min)}</b>  ` +
                   `max <b>${fmt(stats.cell_area.max)}</b>  total <b>${fmt(stats.cell_area.total)}</b> ${units()}²`);
    }
    host.innerHTML = lines.join("<br>");
}

function refreshTopologyViews() {
    const topology = state.project.topology || { nodes: [], edges: [], cells: [] };
    $("node-count").textContent = topology.nodes.length;
    $("edge-count").textContent = topology.edges.length;
    $("cell-count").textContent = topology.cells.length;

    buildTable($("node-table"), ["id", "x", "y", "region"],
        topology.nodes.map((node) => ({
            key: node.id, selected: state.selection.nodes.includes(node.id),
            cells: [node.id, fmt(node.x), fmt(node.y), node.region || "—"],
            onClick: (additive) => selectEntity("nodes", node.id, additive),
        })), "No nodes yet — pick “Add node” and click the canvas.");

    buildTable($("edge-table"), ["id", "from", "to", "boundary"],
        topology.edges.map((edge) => ({
            key: edge.id, selected: state.selection.edges.includes(edge.id),
            cells: [edge.id, edge.source, edge.target, edge.boundary || "—"],
            onClick: (additive) => selectEntity("edges", edge.id, additive),
        })), "No edges yet — use “Connect”.");

    buildTable($("cell-table"), ["id", "type", "area", "nodes", "neighbours"],
        topology.cells.map((cell) => ({
            key: cell.id, selected: state.selection.cells.includes(cell.id),
            cells: [cell.id, cell.type || "generic", fmt(cell.area),
                    (cell.nodes || []).length, (cell.neighbors || []).length],
            onClick: (additive) => selectEntity("cells", cell.id, additive),
        })), "No cells yet — select a closed loop and make one.");

    drawCanvas();
}

/**
 * Shared table builder.
 *
 * Two things used to make this the hot spot of the whole stage. It cleared the
 * table with `innerHTML = ""` and then inserted every row into a tbody that was
 * already in the document, so the browser had a chance to reflow on each append;
 * and it attached a fresh click listener to every row. At the stated target of
 * 1000 nodes / 800 edges that is ~1800 rows and ~1800 listeners recreated on every
 * interaction, selection included.
 *
 * Now the tbody is built DETACHED and swapped in with a single replaceWith(), so
 * the document is touched once per refresh, and there is exactly ONE delegated
 * click listener per table, installed on first use and keyed off `tr[data-key]`.
 * Row behaviour lives in a Map beside the table instead of in N closures.
 */
const tableHandlers = new WeakMap();

function buildTable(table, headers, rows, emptyMessage) {
    if (!table) return;

    // The header is static per table: rebuild it only when the columns change.
    const signature = headers.join("\u0000");
    if (!table.tHead || table.dataset.headSig !== signature) {
        if (table.tHead) table.tHead.remove();
        const head = table.createTHead().insertRow();
        headers.forEach((header) => {
            const cell = document.createElement("th");
            cell.textContent = header;
            head.appendChild(cell);
        });
        table.dataset.headSig = signature;
    }

    // One listener for the table's whole lifetime. It resolves the row at click
    // time, so swapping the tbody never leaves a dangling or duplicated binding.
    let handlers = tableHandlers.get(table);
    if (!handlers) {
        handlers = new Map();
        tableHandlers.set(table, handlers);
        table.addEventListener("click", (event) => {
            const row = event.target && event.target.closest
                ? event.target.closest("tr[data-key]")
                : null;
            if (!row || !table.contains(row)) return;
            const handler = handlers.get(row.dataset.key);
            if (handler) handler(event.shiftKey);
        });
    }
    handlers.clear();

    const body = document.createElement("tbody");
    if (!rows.length) {
        const row = body.insertRow();
        const cell = row.insertCell();
        cell.colSpan = headers.length;
        cell.className = "wf-empty";
        cell.textContent = emptyMessage;
    } else {
        rows.forEach((spec, index) => {
            const row = body.insertRow();
            // Delegation keys off data-key, so a missing or duplicated key would
            // silently wire two rows to one handler. Fall back to the position, and
            // disambiguate a genuine collision rather than let a click fire the
            // wrong row's action. Unique ids (node/edge/cell/run/condition) are
            // untouched, which is what paintSelection matches against.
            let key = spec.key === undefined || spec.key === null
                ? `#${index}` : String(spec.key);
            if (handlers.has(key)) key = `${key}#${index}`;
            row.dataset.key = key;
            if (spec.selected) row.classList.add("selected");
            spec.cells.forEach((value, column) => {
                const cell = row.insertCell();
                cell.textContent = value;
                if (column === 0) cell.classList.add("mono");
            });
            if (spec.onClick) handlers.set(key, spec.onClick);
        });
    }

    const existing = table.tBodies[0];
    if (existing) existing.replaceWith(body);
    else table.appendChild(body);
}

/**
 * Selection-only repaint: the rows on screen are already correct, so only their
 * `selected` class and the canvas need to change.
 *
 * Clicking a node used to run refreshTopologyViews(), which rebuilt all three
 * tables from scratch -- ~1800 rows at the target -- to change a highlight. This
 * touches one class per existing row and redraws the canvas, and allocates nothing.
 */
function paintSelection() {
    const tables = [
        ["node-table", state.selection.nodes],
        ["edge-table", state.selection.edges],
        ["cell-table", state.selection.cells],
    ];
    tables.forEach(([id, ids]) => {
        const table = $(id);
        const body = table && table.tBodies[0];
        if (!body) return;
        const chosen = new Set((ids || []).map(String));
        const rows = body.rows;
        for (let i = 0; i < rows.length; i += 1) {
            const key = rows[i].dataset.key;
            if (key === undefined) continue;
            rows[i].classList.toggle("selected", chosen.has(key));
        }
    });
    drawCanvas();
}

function selectEntity(kind, id, additive) {
    if (!additive) state.selection = { nodes: [], edges: [], cells: [] };
    const list = state.selection[kind];
    const at = list.indexOf(id);
    if (at >= 0) list.splice(at, 1);
    else list.push(id);
    paintSelection();
}

async function exportTopology() {
    try {
        const payload = await post("/api/topology/export", { topology: state.project.topology });
        Object.entries(payload.files).forEach(([name, contents]) => {
            download(name, contents, "text/csv");
        });
        toast("Exported nodes, edges and cells as CSV.", "success");
    } catch (error) {
        toast(`Export failed: ${error.message}`, "error");
    }
}


/* --------------------------------------------------------------------------
 * Stage 3: mesh
 * ----------------------------------------------------------------------- */
function updateMeshControlsForDomain() {
    const is1d = state.project && state.project.domain.kind === "interval";
    $("mesh-1d-controls").hidden = !is1d;
    $("mesh-2d-controls").hidden = is1d;
    const structured = state.project && state.project.domain.kind === "rectangle";
    const kindSelect = $("mesh-kind");
    // Only a rectangle can carry a structured grid; anything else must be Delaunay.
    Array.from(kindSelect.options).forEach((option) => {
        option.disabled = option.value === "structured" && !structured;
    });
    if (!structured) kindSelect.value = "unstructured";
}

function meshSettings() {
    const domain = state.project.domain;
    if (domain.kind === "interval") {
        return { element_count: Number($("mesh-elements").value) || 40 };
    }
    const settings = { kind: $("mesh-kind").value };
    const spacing = Number($("mesh-spacing").value);
    if (Number.isFinite(spacing) && spacing > 0) {
        settings.target_spacing = spacing;
    } else {
        settings.rows = Number($("mesh-rows").value) || 10;
        settings.cols = Number($("mesh-cols").value) || 10;
    }
    return settings;
}

async function generateMesh() {
    setStatus("Meshing…", "yellow");
    try {
        const payload = await post("/api/mesh/generate", {
            domain: state.project.domain,
            settings: meshSettings(),
        });
        state.project.mesh = payload.mesh;
        state.project.mesh_settings = meshSettings();
        renderIssues("mesh-issues", payload.validation, "Mesh is valid.");
        renderMeshStats(payload.mesh);
        state.view = "mesh";
        document.querySelectorAll(".wf-view").forEach((button) => {
            button.classList.toggle("active", button.dataset.view === "mesh");
        });
        drawCanvas();
        setStatus("Ready", "green");
        await validateProject();
    } catch (error) {
        // A refused mesh carries an actionable message (spacing too fine, wrong
        // domain kind, SciPy missing); show it rather than a generic failure.
        renderIssues("mesh-issues", { issues: [{ severity: "error", message: error.message }] });
        setStatus("Mesh failed", "red");
        toast(error.message, "error", 9000);
    }
}

function renderMeshStats(mesh) {
    const host = $("mesh-stats");
    if (!mesh) { host.textContent = "No mesh generated yet."; return; }
    const stats = mesh.stats || {};
    // Every string below can originate in an IMPORTED project file, which is this
    // app's untrusted entry point -- a crafted project could otherwise inject markup
    // into the page with access to the same API. fmt()'d numbers are safe as-is;
    // strings are not.
    const lines = [
        `kind <b>${escapeHtml(mesh.kind)}</b> (${escapeHtml(stats.element_kind || "—")})`,
        `nodes <b>${stats.node_count}</b>  edges <b>${stats.edge_count}</b>  elements <b>${stats.element_count}</b>`,
    ];
    if (stats.element_size) {
        // A length in 1D, an area in 2D. Quoting the wrong one is a squared error,
        // so the unit comes from the mesh's own dimension rather than a guess.
        const sizeUnit = elementSizeUnit(mesh);
        lines.push(`element size  min <b>${fmt(stats.element_size.min)}</b>  ` +
                   `max <b>${fmt(stats.element_size.max)}</b>  ` +
                   `mean <b>${fmt(stats.element_size.mean)}</b> ${escapeHtml(sizeUnit)}`);
    }
    if (stats.min_angle_deg !== undefined) {
        lines.push(`min angle <b>${fmt(stats.min_angle_deg, 1)}°</b>  ` +
                   `slivers <b>${stats.sliver_elements || 0}</b>`);
    }
    if (stats.coverage_fraction !== undefined) {
        lines.push(`domain coverage <b>${fmt(stats.coverage_fraction * 100, 1)}%</b>`);
    }
    lines.push(`degenerate elements <b>${stats.degenerate_elements || 0}</b>`);
    lines.push(`estimated cost <b>${escapeHtml(String(stats.estimated_cost_class))}</b> ` +
               `(${escapeHtml(String(stats.unknowns_per_field))} unknowns per field)`);
    host.innerHTML = lines.join("<br>");

    const boundaryHost = $("mesh-boundary-list");
    boundaryHost.innerHTML = "";
    Object.entries(mesh.boundaries || {}).forEach(([id, info]) => {
        const item = el("li");
        item.appendChild(el("span", null, `${info.name || id}: ${(info.nodes || []).length} nodes`));
        boundaryHost.appendChild(item);
    });
}

function clearMesh() {
    state.project.mesh = null;
    renderMeshStats(null);
    $("mesh-boundary-list").innerHTML = "";
    renderIssues("mesh-issues", null);
    drawCanvas();
    validateProject();
}

/* --------------------------------------------------------------------------
 * Stage 4: PDE model
 * ----------------------------------------------------------------------- */
async function loadPresetList() {
    try {
        const payload = await api("/api/model/presets");
        const select = $("model-preset");
        payload.presets.forEach((preset) => {
            const option = el("option", null, preset.label || preset.name);
            option.value = preset.name;
            option.title = preset.description || "";
            select.appendChild(option);
        });
    } catch (error) {
        toast(`Could not load model presets: ${error.message}`, "warn");
    }
}

async function applyPreset(name) {
    if (!name) return;
    try {
        const preset = await api(`/api/model/preset/${encodeURIComponent(name)}`);
        state.project.model = { parameters: preset.parameters || {}, fields: preset.fields || [] };
        renderModelControls();
        await validateModel();
        toast(`Loaded preset: ${preset.label || name}`, "success");
    } catch (error) {
        toast(`Could not load that preset: ${error.message}`, "error");
    }
}

function renderModelControls() {
    const model = state.project.model || { parameters: {}, fields: [] };

    const paramHost = $("model-parameters");
    paramHost.innerHTML = "";
    Object.entries(model.parameters || {}).forEach(([name, value]) => {
        const field = el("label", "wf-field");
        const head = el("span");
        head.appendChild(document.createTextNode(name));
        const remove = el("button", null, "×");
        remove.type = "button";
        remove.title = `Remove ${name}`;
        remove.style.cssText = "background:none;border:none;color:#6b7280;cursor:pointer;float:right";
        remove.addEventListener("click", () => {
            delete model.parameters[name];
            renderModelControls();
            validateModel();
        });
        head.appendChild(remove);
        field.appendChild(head);
        const input = el("input");
        input.type = "number";
        input.step = "any";
        input.value = value;
        input.addEventListener("change", () => {
            model.parameters[name] = Number(input.value);
            validateModel();
        });
        field.appendChild(input);
        paramHost.appendChild(field);
    });
    // A model parameter is a user-named scalar whose dimension the app has no way
    // to know -- a rate, a half-saturation constant and a dimensionless exponent
    // all look identical here. So say that plainly, quoting the two base units the
    // rest of the stage is derived from, rather than inventing a unit per box.
    if (Object.keys(model.parameters || {}).length) {
        const note = el("p", "wf-hint",
            `Parameter units are not tracked. Keep them consistent with the ` +
            `${units() || "length"} / ${timeUnits()} system chosen in stage 1.`);
        note.style.gridColumn = "1 / -1";
        paramHost.appendChild(note);
    }

    const fieldHost = $("model-fields");
    fieldHost.innerHTML = "";
    (model.fields || []).forEach((field, index) => {
        fieldHost.appendChild(buildFieldCard(field, index));
    });
    populateConditionSelectors();
}

function buildFieldCard(field, index) {
    const card = el("div", "wf-field-card");
    const head = el("div", "wf-field-card-head");
    head.appendChild(el("strong", null, field.name || `field ${index + 1}`));
    const remove = el("button", "btn btn-secondary btn-sm", "Remove");
    remove.addEventListener("click", () => {
        state.project.model.fields.splice(index, 1);
        renderModelControls();
        validateModel();
    });
    head.appendChild(remove);
    card.appendChild(head);

    const grid = el("div", "wf-param-grid");
    // Labels that depend on the field's OWN declared units, kept as closures so a
    // change to the Units box relabels them IN PLACE. Re-rendering the card
    // instead would destroy focus mid-Tab, for the reason the `name` branch below
    // spells out.
    const relabellers = [];
    const textInputs = [
        ["name", () => "Name"],
        ["units", () => "Units"],
        // L²/T and [u]/T are derived from the two stored base units and the field's
        // own units -- never asked for as a separate box.
        ["diffusion", () => `Diffusion D${unitSuffix(diffusionUnit())}`],
        ["initial", () => `Initial value${unitSuffix(fieldUnit(field))}`],
        ["reaction", () => `Reaction R(u,x,t)${unitSuffix(rateUnit(field))}`],
        ["source", () => `Source S(x,t)${unitSuffix(rateUnit(field))}`],
    ];
    textInputs.forEach(([key, label]) => {
        const wrapper = el("label", "wf-field");
        const caption = el("span", null, label());
        wrapper.appendChild(caption);
        relabellers.push(() => { caption.textContent = label(); });
        const input = el("input");
        input.type = "text";
        input.value = field[key] !== undefined ? field[key] : "";
        input.addEventListener("change", () => {
            field[key] = input.value;
            // Deliberately NOT calling renderModelControls() here. A `change` event
            // fires when focus LEAVES the input, so rebuilding the card destroyed it
            // mid-Tab: focus fell to <body> and the next Tab restarted from the page
            // header, making the six boxes in a field card impossible to fill in by
            // keyboard. Only the heading actually depends on another input's value.
            if (key === "name") {
                const heading = card.querySelector("strong");
                if (heading) heading.textContent = input.value || `field ${index + 1}`;
            }
            if (key === "units") {
                // Initial value, reaction and source are all quoted against these
                // units, so they must not keep advertising the previous one.
                relabellers.forEach((relabel) => relabel());
                renderInitialConditions();
                renderConditionParams();
                renderConditionTable();
            }
            validateModel();
        });
        wrapper.appendChild(input);
        grid.appendChild(wrapper);
    });
    card.appendChild(grid);

    // Time controls and advection are progressively disclosed: most runs never
    // touch them, and an always-visible wall of numbers hides the essentials.
    const details = el("details", "wf-details");
    details.appendChild(el("summary", null, "Advanced: time window and advection"));
    const advanced = el("div", "wf-param-grid");
    const timeSuffix = unitSuffix(timeUnits());
    [["t_start", `Start time${timeSuffix}`], ["t_end", `End time${timeSuffix}`],
     ["output_interval", `Output interval${timeSuffix}`]]
        .forEach(([key, label]) => {
            const wrapper = el("label", "wf-field");
            wrapper.appendChild(el("span", null, label));
            const input = el("input");
            input.type = "number";
            input.step = "any";
            input.value = field[key] !== undefined ? field[key] : 0;
            input.addEventListener("change", () => {
                field[key] = Number(input.value);
                validateModel();
            });
            wrapper.appendChild(input);
            advanced.appendChild(wrapper);
        });
    const velocitySuffix = unitSuffix(velocityUnit());
    [["vx", `Advection vx${velocitySuffix}`], ["vy", `Advection vy${velocitySuffix}`]]
        .forEach(([key, label]) => {
        const wrapper = el("label", "wf-field");
        wrapper.appendChild(el("span", null, label));
        const input = el("input");
        input.type = "text";
        field.advection = field.advection || { vx: 0, vy: 0 };
        input.value = field.advection[key] !== undefined ? field.advection[key] : 0;
        input.addEventListener("change", () => {
            field.advection[key] = input.value;
            validateModel();
        });
        wrapper.appendChild(input);
        advanced.appendChild(wrapper);
    });
    details.appendChild(advanced);
    card.appendChild(details);
    return card;
}

function addField() {
    const model = state.project.model || (state.project.model = { parameters: {}, fields: [] });
    const used = new Set((model.fields || []).map((f) => f.name));
    let name = "u";
    let suffix = 2;
    while (used.has(name)) { name = `u${suffix}`; suffix += 1; }
    model.fields.push({
        name, units: "mM", diffusion: 1.0, initial: "0.0", reaction: "0", source: "0",
        advection: { vx: 0, vy: 0 }, t_start: 0, t_end: 1, output_interval: 0.1,
    });
    renderModelControls();
    validateModel();
}

async function validateModel() {
    try {
        const payload = await post("/api/model/validate", {
            model: state.project.model,
            domain: state.project.domain,
        });
        renderIssues("model-issues", payload, "Model is valid.");
        renderModelSummary(payload.fields || []);
        populateConditionSelectors();
        // The MCS-to-model-time half of the lattice mapping reads the field time
        // window, so editing that window has to move the mapping line.
        refreshLatticeMapping();
        await validateProject();
        return payload;
    } catch (error) {
        toast(`Model validation failed: ${error.message}`, "error");
        return null;
    }
}

function renderModelSummary(fields) {
    const latexHost = $("model-latex");
    const plainHost = $("model-plain");
    latexHost.innerHTML = "";
    plainHost.innerHTML = "";
    if (!fields.length) {
        latexHost.appendChild(el("div", "wf-empty-state", "Define a field to see its equation."));
        return;
    }
    fields.forEach((entry) => {
        const block = el("div");
        block.style.marginBottom = "10px";
        try {
            // KaTeX throwOnError=false: a malformed expression is already reported as
            // a validation issue, and a thrown renderer should not blank the panel.
            katex.render(entry.latex, block, { throwOnError: false, displayMode: true });
        } catch (error) {
            block.textContent = entry.latex;
        }
        latexHost.appendChild(block);
        plainHost.appendChild(el("p", null, entry.summary));
    });
}

async function solve1DNow() {
    if (state.project.domain.kind !== "interval") {
        toast("Direct 1D solving needs a 1D interval domain. Use an approach run for 2D.", "warn", 7000);
        return;
    }
    setStatus("Solving 1D…", "yellow");
    try {
        const payload = await post("/api/pde/solve1d", {
            domain: state.project.domain,
            model: state.project.model,
            conditions: state.project.conditions || [],
            mesh_settings: state.project.mesh_settings || { element_count: 40 },
        });
        const stability = payload.stability || {};
        const last = (payload.u && payload.u[payload.u.length - 1]) || [];
        // Single pass instead of Math.min(...last): the final profile has one entry
        // per mesh node, so a fine 1D mesh spread past the ~65k argument limit and
        // threw RangeError here -- which blanked the whole readout, including the
        // stability numbers that were already computed correctly.
        const range = extent(last);
        const mass = payload.mass || [];
        const time = escapeHtml(timeUnits());
        $("model-solve-readout").innerHTML = [
            `field <b>${escapeHtml(payload.field)}</b> ${payload.units ? `(${escapeHtml(payload.units)})` : ""}`,
            `frames <b>${payload.t.length}</b>  end time <b>${fmt(stability.end_time)}</b> ${time}`,
            `dt <b>${fmt(stability.dt)}</b> ${time} of max <b>${fmt(stability.max_stable_dt)}</b> ${time}`,
            `diffusion number <b>${fmt(stability.diffusion_number)}</b>` +
            (stability.courant_number ? `  Courant <b>${fmt(stability.courant_number)}</b>` : ""),
            stability.numerical_diffusion
                ? `numerical diffusion from upwinding <b>${fmt(stability.numerical_diffusion)}</b>` : "",
            range.count
                ? `final range <b>${fmt(range.low)}</b> … <b>${fmt(range.high)}</b>`
                : `final range <b>—</b> (the solver returned no finite values for the last frame)`,
            mass.length
                ? `mass first <b>${fmt(mass[0])}</b> last <b>${fmt(mass[mass.length - 1])}</b>`
                : `mass <b>—</b> (not reported)`,
            stability.diverged_at ? `<span style="color:#ff007f">diverged at t=${fmt(stability.diverged_at)}</span>` : "",
        ].filter(Boolean).join("<br>");
        setStatus("Ready", "green");
        toast("1D solve finished.", "success");
    } catch (error) {
        $("model-solve-readout").innerHTML =
            `<span style="color:#ff007f">${escapeHtml(error.message)}</span>`;
        setStatus("Solve failed", "red");
    }
}


/* --------------------------------------------------------------------------
 * Stage 5: boundary conditions
 * ----------------------------------------------------------------------- */
/**
 * Labels are FUNCTIONS of the field the condition is being written against,
 * because a boundary value is in that field's own units -- the same pattern the
 * initial-conditions panel already uses to render "u(x, 0) [mM]".
 *
 * Derivations, from the two stored base units L and T and the field's units [u]:
 *   Dirichlet value   [u]           it IS a value of u
 *   Neumann flux      [u]·L/T       D·∂u/∂n = (L²/T)·([u]/L)
 *   Robin h           L/T           -D·∂u/∂n = h·(u - u∞), so h = (L²/T)/L
 *   Robin u∞          [u]           it IS a value of u
 * A partner boundary is a name, not a measurement, so it carries no unit.
 */
const BC_PARAMS = {
    dirichlet: [["value", (field) => `Fixed value${unitSuffix(fieldUnit(field))}`, 0]],
    neumann: [["flux", (field) => {
        const unit = fluxUnit(field);
        // The sign convention has to survive an underivable unit, so it lives in
        // the same parenthesis rather than depending on one being there.
        return unit ? `Flux (${unit}, negative = inward)` : "Flux (negative = inward)";
    }, 0]],
    no_flux: [],
    robin: [["transfer_coefficient", () => `Transfer h${unitSuffix(velocityUnit())}`, 1],
            ["ambient_value", (field) => `Ambient u∞${unitSuffix(fieldUnit(field))}`, 0]],
    periodic: [["partner", () => "Partner boundary", ""]],
};

function populateConditionSelectors() {
    if (!state.project) return;
    const fieldSelect = $("bc-field");
    const previousField = fieldSelect.value;
    fieldSelect.innerHTML = "";
    ((state.project.model && state.project.model.fields) || []).forEach((field) => {
        if (!field.name) return;
        const option = el("option", null, field.name);
        option.value = field.name;
        fieldSelect.appendChild(option);
    });
    if (previousField) fieldSelect.value = previousField;

    const boundarySelect = $("bc-boundary");
    const previousBoundary = boundarySelect.value;
    boundarySelect.innerHTML = "";
    ((state.project.domain && state.project.domain.boundaries) || []).forEach((boundary) => {
        const option = el("option", null, `${boundary.name} (${boundary.id})`);
        option.value = boundary.id;
        boundarySelect.appendChild(option);
    });
    if (previousBoundary) boundarySelect.value = previousBoundary;
    renderConditionParams();
}

function renderConditionParams() {
    const host = $("bc-params");
    host.innerHTML = "";
    const kind = $("bc-type").value;
    // The value boxes are quoted in the SELECTED field's units, so the label is
    // resolved against that field every time this panel is rebuilt.
    const field = conditionField();
    (BC_PARAMS[kind] || []).forEach(([key, label, fallback]) => {
        const wrapper = el("label", "wf-field");
        wrapper.appendChild(el("span", null,
            typeof label === "function" ? label(field) : label));
        if (key === "partner") {
            const select = el("select");
            select.id = `bc-param-${key}`;
            ((state.project.domain.boundaries) || []).forEach((boundary) => {
                const option = el("option", null, boundary.id);
                option.value = boundary.id;
                select.appendChild(option);
            });
            wrapper.appendChild(select);
        } else {
            const input = el("input");
            input.type = "number";
            input.step = "any";
            input.id = `bc-param-${key}`;
            input.value = fallback;
            wrapper.appendChild(input);
        }
        host.appendChild(wrapper);
    });
}

async function addCondition() {
    const kind = $("bc-type").value;
    const field = $("bc-field").value;
    const boundary = $("bc-boundary").value;
    if (!field) { toast("Define a field in stage 4 first.", "warn"); return; }
    if (!boundary) { toast("This domain has no named boundaries.", "warn"); return; }

    const condition = {
        id: `bc_${field}_${boundary}_${kind}_${Date.now().toString(36)}`,
        type: kind, field, boundary, enabled: true,
    };
    (BC_PARAMS[kind] || []).forEach(([key]) => {
        const input = $(`bc-param-${key}`);
        if (!input) return;
        condition[key] = key === "partner" ? input.value : Number(input.value);
    });

    const conditions = state.project.conditions || (state.project.conditions = []);
    conditions.push(condition);

    // Periodicity is only meaningful declared on both sides, so add the mirror
    // automatically rather than letting validation reject a half-declaration.
    if (kind === "periodic" && condition.partner) {
        const mirrored = conditions.some((c) =>
            c.type === "periodic" && c.field === field &&
            c.boundary === condition.partner && c.partner === boundary);
        if (!mirrored) {
            conditions.push({
                id: `${condition.id}_mirror`, type: "periodic", field,
                boundary: condition.partner, partner: boundary, enabled: true,
            });
        }
    }
    await validateConditions();
}

async function validateConditions() {
    const fields = ((state.project.model && state.project.model.fields) || [])
        .map((field) => field.name).filter(Boolean);
    try {
        const payload = await post("/api/conditions/validate", {
            conditions: state.project.conditions || [],
            domain: state.project.domain,
            fields,
        });
        renderIssues("conditions-issues", payload, "Conditions are consistent.");
        renderConditionTable();
        const list = $("bc-description");
        list.innerHTML = "";
        (payload.description || []).forEach((line) => list.appendChild(el("li", null, line)));
        renderBoundaryChips();
        drawCanvas();
        await validateProject();
        return payload;
    } catch (error) {
        toast(`Condition validation failed: ${error.message}`, "error");
        return null;
    }
}

function renderConditionTable() {
    const conditions = state.project.conditions || [];
    buildTable($("bc-table"), ["field", "boundary", "type", "detail", ""],
        conditions.map((condition, index) => ({
            key: condition.id,
            selected: false,
            cells: [
                condition.field, condition.boundary, condition.type,
                describeCondition(condition),
                condition.enabled === false ? "disabled" : "on",
            ],
            onClick: (shift) => toggleCondition(index, shift),
        })), "No conditions assigned — every boundary is treated as no-flux.");
    renderInitialConditions();
}

function describeCondition(condition) {
    // The table reads assigned numbers back at the researcher, so it quotes the
    // same derived units the input boxes were labelled with. Text lands in a
    // table cell via textContent (see buildTable), and every unit part is
    // sanitised at source, so no escaping is owed here.
    const field = conditionField(condition.field);
    const value = unitSuffix(fieldUnit(field));
    switch (condition.type) {
        case "dirichlet": return `u = ${fmt(condition.value)}${value}`;
        case "neumann": return `flux = ${fmt(condition.flux)}${unitSuffix(fluxUnit(field))}`;
        case "no_flux": return "sealed";
        case "robin": return `h = ${fmt(condition.transfer_coefficient)}` +
                             `${unitSuffix(velocityUnit())}, u∞ = ${fmt(condition.ambient_value)}${value}`;
        case "periodic": return `pairs with ${condition.partner}`;
        default: return "—";
    }
}

/**
 * Clicking a row toggles enabled; shift-click removes it.
 * The removal branch used to be missing while this comment claimed it existed --
 * buildTable already forwarded event.shiftKey, the handler just ignored it.
 */
function toggleCondition(index, remove) {
    const conditions = state.project.conditions || [];
    const condition = conditions[index];
    if (!condition) return;
    if (remove) {
        conditions.splice(index, 1);
        toast(`Removed the ${condition.type} condition on "${condition.boundary}".`, "info");
        validateConditions();
        return;
    }
    condition.enabled = condition.enabled === false;
    validateConditions();
}

/**
 * Initial conditions for every field, editable on the stage whose title promises
 * them. These bind to the SAME field objects stage 4 edits, so there is one source
 * of truth and no risk of two divergent copies of the initial state.
 */
function renderInitialConditions() {
    const host = $("ic-list");
    if (!host) return;
    host.innerHTML = "";
    const fields = ((state.project || {}).model || {}).fields || [];
    if (!fields.length) {
        host.appendChild(el("div", "wf-empty-state",
            "No fields defined yet — add one in stage 4."));
        return;
    }
    fields.forEach((field) => {
        const wrapper = el("label", "wf-field");
        const name = field.name || "field";
        const unit = fieldUnit(field);
        wrapper.appendChild(el("span", null, `${name}(x, 0)${unit ? ` [${unit}]` : ""}`));
        const input = el("input");
        input.type = "text";
        input.value = field.initial !== undefined ? field.initial : "";
        input.placeholder = "e.g. 0.5  or  exp(-((x-5)^2))";
        input.addEventListener("change", () => {
            field.initial = input.value;
            // Re-validate through the model path: an initial expression is parsed by
            // the same sandboxed parser as the reaction term, so a bad one must be
            // reported here rather than at run time. Note we do NOT re-render this
            // list -- `change` fires on focus loss, so rebuilding it would destroy
            // focus mid-Tab exactly as it did in the model field cards.
            validateModel();
        });
        wrapper.appendChild(input);
        host.appendChild(wrapper);
    });
}

/* --------------------------------------------------------------------------
 * Stage 6/7: approach selection and configuration
 * ----------------------------------------------------------------------- */
const APPROACH_DEFAULTS = {
    abm: {
        grid: { width: 60, height: 60 }, temperature: 10, num_mcs: 200, save_every: 10,
        seed: 1,
        cell_types: [{
            type_id: 1, name: "CellA", target_volume: 25, lambda_volume: 2.0,
            target_surface: 20, lambda_surface: 0.5, max_volume_before_division: 400,
            growth_rate: 0, death_probability: 0, color: [0, 200, 255], initial_count: 8,
        }],
    },
    cc3d: {
        lattice: { x: 60, y: 60, z: 1 }, steps: 500, temperature: 10, neighbor_order: 2,
        cell_types: [{ type_id: 1, name: "CellA" }],
        contact_energies: [{ type1: "CellA", type2: "Medium", energy: 12 }],
        volume_constraint: { target_volume: 25, lambda_volume: 2 },
    },
    mpc: {
        controlled_input: "u", measured_output: "y",
        target: 1.0, duration: 30.0, control_interval: 0.5,
        prediction_horizon: 12, control_horizon: 4,
        input_min: -2, input_max: 2, input_rate_limit: 0.5,
        // null means "no constraint"; the editor renders these as blank-able numbers.
        output_min: null, output_max: null,
        weight_tracking: 1.0, weight_effort: 0.05, weight_rate: 0.05,
        tolerance: 1e-6,
        // Without these the results workspace plotted unitless axes.
        output_units: "a.u.", input_units: "a.u.",
        model: { kind: "first_order", tau: 5, gain: 1 },
        plant: { kind: "first_order", tau: 5, gain: 1 },
    },
};

let approachCapabilities = [];

/* --------------------------------------------------------------------------
 * Approach configuration units
 *
 * buildConfigEditor renders whatever keys the config object holds, so units are
 * attached by dotted PATH -- the same path the editor already builds for its
 * labels. Two maps, both keyed by approach id then path:
 *
 *   CONFIG_LABELS  replaces the whole caption. Used only where the KEY ITSELF
 *                  misleads, and the JSON key is never renamed with it.
 *   CONFIG_UNITS   appends a derived unit. A string, or a function of the
 *                  config so a unit can be read off the config's own
 *                  declarations (MPC states its input_units and output_units).
 *
 * A path absent from both is left bare on purpose: its dimension is not
 * derivable, and an invented unit is worse than none.
 * ----------------------------------------------------------------------- */

/**
 * The Potts fluctuation amplitude. Every biologist reads "temperature: 10" as
 * ten degrees Celsius; it is a dimensionless amplitude in the Metropolis
 * acceptance criterion and has nothing to do with heat. Relabelled, not renamed
 * -- the backend key stays `temperature`.
 */
const POTTS_TEMPERATURE_LABEL =
    'Fluctuation amplitude (Potts "temperature", dimensionless - NOT degrees C)';

const CONFIG_LABELS = {
    abm: { temperature: POTTS_TEMPERATURE_LABEL },
    cc3d: { temperature: POTTS_TEMPERATURE_LABEL },
    mpc: {},
};

const CONFIG_UNITS = {
    abm: {
        "grid.width": "lattice sites", "grid.height": "lattice sites",
        num_mcs: "MCS", save_every: "MCS", seed: "dimensionless",
    },
    cc3d: {
        "lattice.x": "lattice sites", "lattice.y": "lattice sites",
        "lattice.z": "lattice sites", steps: "MCS",
        neighbor_order: "dimensionless",
        "volume_constraint.target_volume": "lattice sites",
    },
    mpc: {
        // Times are in the project's own time unit; the horizons count control
        // intervals, not time. Amplitudes take the units MPC already declares.
        duration: () => timeUnits(),
        control_interval: () => timeUnits(),
        prediction_horizon: "control intervals",
        control_horizon: "control intervals",
        target: (config) => sanitiseUnit(config.output_units),
        output_min: (config) => sanitiseUnit(config.output_units),
        output_max: (config) => sanitiseUnit(config.output_units),
        input_min: (config) => sanitiseUnit(config.input_units),
        input_max: (config) => sanitiseUnit(config.input_units),
        input_rate_limit: (config) => {
            const input = sanitiseUnit(config.input_units);
            return input ? `${input}/${timeUnits()}` : "";
        },
        "model.tau": () => timeUnits(),
        "plant.tau": () => timeUnits(),
        weight_tracking: "dimensionless", weight_effort: "dimensionless",
        weight_rate: "dimensionless", tolerance: "dimensionless",
    },
};

/**
 * Resolves a config path to its caption, units included where derivable.
 * The unit is either an author-written literal ("lattice sites", "MCS") or
 * composed from parts already put through sanitiseUnit -- so it is NOT sanitised
 * again here, which would strip the spaces and the "/" out of a valid label.
 * Captions reach the DOM through el()/textContent, never innerHTML.
 */
function configCaption(approachId, path, config) {
    const override = (CONFIG_LABELS[approachId] || {})[path];
    if (override) return override;
    const spec = (CONFIG_UNITS[approachId] || {})[path];
    const unit = typeof spec === "function" ? spec(config || {}) : spec;
    return `${path}${unitSuffix(unit || "")}`;
}

/**
 * A cell's volume is a count of lattice SITES, and how many sites a cell of a
 * given edge width occupies depends on the lattice depth: w² sites on a 2D
 * lattice, w³ on a 3D one. A real CompuCell3D run with z > 1 seeds cell_width**3
 * sites, so the same target_volume means a very different cell in the two cases.
 */
const SITE_COUNT_NOTE =
    "target_volume counts lattice SITES: a cell of edge width w occupies w² sites " +
    "on a 2D lattice (z = 1) and w³ on a 3D one (z > 1)";

/** The same rule, compact enough to sit inside the one-line mapping. */
const SITE_COUNT_SHORT = "sites, not area: w² sites in 2D, w³ in 3D";

/** The lattice dimensions an approach uses, or null when it is not lattice-based. */
function latticeDimensions(approachId, config) {
    if (approachId === "abm") {
        const grid = config.grid || {};
        return {
            nx: Number(grid.width), ny: Number(grid.height),
            // The ABM grid is 2D by construction -- it has no depth key at all.
            nz: 1, steps: Number(config.num_mcs),
        };
    }
    if (approachId === "cc3d") {
        const lattice = config.lattice || {};
        const nz = Number(lattice.z);
        return {
            nx: Number(lattice.x), ny: Number(lattice.y),
            nz: Number.isFinite(nz) && nz > 0 ? nz : 1, steps: Number(config.steps),
        };
    }
    return null;
}

/** target_volume lives in a JSON list for ABM and a nested object for CC3D. */
function configTargetVolume(approachId, config) {
    if (approachId === "abm") {
        const types = Array.isArray(config.cell_types) ? config.cell_types : [];
        const first = types.find((entry) => entry && Number.isFinite(Number(entry.target_volume)));
        return first ? Number(first.target_volume) : NaN;
    }
    if (approachId === "cc3d") {
        return Number((config.volume_constraint || {}).target_volume);
    }
    return NaN;
}

/** The model's time window, for converting Monte Carlo steps into model time. */
function modelTimeWindow() {
    const field = (((state.project || {}).model || {}).fields || [])[0];
    if (!field) return NaN;
    const span = Number(field.t_end) - Number(field.t_start || 0);
    return Number.isFinite(span) && span > 0 ? span : NaN;
}

/**
 * The single mapping a researcher currently has to guess: a 60x60 lattice and a
 * 40x25 um domain were shown side by side with nothing stating that they cover
 * the same space, that target_volume counts SITES rather than um², or what one
 * Monte Carlo step is worth in model time.
 *
 * Site size is the domain extent divided by the lattice count in each axis, so
 * sites are generally NOT square; site area is their product, and a
 * target_volume in sites times that area is the cell's physical area. The MCS
 * conversion divides the first field's time window by the step count. Every
 * branch that cannot be computed says so instead of printing NaN.
 */
function latticeMappingText() {
    const approachId = (state.project || {}).selected_approach;
    const config = ((state.project || {}).approaches || {})[approachId];
    if (!config) return "";
    const dims = latticeDimensions(approachId, config);
    if (!dims) return "";

    const domain = (state.project || {}).domain || {};
    if (domain.kind === "interval") {
        return "This lattice covers a 2D region — the current domain is a 1D interval, " +
               "so there is no site-to-domain mapping to state.";
    }
    if (!(dims.nx > 0 && dims.ny > 0)) {
        return "Set positive lattice dimensions to see how one site maps onto the domain.";
    }

    const size = latticeSiteSize();
    if (!size) {
        return "Apply a domain with a positive extent to see the site-to-domain mapping.";
    }

    const length = units();
    const inPlane =
        `${fmt(size.domainWidth, 2)} × ${fmt(size.domainHeight, 2)} ${length} ` +
        `→ 1 site = ${fmt(size.width, 2)} × ${fmt(size.height, 2)} ${length}`;
    const targetVolume = configTargetVolume(approachId, config);

    // A lattice deeper than one site is 3D, but every domain shape this app offers
    // is 2D -- there is no third extent to divide, so the site's depth is NOT
    // derivable and sites cannot be converted to an area or a volume. Say that,
    // rather than quoting an in-plane area that would be wrong by a factor of z.
    if (dims.nz > 1) {
        const parts = [
            `${dims.nx} × ${dims.ny} × ${dims.nz} sites over a 2D domain of ${inPlane} in-plane`,
            `the domain has no third extent, so one site's depth — and any conversion ` +
            `from sites to ${areaUnit()} — is not derivable`,
        ];
        if (Number.isFinite(targetVolume) && targetVolume > 0) {
            parts.push(`target_volume ${fmt(targetVolume, 0)} is a count of sites in 3D ` +
                       `(w³ for a cell of edge width w), not an area`);
        }
        parts.push(mcsClause(dims));
        return parts.filter(Boolean).join("; ");
    }

    const parts = [`${dims.nx} × ${dims.ny} sites over ${inPlane}`];
    if (Number.isFinite(targetVolume) && targetVolume > 0) {
        parts.push(`target_volume ${fmt(targetVolume, 0)} sites ≈ ` +
                   `${fmt(targetVolume * size.area, 2)} ${areaUnit()} (${SITE_COUNT_SHORT})`);
    }
    parts.push(mcsClause(dims));
    return parts.filter(Boolean).join("; ");
}

/** What one Monte Carlo step is worth in model time, or why it cannot be said. */
function mcsClause(dims) {
    if (!(dims.steps > 0)) return "";
    const window = modelTimeWindow();
    if (!Number.isFinite(window)) {
        return `${dims.steps} MCS — set a field time window in stage 4 to see what ` +
               `one MCS is worth in ${timeUnits()}`;
    }
    return `${dims.steps} MCS over the field time window of ${fmt(window, 3)} ` +
           `${timeUnits()} → 1 MCS ≈ ${fmt(window / dims.steps, 4)} ${timeUnits()}`;
}

/**
 * Rewrites the mapping line in place. Called whenever either side of the mapping
 * can have moved: the domain, the lattice, target_volume, the time window, or a
 * unit selection.
 */
function refreshLatticeMapping() {
    const host = $("approach-lattice-map");
    if (!host) return;
    const text = latticeMappingText();
    // textContent, not innerHTML: nothing here needs markup, and the numbers come
    // from an imported project.
    host.textContent = text;
    host.hidden = !text;
}

async function loadApproaches() {
    try {
        const payload = await api("/api/approaches");
        approachCapabilities = payload.approaches || [];
        renderApproachCards();
    } catch (error) {
        $("approach-cards").innerHTML =
            `<div class="wf-issue">Could not load approaches: ${escapeHtml(error.message)}</div>`;
    }
}

function renderApproachCards() {
    const host = $("approach-cards");
    host.innerHTML = "";
    approachCapabilities.forEach((capability) => {
        const card = el("div", "wf-approach-card");
        card.tabIndex = 0;
        card.setAttribute("role", "button");
        card.setAttribute("aria-pressed",
            String(state.project && state.project.selected_approach === capability.approach_id));
        if (!capability.available) card.classList.add("unavailable");
        if (state.project && state.project.selected_approach === capability.approach_id) {
            card.classList.add("selected");
        }

        const heading = el("h3", null, capability.label);
        card.appendChild(heading);
        const pill = el("span", `wf-approach-pill ${capability.available ? "on" : "off"}`,
                        capability.available ? "available" : "unavailable");
        card.appendChild(pill);

        if (capability.notes) card.appendChild(el("p", "wf-approach-note", capability.notes));

        // An unavailable approach states exactly what is missing. It is never hidden,
        // and never silently replaced by another approach.
        if (!capability.available && capability.unavailable_reason) {
            card.appendChild(el("p", "wf-approach-reason", capability.unavailable_reason));
        }

        const meta = el("div", "wf-approach-meta");
        meta.innerHTML = [
            `engine: ${escapeHtml(capability.engine_name || "—")}` +
            (capability.engine_version ? ` ${escapeHtml(capability.engine_version)}` : ""),
            `dims: ${(capability.dimensions || []).join("/") || "—"}`,
            `pause ${capability.supports_pause ? "yes" : "no"} · ` +
            `cancel ${capability.supports_cancel ? "yes" : "no"} · ` +
            `seeded ${capability.deterministic_with_seed ? "reproducible" : "no"}`,
        ].join("<br>");
        card.appendChild(meta);

        const choose = () => {
            // An unavailable approach is still SELECTABLE on purpose. The adapter
            // contract is that configuration and runnable-project export stay
            // available without the engine, so blocking selection here would have
            // made both unreachable -- the opposite of the honest-unavailable
            // design. Running it is refused instead, by the backend, on its merits.
            if (!capability.available) {
                toast(capability.unavailable_reason ||
                      `${capability.label} is not available on this machine.`, "warn", 12000);
            }
            selectApproach(capability.approach_id);
        };
        card.addEventListener("click", choose);
        card.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(); }
        });
        host.appendChild(card);
    });
}

function selectApproach(id) {
    state.project.selected_approach = id;
    const approaches = state.project.approaches || (state.project.approaches = {});
    // Each approach keeps its own block, so switching preserves the others.
    if (!approaches[id]) {
        approaches[id] = JSON.parse(JSON.stringify(APPROACH_DEFAULTS[id] || {}));
    }
    renderApproachCards();
    renderApproachConfig();
    validateProject();
}

/**
 * Config textareas whose text does not parse as JSON, keyed by field path.
 * While this is non-empty the SCREEN and the value that would RUN disagree, so the
 * approach stage is reported invalid and the run is refused.
 */
const unparsedConfigText = {};

function renderUnparsedConfigWarning() {
    const keys = Object.keys(unparsedConfigText);
    if (!keys.length) {
        renderIssues("approach-issues", { issues: [] });
        return;
    }
    renderIssues("approach-issues", {
        issues: keys.map((key) => ({
            severity: "error",
            message: `"${key}" is not valid JSON (${unparsedConfigText[key]}). `
                     + `The box shows your edit, but the simulation would still use the `
                     + `previous value. Fix the JSON before running.`,
        })),
    });
}

function renderApproachConfig() {
    const host = $("approach-config");
    host.innerHTML = "";
    const id = state.project.selected_approach;
    $("approach-config-name").textContent = id
        ? (approachCapabilities.find((c) => c.approach_id === id) || {}).label || id
        : "—";
    // The runnable-project export is CompuCell3D-specific, and deliberately offered
    // even when the engine is unavailable -- that export is the whole point of the
    // honest-unavailable state.
    const cc3dExport = $("approach-export-cc3d");
    if (cc3dExport) cc3dExport.hidden = id !== "cc3d";
    if (!id) {
        host.appendChild(el("div", "wf-empty-state", "Select an approach above."));
        return;
    }
    const config = state.project.approaches[id] || {};
    buildConfigEditor(host, config, "", id, config);
    refreshLatticeMapping();
}

/**
 * Downloads the runnable CompuCell3D project as individual files.
 * Works whether or not CompuCell3D is installed here, because the package is
 * generated from the project rather than by the engine.
 */
async function exportCc3dProject() {
    try {
        const payload = await post("/api/project/export/cc3d", { project: state.project });
        const files = (payload && payload.files) || {};
        const names = Object.keys(files);
        if (!names.length) {
            toast("The CompuCell3D exporter returned no files.", "error", 8000);
            return;
        }
        names.forEach((name) => {
            const type = name.endsWith(".xml") ? "application/xml" : "text/plain";
            download(name.replace(/[\\/]/g, "_"), files[name], type);
        });
        toast(`Exported ${names.length} CompuCell3D file(s): ${names.join(", ")}.`,
              "success", 9000);
    } catch (error) {
        toast(`Could not export the CompuCell3D project: ${error.message}`, "error", 9000);
    }
}

/**
 * Renders scalars as inputs and nested objects as sub-grids, recursively.
 * `approachId` and `root` are carried down so a caption can look its unit up by
 * dotted path, and so the lattice/domain mapping line can read the whole config
 * rather than just the sub-object it is being drawn under.
 */
function buildConfigEditor(host, object, prefix, approachId, root) {
    Object.entries(object).forEach(([key, value]) => {
        const path = `${prefix}${key}`;
        if (Array.isArray(value)) {
            const wrapper = el("label", "wf-field");
            wrapper.style.gridColumn = "1 / -1";
            wrapper.appendChild(el("span", null, `${path} (JSON list)`));
            const area = el("textarea");
            area.rows = Math.min(10, Math.max(3, JSON.stringify(value, null, 1).split("\n").length));
            area.value = JSON.stringify(value, null, 1);
            area.addEventListener("change", () => {
                try {
                    object[key] = JSON.parse(area.value);
                    area.style.borderColor = "";
                    area.removeAttribute("aria-invalid");
                    delete unparsedConfigText[path];
                    renderUnparsedConfigWarning();
                    // ABM keeps target_volume inside this list, so the mapping line
                    // is only right if it is recomputed when the list is edited.
                    refreshLatticeMapping();
                    validateProject();
                } catch (error) {
                    // Keep the text so the user can fix it; mark it instead of reverting.
                    area.style.borderColor = "#ff007f";
                    area.setAttribute("aria-invalid", "true");
                    // The OLD array is still what runs. Previously the only lasting
                    // signal was a 1px border -- colour alone, no text -- and
                    // validateProject was never called, so the rail badge still read
                    // valid while the screen showed cell types the run was not using.
                    // That divergence is now persistent, textual, and blocks the run.
                    unparsedConfigText[path] = error.message;
                    renderUnparsedConfigWarning();
                    toast(`${key} is not valid JSON: ${error.message}`, "error", 9000);
                }
            });
            wrapper.appendChild(area);
            // A JSON textarea has no per-key label to hang a unit on, so the units
            // of the keys inside it are stated once, here.
            if (key === "cell_types") {
                wrapper.appendChild(el("p", "wf-hint",
                    `${SITE_COUNT_NOTE}. max_volume_before_division counts sites too, ` +
                    "and target_surface counts site edges — none of them is an area. " +
                    "The line under the lattice size converts sites to " +
                    `${areaUnit() || "domain units"} when the lattice is 2D.`));
            }
            if (key === "contact_energies") {
                // Reinforces the fluctuation-amplitude relabel: what governs the
                // simulation is the RATIO of these energies to that amplitude, and
                // neither carries a physical unit.
                wrapper.appendChild(el("p", "wf-hint",
                    "Contact energies are in dimensionless Potts energy units — the same " +
                    "scale as the fluctuation amplitude above, whose ratio to them sets " +
                    "how readily boundaries move. Not joules, and not degrees."));
            }
            host.appendChild(wrapper);
            return;
        }
        if (value && typeof value === "object") {
            const group = el("div");
            group.style.gridColumn = "1 / -1";
            group.appendChild(el("div", "wf-subhead", path));
            const grid = el("div", "wf-param-grid");
            // Pass the group name down as the prefix. Dropping it made MPC's
            // predictive-model and plant blocks render as two identical
            // kind/tau/gain triplets, so a researcher could not tell which was
            // which -- and model-versus-plant mismatch is the point of MPC.
            buildConfigEditor(grid, value, `${path}.`, approachId, root);
            group.appendChild(grid);
            // The lattice-to-domain mapping belongs directly under the inputs it
            // explains, so it sits inside this group rather than at the panel foot.
            if (!prefix && (key === "grid" || key === "lattice")) {
                const mapping = el("p", "wf-hint");
                mapping.id = "approach-lattice-map";
                group.appendChild(mapping);
            }
            // CC3D keeps target_volume in its own nested block rather than in the
            // cell_types list, so the site-count rule has to be stated here too.
            if (!prefix && key === "volume_constraint") {
                group.appendChild(el("p", "wf-hint", `${SITE_COUNT_NOTE}.`));
            }
            host.appendChild(group);
            return;
        }
        const field = el("label", "wf-field");
        field.appendChild(el("span", null, configCaption(approachId, path, root)));
        const input = el("input");
        // A null means "no constraint" (MPC's output_min / output_max). It has to
        // stay a NULLABLE NUMBER: the text path sent the literal string "null" or
        // "" to the solver instead of clearing the constraint.
        const nullable = value === null;
        const numeric = typeof value === "number" || nullable;
        input.type = numeric ? "number" : "text";
        if (numeric) input.step = "any";
        input.value = nullable ? "" : value;
        if (nullable) input.placeholder = "none";
        input.addEventListener("change", () => {
            const raw = input.value.trim();
            if (nullable) {
                object[key] = raw === "" ? null : Number(raw);
            } else if (numeric) {
                if (raw === "" || !Number.isFinite(Number(raw))) {
                    // Never silently coerce a blank to 0. Number("") === 0, so a
                    // cleared diffusion coefficient used to become zero -- a
                    // scientific error with no visible symptom.
                    input.value = String(object[key]);
                    // The path, not the caption: this names the field being refused.
                    toast(`${path} must be a number.`, "warn");
                    return;
                }
                object[key] = Number(raw);
            } else {
                object[key] = input.value;
            }
            // A lattice dimension, a step count or a target_volume all move the
            // mapping, so it is recomputed on every scalar edit rather than only
            // when the panel is rebuilt.
            refreshLatticeMapping();
            validateProject();
        });
        field.appendChild(input);
        host.appendChild(field);
    });
}

async function exportApproachConfiguration() {
    const id = state.project.selected_approach;
    if (!id) { toast("Select an approach first.", "warn"); return; }
    try {
        const payload = await post(`/api/approaches/${encodeURIComponent(id)}/export`,
                                   { project: state.project });
        if (payload.files) {
            Object.entries(payload.files).forEach(([name, contents]) => {
                download(name.replace(/\//g, "_"), contents, "text/plain");
            });
            toast(`Exported ${Object.keys(payload.files).length} file(s).`, "success");
        } else {
            download(`${id}_configuration.json`, JSON.stringify(payload, null, 2));
            toast("Configuration exported.", "success");
        }
    } catch (error) {
        toast(`Export failed: ${error.message}`, "error");
    }
}

/* --------------------------------------------------------------------------
 * Stage 8/9: validate, compile, run
 * ----------------------------------------------------------------------- */
async function validateProject() {
    if (!state.project) return null;
    try {
        const payload = await post("/api/project/validate", {
            project: state.project,
            approach: state.project.selected_approach || null,
        });
        state.validation = payload;
        paintStageBadges(payload);
        const approachStage = payload.stages && payload.stages.approach;
        renderIssues("approach-issues", approachStage,
                     approachStage && approachStage.status === "valid"
                         ? "Approach configuration is valid." : null);
        return payload;
    } catch (error) {
        toast(`Project validation failed: ${error.message}`, "error");
        return null;
    }
}

async function validateEverything() {
    const payload = await validateProject();
    if (!payload) return;
    renderIssues("run-issues", payload,
                 payload.valid ? "Every stage is valid. Ready to compile." : null);
    if (payload.valid) {
        toast("All stages valid.", "success");
        setStatus("Ready", "green");
    } else {
        toast(`${payload.error_count} error(s) across the workflow. See the stage badges.`,
              "error", 8000);
        setStatus("Invalid", "red");
    }
}

async function startRun() {
    if (!state.project.selected_approach) {
        toast("Select an approach in stage 6 first.", "warn");
        showStage("approach");
        return;
    }
    // Refuse while any config textarea's text does not parse: the researcher would be
    // running the PREVIOUS value while reading their edit on screen.
    const unparsed = Object.keys(unparsedConfigText);
    if (unparsed.length) {
        toast(`Cannot run: ${unparsed.join(", ")} contains invalid JSON, so the run `
              + `would not use what the box shows. Fix it in stage 7.`, "error", 10000);
        showStage("approach");
        return;
    }
    setStatus("Compiling…", "yellow");
    setRunButtons("compiling");
    try {
        const status = await post("/api/runs", {
            project: state.project,
            approach: state.project.selected_approach,
            seed: Number($("run-seed").value),
            start: true,
        });
        state.run = status;
        applyRunStatus(status);
        // A run that failed validation or compilation comes back with that state and
        // no polling to do: report it as-is rather than implying it started.
        if (["invalid", "failed", "cancelled"].includes(status.state)) {
            setStatus(status.state === "invalid" ? "Invalid" : "Failed", "red");
            toast(status.error || `Run ${status.state}.`, "error", 9000);
            renderIssues("run-issues", { issues: status.issues || [] });
            return;
        }
        pollRun(status.run_id);
    } catch (error) {
        setStatus("Failed", "red");
        setRunButtons("failed");
        toast(`Could not start the run: ${error.message}`, "error", 9000);
    }
}

function pollRun(runId) {
    if (state.runPoll) clearInterval(state.runPoll);
    state.runPoll = setInterval(async () => {
        try {
            const status = await api(`/api/runs/${encodeURIComponent(runId)}`);
            state.run = status;
            applyRunStatus(status);
            await refreshRunLog(runId);
            if (["completed", "failed", "cancelled"].includes(status.state)) {
                clearInterval(state.runPoll);
                state.runPoll = null;
                await finishRun(status);
            }
        } catch (error) {
            clearInterval(state.runPoll);
            state.runPoll = null;
            toast(`Lost contact with the run: ${error.message}`, "error");
        }
    }, 600);
}

async function finishRun(status) {
    await refreshRunList();
    if (status.state === "completed") {
        setStatus("Completed", "green");
        toast("Run completed.", "success");
        await loadResults(status.run_id);
        showStage("results");
    } else if (status.state === "failed") {
        setStatus("Failed", "red");
        renderIssues("run-issues", { issues: [{ severity: "error", message: status.error || "The run failed." }] });
        toast(status.error || "The run failed.", "error", 10000);
    } else {
        setStatus("Cancelled", "yellow");
        toast("Run cancelled.", "warn");
    }
}

function applyRunStatus(status) {
    const badge = $("run-state-badge");
    badge.textContent = status.state;
    badge.className = `wf-run-badge ${status.state}`;
    $("run-progress").style.width = `${Math.round((status.progress || 0) * 100)}%`;
    const engine = status.engine || {};
    $("run-message").textContent = [
        status.message || "",
        engine.engine_name ? `· ${engine.engine_name}` : "",
        engine.engine_version ? engine.engine_version : "",
        status.duration_secs ? `· ${fmt(status.duration_secs, 1)}s` : "",
    ].filter(Boolean).join(" ");
    setRunButtons(status.state);
    // The run and results badges are derived from this state, so they have to be
    // repainted here as well -- validation alone never fires again after a run
    // starts, which is why they previously froze at "not done yet".
    paintStageBadges(state.validation);
}

function setRunButtons(runState) {
    const running = runState === "running";
    const paused = runState === "paused";
    const active = running || paused || runState === "queued" || runState === "compiling";
    $("run-start").disabled = active;
    $("run-pause").disabled = !running;
    $("run-resume").disabled = !paused;
    $("run-cancel").disabled = !active;
}

async function refreshRunLog(runId) {
    try {
        const payload = await api(`/api/runs/${encodeURIComponent(runId)}/logs?limit=300`);
        const host = $("run-log");
        host.innerHTML = (payload.logs || []).map((entry) =>
            `<div class="lvl-${escapeHtml(entry.level)}">${escapeHtml(entry.message)}</div>`
        ).join("");
        host.scrollTop = host.scrollHeight;
    } catch (error) {
        /* the log is a convenience; a failure here must not stop the run view */
    }
}

async function controlRun(action) {
    if (!state.run) { toast("No run to control.", "warn"); return; }
    try {
        const status = await post(`/api/runs/${encodeURIComponent(state.run.run_id)}/${action}`, {});
        state.run = status;
        applyRunStatus(status);
        if (action === "cancel") toast("Cancellation requested.", "warn");
    } catch (error) {
        // 409 means the state does not allow it, which is information, not a bug.
        toast(error.message, error.status === 409 ? "warn" : "error", 8000);
    }
}

async function refreshRunList() {
    try {
        const payload = await api("/api/runs?limit=30");
        buildTable($("run-table"), ["run", "approach", "state", "seed", "duration", "engine"],
            (payload.runs || []).map((run) => ({
                key: run.run_id,
                selected: state.run && state.run.run_id === run.run_id,
                cells: [
                    run.run_id.replace("run_", ""),
                    run.approach,
                    run.state,
                    run.seed === null || run.seed === undefined ? "—" : run.seed,
                    run.duration_secs ? `${fmt(run.duration_secs, 1)}s` : "—",
                    (run.engine && run.engine.engine_version) || "—",
                ],
                onClick: async () => {
                    state.run = run;
                    applyRunStatus(run);
                    await refreshRunLog(run.run_id);
                    if (run.state === "completed") {
                        await loadResults(run.run_id);
                        showStage("results");
                    }
                },
            })), "No runs yet.");
    } catch (error) {
        /* listing is non-critical */
    }
}


/* --------------------------------------------------------------------------
 * Stage 10: results workspace
 *
 * Left: entity list with filter and selected-cell properties.
 * Centre: the view with playback.
 * Right: plots, aggregate statistics and export.
 * Everything is drawn from the run's actual output; nothing is synthesised.
 * ----------------------------------------------------------------------- */
async function loadResults(runId) {
    try {
        const payload = await api(`/api/runs/${encodeURIComponent(runId)}/results`);
        state.results = payload.results;
        state.frame = 0;
        // Selection and draw transform belong to the PREVIOUS run. Carrying them over
        // meant the detail panel showed a cell from the old run while the lattice
        // ringed whichever cell in the NEW run happened to share that id -- two runs'
        // data on screen simultaneously, both looking authoritative.
        state.selectedCellId = null;
        state.resultsTransform = null;
        stopPlayback();
        prepareResultsView();
        renderCellDetail(null);
    } catch (error) {
        state.results = null;
        toast(`Could not load results: ${error.message}`, "error");
    }
    // Whether results arrived or not, the results badge reflects it.
    paintStageBadges(state.validation);
}

function prepareResultsView() {
    const results = state.results;
    if (!results) return;
    const frames = frameCount();
    const scrub = $("pb-scrub");
    scrub.max = String(Math.max(0, frames - 1));
    scrub.value = "0";

    $("results-view-label").textContent =
        results.kind === "agents" ? "Agent lattice" : "Control trajectory";

    const variableSelect = $("results-variable");
    variableSelect.innerHTML = "";
    if (results.series) {
        Object.keys(results.series).forEach((name) => {
            const option = el("option", null, name);
            option.value = name;
            variableSelect.appendChild(option);
        });
        // Tracking is the point of an MPC run, so default to the output/target pair.
        if (results.series.output) variableSelect.value = "output";
    } else if (results.cell_counts) {
        const keys = new Set();
        results.cell_counts.forEach((entry) => Object.keys(entry || {}).forEach((k) => keys.add(k)));
        keys.forEach((key) => {
            const option = el("option", null, key);
            option.value = key;
            variableSelect.appendChild(option);
        });
    }

    renderResultsSummary();
    renderResultsWarnings();
    renderResultsLegend();
    renderCellList();
    drawResults();
    drawChart();
}

function frameCount() {
    const results = state.results;
    if (!results) return 0;
    if (results.lattice_frames) return results.lattice_frames.length;
    if (results.t) return results.t.length;
    return 0;
}

function currentTime() {
    const results = state.results;
    if (!results || !results.t || !results.t.length) return null;
    return results.t[Math.min(state.frame, results.t.length - 1)];
}

function drawResults() {
    const canvas = $("results-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(7, 9, 19, 0.95)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const results = state.results;
    if (!results) {
        label(ctx, canvas.width / 2, canvas.height / 2,
              "Run a simulation to see results here.", "#6b7280", "center");
        return;
    }
    const time = currentTime();
    // The playback clock read a bare "t = 0.35" in a UI with no time unit at all,
    // so it was a number in a system the researcher had to guess. No unit is quoted
    // when there is no time yet -- "t = — s" would claim a measurement.
    $("results-time-label").textContent =
        time === null ? "t = —" : `t = ${fmt(time)} ${timeUnits()}`;

    if (results.lattice_frames) drawLattice(ctx, canvas, results);
    else if (results.series) drawTrajectory(ctx, canvas, results);
}

function drawLattice(ctx, canvas, results) {
    const frame = results.lattice_frames[Math.min(state.frame, results.lattice_frames.length - 1)];
    if (!frame || !frame.length) return;
    const rows = frame.length, cols = frame[0].length;
    const cell = Math.max(1, Math.floor(Math.min(canvas.width / cols, canvas.height / rows)));
    const originX = Math.floor((canvas.width - cols * cell) / 2);
    const originY = Math.floor((canvas.height - rows * cell) / 2);
    // Persist the transform so a click on the canvas can be mapped back to a cell.
    // Without it the results view was display-only: the table could highlight a cell
    // in the view, but clicking the view itself did nothing.
    state.resultsTransform = { originX, originY, cell, rows, cols };

    for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
            const rgb = frame[row][col];
            if (!rgb) continue;
            ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
            ctx.fillRect(originX + col * cell, originY + row * cell, cell, cell);
        }
    }
    if ($("ov-grid").checked && cell >= 4) {
        ctx.strokeStyle = "rgba(255,255,255,0.05)";
        ctx.lineWidth = 1;
        for (let col = 0; col <= cols; col += 1) {
            ctx.beginPath();
            ctx.moveTo(originX + col * cell, originY);
            ctx.lineTo(originX + col * cell, originY + rows * cell);
            ctx.stroke();
        }
        for (let row = 0; row <= rows; row += 1) {
            ctx.beginPath();
            ctx.moveTo(originX, originY + row * cell);
            ctx.lineTo(originX + cols * cell, originY + row * cell);
            ctx.stroke();
        }
    }

    const rowsForFrame = (results.cells || [])[Math.min(state.frame, (results.cells || []).length - 1)] || [];
    const selectedId = state.selectedCellId;
    rowsForFrame.forEach((row) => {
        const cx = originX + Number(row.lattice_x ?? row.x) * cell;
        const cy = originY + Number(row.lattice_y ?? row.y) * cell;
        if ($("ov-centroids").checked) {
            ctx.fillStyle = row.id === selectedId ? "#00f2fe" : "rgba(255,255,255,0.6)";
            ctx.beginPath();
            ctx.arc(cx, cy, row.id === selectedId ? 5 : 2, 0, Math.PI * 2);
            ctx.fill();
        }
        if (row.id === selectedId) {
            // Ring the selected cell so table selection is visible in the view.
            ctx.strokeStyle = "#00f2fe";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(cx, cy, 10, 0, Math.PI * 2);
            ctx.stroke();
        }
        if ($("ov-ids").checked) label(ctx, cx + 7, cy - 7, String(row.id), "#9ca3af");
    });
}

function drawTrajectory(ctx, canvas, results) {
    const times = results.t || [];
    const output = results.series.output || [];
    const target = results.series.target || [];
    // A silent `return` left the panel looking like a rendering failure. Say why
    // there is no plot instead.
    if (!times.length) {
        label(ctx, canvas.width / 2, canvas.height / 2,
              "This run reported no time points, so there is no trajectory to plot.",
              "#6b7280", "center");
        return;
    }

    const padding = { left: 52, right: 14, top: 16, bottom: 30 };
    const width = canvas.width - padding.left - padding.right;
    const height = canvas.height - padding.top - padding.bottom;
    // ONE pass over both series. Math.min(...output.concat(target)) allocated a
    // joined copy and then spread it onto the call stack, which threw RangeError
    // past ~65k samples -- uncaught, inside a draw, so the plot silently blanked.
    const range = extent(output, target);
    if (!range.count) {
        label(ctx, canvas.width / 2, canvas.height / 2,
              "The output and target series contain no finite values to plot.",
              "#6b7280", "center");
        return;
    }
    let low = range.low, high = range.high;
    if (low === high) { low -= 1; high += 1; }
    const span = high - low;

    const sx = (index) => padding.left + (index / Math.max(1, times.length - 1)) * width;
    const sy = (value) => padding.top + height - ((value - low) / span) * height;

    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i += 1) {
        const y = padding.top + (height * i) / 4;
        ctx.beginPath(); ctx.moveTo(padding.left, y); ctx.lineTo(padding.left + width, y); ctx.stroke();
        label(ctx, padding.left - 6, y + 3, fmt(high - (span * i) / 4, 2), "#6b7280", "right");
    }

    const line = (values, colour, dashed) => {
        if (!values.length) return;
        ctx.save();
        ctx.strokeStyle = colour;
        ctx.lineWidth = 2;
        if (dashed) ctx.setLineDash([5, 4]);
        ctx.beginPath();
        values.forEach((value, index) => {
            const x = sx(index), y = sy(value);
            if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.restore();
    };
    line(target, "#9ca3af", true);
    line(output, "#00f2fe", false);

    // Playhead: ties the plot to the timeline scrubber.
    const playX = sx(Math.min(state.frame, times.length - 1));
    ctx.strokeStyle = "#ff007f";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playX, padding.top);
    ctx.lineTo(playX, padding.top + height);
    ctx.stroke();
    label(ctx, padding.left, canvas.height - 8, "target (dashed) vs output (solid)", "#6b7280");
}

function renderResultsLegend() {
    const host = $("results-legend");
    host.innerHTML = "";
    const results = state.results;
    if (!results) return;
    if (results.cell_types) {
        Object.entries(results.cell_types).forEach(([id, info]) => {
            const item = el("span");
            const swatch = el("i", "wf-swatch");
            const colour = info.color || [128, 128, 128];
            swatch.style.background = `rgb(${colour[0]},${colour[1]},${colour[2]})`;
            item.appendChild(swatch);
            item.appendChild(document.createTextNode(`${info.name} (type ${id})`));
            host.appendChild(item);
        });
    } else if (results.series) {
        host.innerHTML =
            '<span><i class="wf-swatch" style="background:#00f2fe"></i>output</span>' +
            '<span><i class="wf-swatch" style="background:#9ca3af"></i>target</span>';
    }
}

const CELL_COLUMNS = ["id", "type", "state", "x", "y", "vol"];

/** The six rendered values for one cell row, in column order. */
function cellRowValues(row) {
    return [row.id, row.type, row.state, fmt(row.x, 1), fmt(row.y, 1),
            row.volume === undefined ? "—" : fmt(row.volume, 0)];
}

/**
 * Row click. The cell is resolved from the CURRENT frame by id rather than captured
 * when the table was built, so the detail panel cannot describe a cell as it was in
 * whichever frame the table last happened to be rebuilt in. It also means the
 * delegated handler stays valid while playback advances.
 */
function selectResultsCell(id) {
    state.selectedCellId = state.selectedCellId === id ? null : id;
    const frames = (state.results && state.results.cells) || [];
    const rows = frames[Math.min(state.frame, frames.length - 1)] || [];
    const row = rows.find((entry) => entry.id === id) || null;
    renderCellList();
    renderCellDetail(row);
    drawResults();
}

function renderCellList() {
    const results = state.results;
    const frames = (results && results.cells) || [];
    const rows = frames[Math.min(state.frame, frames.length - 1)] || [];
    const filterInput = $("results-filter");
    const filter = filterInput ? filterInput.value.trim().toLowerCase() : "";
    const visible = rows.filter((row) => !filter ||
        String(row.id).includes(filter) ||
        String(row.type || "").toLowerCase().includes(filter) ||
        String(row.state || "").toLowerCase().includes(filter));

    // The TRUE count of matching cells, written on every path and independently of
    // how many rows the table below ends up showing, so any row cap stays honest.
    $("results-cell-count").textContent = visible.length;

    const table = $("results-cell-table");
    if (!table) return;
    const body = table.tBodies[0];
    // Structural signature: which cells, in which order, plus whether we are in the
    // "no results at all" state, whose empty message differs from "no cells here".
    const key = `${results ? "r" : "-"}:${visible.length}:` +
                `${visible.map((row) => row.id).join(",")}`;

    if (body && key === state.cellListKey) {
        // Playback fast path. setFrame() calls this once per frame -- 25 times a
        // second at 40ms -- and the cell POPULATION rarely differs between
        // neighbouring frames even though x, y and volume do. So the existing rows
        // are reused and only their text is written: no element creation, no
        // listener churn, no tbody swap.
        //
        // Returning early on an unchanged id set alone would be wrong: positions and
        // volumes advance every frame while the ids do not, so the table would sit
        // frozen on stale numbers while the lattice moved beneath it.
        for (let i = 0; i < visible.length && i < body.rows.length; i += 1) {
            const tr = body.rows[i];
            const values = cellRowValues(visible[i]);
            for (let c = 0; c < values.length && c < tr.cells.length; c += 1) {
                const text = String(values[c]);
                if (tr.cells[c].textContent !== text) tr.cells[c].textContent = text;
            }
            tr.classList.toggle("selected", state.selectedCellId === visible[i].id);
        }
        // The delegated handlers are keyed by cell id and resolve the row live, so
        // an unchanged id set needs no listener work at all.
        return;
    }

    state.cellListKey = key;
    buildTable(table, CELL_COLUMNS,
        visible.map((row) => ({
            key: row.id,
            selected: state.selectedCellId === row.id,
            cells: cellRowValues(row),
            onClick: () => selectResultsCell(row.id),
        })), results ? "No cells in this frame." : "Run a simulation first.");
}

/**
 * Maps a click on the results canvas back to the nearest cell in the current frame
 * and selects it, mirroring what clicking a table row does. The spec asks for table
 * selection to highlight the view; this is the other direction, which was missing --
 * the canvas had no listeners at all.
 */
function handleResultsCanvasClick(event) {
    const results = state.results;
    const transform = state.resultsTransform;
    const canvas = $("results-canvas");
    if (!results || !transform || !canvas) return;
    const frames = results.cells || [];
    const rows = frames[Math.min(state.frame, frames.length - 1)] || [];
    if (!rows.length) return;

    // Canvas pixels, not CSS pixels: the element can be scaled by layout.
    const box = canvas.getBoundingClientRect();
    const px = (event.clientX - box.left) * (canvas.width / box.width);
    const py = (event.clientY - box.top) * (canvas.height / box.height);

    let best = null;
    let bestDistance = Infinity;
    rows.forEach((row) => {
        const cx = transform.originX + Number(row.lattice_x ?? row.x) * transform.cell;
        const cy = transform.originY + Number(row.lattice_y ?? row.y) * transform.cell;
        const distance = Math.hypot(px - cx, py - cy);
        if (distance < bestDistance) { bestDistance = distance; best = row; }
    });
    // Require a real hit rather than selecting whatever is furthest away.
    const radius = Math.max(12, transform.cell * 4);
    if (!best || bestDistance > radius) {
        state.selectedCellId = null;
        renderCellList();
        renderCellDetail(null);
        drawResults();
        return;
    }
    state.selectedCellId = state.selectedCellId === best.id ? null : best.id;
    renderCellList();
    renderCellDetail(best);
    drawResults();
}

function renderCellDetail(row, reason) {
    const host = $("results-cell-detail");
    if (!row || state.selectedCellId !== row.id) {
        // "gone" means the cell IS selected but does not exist in this frame -- it
        // divided, died, or is not born yet. Reporting that as "No cell selected"
        // read as if the user had deselected it, hiding a real biological event.
        host.textContent = reason === "gone"
            ? `Cell ${state.selectedCellId} is not present in this frame — it may have ` +
              `divided, died, or not yet appeared.`
            : "No cell selected.";
        return;
    }
    const unitLabel = units();
    // A cell's volume and surface are counts of lattice SITES and site edges, not
    // areas -- "volume 25" alongside a 40x25 um domain read as 25 um². The physical
    // equivalent is appended only when the site size is derivable, which it is not
    // on a 3D lattice: the 2D domain supplies no site depth there.
    const siteArea = latticeSiteArea();
    const physical = Number.isFinite(siteArea) && row.volume !== undefined
        ? ` ≈ ${fmt(Number(row.volume) * siteArea, 2)} ${areaUnit()}`
        : " (no area equivalent — the lattice is 3D or its size is unset)";
    host.innerHTML = [
        `id <b>${row.id}</b>  type <b>${escapeHtml(row.type)}</b>`,
        `state <b>${escapeHtml(row.state)}</b>`,
        `position <b>${fmt(row.x)}</b>, <b>${fmt(row.y)}</b> ${escapeHtml(unitLabel)}`,
        row.volume !== undefined
            ? `volume <b>${fmt(row.volume, 0)}</b> lattice sites${escapeHtml(physical)}` : "",
        row.surface !== undefined ? `surface <b>${fmt(row.surface, 0)}</b> site edges` : "",
        row.age !== undefined && row.age !== null ? `age <b>${row.age}</b> MCS` : "",
        `parent <b>${row.parent_id === null || row.parent_id === undefined ? "—" : row.parent_id}</b>` +
        (row.generation !== undefined ? `  generation <b>${row.generation}</b>` : ""),
    ].filter(Boolean).join("<br>");
}

/**
 * The area one lattice site covers, or NaN when it cannot be derived. Both the
 * mapping line and the cell detail panel go through latticeSiteSize, so the
 * "1 site = ..." line and the "volume ... ≈ ..." readout can never disagree.
 *
 * NaN on a 3D lattice on purpose: a site there has a depth the 2D domain cannot
 * supply, so a site count converts to no area this app can state. Callers show
 * the site count alone rather than a number wrong by a factor of z.
 */
function latticeSiteArea() {
    const size = latticeSiteSize();
    if (!size || size.dims.nz > 1) return NaN;
    return size.area;
}

/**
 * One lattice site's extent in domain units, or null when it is not derivable.
 * Sites are generally NOT square: each axis is divided independently, so a
 * 60x60 lattice over a 40x25 domain gives sites wider than they are tall.
 */
function latticeSiteSize() {
    const approachId = (state.project || {}).selected_approach;
    const config = ((state.project || {}).approaches || {})[approachId];
    if (!config) return null;
    const dims = latticeDimensions(approachId, config);
    if (!dims || !(dims.nx > 0 && dims.ny > 0)) return null;
    if (((state.project || {}).domain || {}).kind === "interval") return null;
    const [x0, y0, x1, y1] = domainBounds();
    const domainWidth = x1 - x0, domainHeight = y1 - y0;
    if (!(domainWidth > 0 && domainHeight > 0)) return null;
    const width = domainWidth / dims.nx, height = domainHeight / dims.ny;
    return { width, height, area: width * height, domainWidth, domainHeight, dims };
}

function renderResultsSummary() {
    const host = $("results-summary");
    const results = state.results;
    if (!results || !results.summary) { host.textContent = "—"; return; }
    host.innerHTML = Object.entries(results.summary).map(([key, value]) => {
        const shown = typeof value === "number" ? fmt(value, 4)
            : typeof value === "boolean" ? (value ? "yes" : "no")
            : escapeHtml(String(value));
        return `${escapeHtml(key.replace(/_/g, " "))} <b>${shown}</b>`;
    }).join("<br>");
}

function renderResultsWarnings() {
    const host = $("results-warnings");
    const results = state.results;
    const notes = [];
    if (results && results.summary) {
        if (results.summary.optimisation_failures) {
            notes.push(`${results.summary.optimisation_failures} optimisation step(s) did not converge.`);
        }
        if (results.summary.constraint_violations) {
            notes.push(`${results.summary.constraint_violations} constraint violation(s) recorded.`);
        }
        if (results.summary.deterministic === false) {
            notes.push("No seed was supplied, so this run is not reproducible.");
        }
    }
    if (results && results.violations && results.violations.length) {
        notes.push(`First violation at t = ${fmt(results.violations[0].time)} ${timeUnits()} ` +
                   `(${results.violations[0].kind}).`);
    }
    host.innerHTML = notes.length
        ? notes.map((note) => escapeHtml(note)).join("<br>")
        : "None.";
}

function drawChart() {
    const results = state.results;
    const canvas = $("results-chart");
    if (!canvas || !results || typeof Chart === "undefined") return;
    const name = $("results-variable").value;
    const times = results.t || [];
    let values = [];
    if (results.series && results.series[name]) values = results.series[name];
    else if (results.cell_counts) values = results.cell_counts.map((entry) => (entry || {})[name] ?? 0);
    if (state.chart) state.chart.destroy();
    state.chart = new Chart(canvas.getContext("2d"), {
        type: "line",
        data: {
            labels: times.map((t) => fmt(t, 2)),
            datasets: [{
                label: name || "value",
                data: values,
                borderColor: "#00f2fe",
                backgroundColor: "rgba(0,242,254,0.12)",
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.25,
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { labels: { color: "#9ca3af" } } },
            scales: {
                x: { ticks: { color: "#6b7280", maxTicksLimit: 6 }, grid: { color: "rgba(255,255,255,0.06)" } },
                y: { ticks: { color: "#6b7280" }, grid: { color: "rgba(255,255,255,0.06)" } },
            },
        },
    });
}

/* --- playback ----------------------------------------------------------- */
function setFrame(index) {
    const frames = frameCount();
    if (!frames) return;
    state.frame = Math.max(0, Math.min(frames - 1, index));
    $("pb-scrub").value = String(state.frame);
    renderCellList();
    // Re-resolve the selected cell IN THIS FRAME. Without this the detail panel kept
    // the volume, surface, age and position from whichever frame you clicked in, and
    // presented them as the current state while the lattice and table advanced --
    // wrong scientific numbers shown with no indication they were stale.
    const allFrames = (state.results && state.results.cells) || [];
    const rows = allFrames[Math.min(state.frame, allFrames.length - 1)] || [];
    const selected = state.selectedCellId === null || state.selectedCellId === undefined
        ? null
        : (rows.find((row) => row.id === state.selectedCellId) || null);
    if (state.selectedCellId !== null && state.selectedCellId !== undefined && !selected) {
        // The cell is genuinely absent this frame (divided, died, or not yet born).
        // That is a different statement from "nothing is selected".
        renderCellDetail(null, "gone");
    } else {
        renderCellDetail(selected);
    }
    drawResults();
}

function togglePlayback() {
    if (state.playing) { stopPlayback(); return; }
    const frames = frameCount();
    if (frames < 2) { toast("Only one frame to show.", "warn"); return; }
    state.playing = true;
    $("pb-play").textContent = "⏸";
    const interval = Number($("pb-speed").value) || 250;
    state.playTimer = setInterval(() => {
        const next = state.frame + 1;
        if (next >= frameCount()) { stopPlayback(); return; }
        setFrame(next);
    }, interval);
}

function stopPlayback() {
    state.playing = false;
    if (state.playTimer) clearInterval(state.playTimer);
    state.playTimer = null;
    const button = $("pb-play");
    if (button) button.textContent = "▶";
}

function snapshotFrame() {
    const canvas = $("results-canvas");
    if (!canvas) { toast("No results view to snapshot.", "warn"); return; }
    canvas.toBlob((blob) => {
        // toBlob hands back null when the encode fails. Reading .type off null threw
        // inside this callback, where nothing catches it, so the user saw silence.
        if (!blob) { toast("Could not encode the frame as a PNG.", "error"); return; }
        const url = URL.createObjectURL(blob);
        const link = el("a");
        link.href = url;
        link.download = `frame_${state.frame}.png`;
        link.style.display = "none";
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
}

async function exportResults(kind) {
    if (!state.run) { toast("No run selected.", "warn"); return; }
    const runId = state.run.run_id;
    try {
        if (kind === "metadata") {
            const payload = await api(`/api/runs/${encodeURIComponent(runId)}/export/metadata`);
            // JSON.stringify(undefined) is the string "undefined", so an empty
            // response used to produce a 9-byte file literally containing
            // "undefined" -- and then report success.
            if (!payload) {
                toast("The server returned no metadata for that run.", "error", 8000);
                return;
            }
            download(`${runId}_metadata.json`, JSON.stringify(payload, null, 2));
        } else {
            const payload = await api(`/api/runs/${encodeURIComponent(runId)}/export/${kind}`);
            // Same failure mode: new Blob([undefined]) stringifies to "undefined" and
            // link.download = undefined names the file "undefined".
            if (!payload || typeof payload.csv !== "string" || !payload.csv.length
                || !payload.filename) {
                toast(`The server returned no CSV for the ${kind} export.`, "error", 8000);
                return;
            }
            download(payload.filename, payload.csv, "text/csv");
        }
        toast("Export downloaded.", "success");
    } catch (error) {
        toast(`Export failed: ${error.message}`, "error", 8000);
    }
}


/* --------------------------------------------------------------------------
 * Project import / export
 * ----------------------------------------------------------------------- */
async function exportProject() {
    try {
        const payload = await post("/api/project/export", { project: state.project });
        Object.entries(payload.files).forEach(([name, contents]) => {
            download(name.replace(/\//g, "_"), contents,
                     name.endsWith(".csv") ? "text/csv" : "application/json");
        });
        toast(`Exported ${Object.keys(payload.files).length} file(s).`, "success");
    } catch (error) {
        toast(`Export failed: ${error.message}`, "error");
    }
}

async function importProject(file) {
    let text;
    try {
        text = await file.text();
    } catch (error) {
        toast(`Could not read that file: ${error.message}`, "error");
        return;
    }
    try {
        const payload = await post("/api/project/import", { text });
        state.project = payload.project;
        if ((payload.migrations_applied || []).length) {
            // Say what changed rather than silently reinterpreting an old document.
            toast(`Project migrated: ${payload.migrations_applied.join("; ")}`, "warn", 11000);
        }
        syncControlsFromProject();
        state.validation = payload.validation;
        paintStageBadges(payload.validation);
        toast("Project imported.", "success");
        showStage("domain");
    } catch (error) {
        toast(`Import failed: ${error.message}`, "error", 10000);
    }
}

/**
 * Re-quotes every label that carries a unit but is NOT rebuilt by a render call
 * -- static HTML captions, and the two results labels that only redraw on a
 * frame change. Everything else is relabelled by re-rendering its own panel.
 */
function refreshUnitLabels() {
    const spacing = $("mesh-spacing-label");
    if (spacing) {
        // Target spacing is an element edge length, so it is in the length unit.
        const length = units();
        spacing.textContent = length
            ? `Target spacing (${length}, optional)` : "Target spacing (optional)";
    }
}

/**
 * Everything downstream of a unit change: the two base units are quoted in
 * labels that four different render paths own, so each is asked to redraw rather
 * than the labels being patched individually.
 */
function applyUnitChange() {
    // The unit selects are live during boot's async stretch, before applyDomain has
    // created the document. Relabel what exists and leave the rest to boot rather
    // than dereferencing a null project.
    if (!state.project) {
        renderDomainControls();
        refreshUnitLabels();
        return;
    }
    renderDomainControls();
    refreshUnitLabels();
    renderMeshStats(state.project.mesh);
    renderModelControls();
    renderConditionTable();
    renderConditionParams();
    renderApproachConfig();
    drawResults();
}

/** Push a loaded project back into every control. */
function syncControlsFromProject() {
    const project = state.project;
    if (!project) return;
    $("domain-kind").value = project.domain.kind;
    $("domain-units").value = project.domain.units || "um";
    // An imported document may predate the time unit, or carry one this build does
    // not offer. Fall back to seconds rather than leaving the select on whatever
    // the previous project happened to choose.
    const timeSelect = $("domain-time-units");
    const importedTime = sanitiseUnit(project.domain.time_units || "");
    const offered = Array.from(timeSelect.options).some((option) => option.value === importedTime);
    timeSelect.value = offered ? importedTime : "s";
    if (!offered) {
        // Write the fallback back onto the document so the screen and the value
        // that would RUN cannot disagree about the time unit.
        project.domain.time_units = timeSelect.value;
    }
    renderDomainControls();
    refreshUnitLabels();
    renderBoundaryChips();
    renderRegionChips();
    updateMeshControlsForDomain();

    const settings = project.mesh_settings || {};
    if (settings.element_count) $("mesh-elements").value = settings.element_count;
    if (settings.rows) $("mesh-rows").value = settings.rows;
    if (settings.cols) $("mesh-cols").value = settings.cols;
    if (settings.target_spacing) $("mesh-spacing").value = settings.target_spacing;
    renderMeshStats(project.mesh);

    renderModelControls();
    renderConditionTable();
    renderApproachCards();
    renderApproachConfig();
    refreshTopologyViews();
    fitView();
}

/* --------------------------------------------------------------------------
 * Wiring
 * ----------------------------------------------------------------------- */
function bindControls() {
    // Stage 1
    on("domain-kind", "change", renderDomainControls);
    // Both base units feed labels on stages 3, 4, 5, 7 and 10, so a change to
    // either redraws all of them rather than only the domain dimension captions.
    on("domain-units", "change", applyUnitChange);
    on("domain-time-units", "change", () => {
        // Committed to the document immediately. Waiting for "Apply domain" would
        // have left the labels quoting a unit the project did not yet carry, so an
        // export in between would have disagreed with the screen.
        if (state.project && state.project.domain) {
            state.project.domain.time_units = $("domain-time-units").value;
        }
        applyUnitChange();
    });
    on("domain-apply", "click", applyDomain);
    on("canvas-fit", "click", fitView);
    on("region-add", "click", addRegion);

    // Stage 2
    document.querySelectorAll(".wf-tool").forEach((button) => {
        button.addEventListener("click", () => {
            state.tool = button.dataset.tool;
            document.querySelectorAll(".wf-tool").forEach((other) =>
                other.classList.toggle("active", other === button));
            drawCanvas();
        });
    });
    document.querySelectorAll(".wf-view").forEach((button) => {
        button.addEventListener("click", () => {
            state.view = button.dataset.view;
            document.querySelectorAll(".wf-view").forEach((other) =>
                other.classList.toggle("active", other === button));
            drawCanvas();
        });
    });
    on("topo-make-cell", "click", makeCellFromSelection);
    on("topo-find-loops", "click", findLoops);
    on("topo-delete", "click", deleteSelected);
    on("topo-clear", "click", clearTopology);
    on("topo-export", "click", exportTopology);

    // Stage 3
    on("mesh-generate", "click", generateMesh);
    on("mesh-clear", "click", clearMesh);

    // Stage 4
    on("model-preset", "change", (event) => applyPreset(event.target.value));
    on("param-add", "click", () => {
        const name = $("param-name").value.trim();
        const value = Number($("param-value").value);
        if (!name) { toast("Give the parameter a name.", "warn"); return; }
        if (!Number.isFinite(value)) { toast("Give the parameter a numeric value.", "warn"); return; }
        state.project.model.parameters[name] = value;
        $("param-name").value = "";
        $("param-value").value = "";
        renderModelControls();
        validateModel();
    });
    on("field-add", "click", addField);
    on("model-solve1d", "click", solve1DNow);

    // Stage 5
    on("bc-type", "change", renderConditionParams);
    // A boundary value is in the SELECTED field's units, so changing the field has
    // to re-quote the value boxes -- otherwise they keep advertising the units of
    // the field that happened to be selected first.
    on("bc-field", "change", renderConditionParams);
    on("bc-add", "click", addCondition);

    // Stage 6/7
    on("approach-export", "click", exportApproachConfiguration);
    on("approach-export-cc3d", "click", exportCc3dProject);
    on("results-canvas", "click", handleResultsCanvasClick);

    // Stage 8/9
    on("run-validate", "click", validateEverything);
    on("run-start", "click", startRun);
    on("run-pause", "click", () => controlRun("pause"));
    on("run-resume", "click", () => controlRun("resume"));
    on("run-cancel", "click", () => controlRun("cancel"));

    // Stage 10
    on("pb-first", "click", () => { stopPlayback(); setFrame(0); });
    on("pb-prev", "click", () => { stopPlayback(); setFrame(state.frame - 1); });
    on("pb-play", "click", togglePlayback);
    on("pb-next", "click", () => { stopPlayback(); setFrame(state.frame + 1); });
    on("pb-last", "click", () => { stopPlayback(); setFrame(frameCount() - 1); });
    on("pb-scrub", "input", (event) => {
        stopPlayback();
        setFrame(Number(event.target.value));
    });
    on("pb-speed", "change", () => {
        if (state.playing) { stopPlayback(); togglePlayback(); }
    });
    ["ov-grid", "ov-ids", "ov-centroids"].forEach((id) =>
        on(id, "change", drawResults));
    on("results-filter", "input", renderCellList);
    on("results-variable", "change", drawChart);
    on("results-snapshot", "click", snapshotFrame);
    on("results-export-cells", "click", () => exportResults("cells"));
    on("results-export-series", "click", () => exportResults("timeseries"));
    on("results-export-meta", "click", () => exportResults("metadata"));

    // Header
    on("wf-export", "click", exportProject);
    on("wf-import", "click", () => $("wf-import-file").click());
    on("wf-import-file", "change", (event) => {
        const file = event.target.files && event.target.files[0];
        if (file) importProject(file);
        event.target.value = "";
    });

    // Keyboard: arrow keys scrub the timeline when the results stage is open.
    document.addEventListener("keydown", (event) => {
        if (state.stage !== "results") return;
        if (event.target && /INPUT|SELECT|TEXTAREA/.test(event.target.tagName)) return;
        if (event.key === "ArrowRight") { stopPlayback(); setFrame(state.frame + 1); }
        else if (event.key === "ArrowLeft") { stopPlayback(); setFrame(state.frame - 1); }
        else if (event.key === " ") { event.preventDefault(); togglePlayback(); }
    });
}

/* --------------------------------------------------------------------------
 * Boot
 * ----------------------------------------------------------------------- */
async function boot() {
    buildStageRail();
    bindControls();
    bindCanvasCamera();
    renderDomainControls();
    refreshUnitLabels();
    showStage("domain");
    setStatus("Loading…", "yellow");

    await loadPresetList();
    await loadApproaches();

    // Create the starting project from the default rectangle so every stage has a
    // real document to work against rather than a null the UI has to guard.
    const created = await applyDomain();
    if (created) {
        renderModelControls();
        await validateModel();
        await validateTopology();
        await validateConditions();
        await refreshRunList();
        setStatus("Ready", "green");
    }
    drawResults();
}

document.addEventListener("DOMContentLoaded", boot);
