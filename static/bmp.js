// ==========================================================================
// EMBRYO BMP TAB - Umulis et al. (2010), Developmental Cell 18:260-274
//
// Reads the precomputed solutions in /data/bmp/cache.json (rebuilt by
// verify_bmp_umulis2010.py), the validation report from /api/bmp/validation, and the
// equations / parameter sources from /api/bmp/info. "Solve cross-section" runs the real
// model on the server. Everything drawn here is solver output or a digitised paper value,
// and each is labelled as such.
// ==========================================================================
(function () {
    "use strict";

    const BMP = {
        initialised: false, cache: null, info: null, validation: null,
        charts: {}, view: "both", playTimer: null, live: null,
    };
    const PERT_LABELS = {
        wt: "wild type", sog_het: "sog +/−", tsg_het: "tsg +/−", scw_het: "scw +/−", dpp_het: "dpp +/−",
        sog_null: "sog −/−", tsg_null: "tsg −/−", tld_null: "tld −/−", invitro_k3: "in-vitro Sog/BMP on-rate",
    };
    const KIND_LABELS = {
        solver_verification: "Solver check", diagnostic: "Diagnostic", paper_figure: "Figure match",
        paper_claim: "Paper claim", reproduction_gap: "Known gap",
    };
    const TIME_COLORS = { 15: "#38bdf8", 30: "#00f2fe", 45: "#a78bfa", 60: "#ff007f" };
    const PERT_COLORS = {
        wt: "#00f2fe", sog_het: "#a78bfa", sog_null: "#fbbf24", tsg_het: "#34d399", tsg_null: "#f472b6",
        tld_null: "#ff007f", scw_het: "#60a5fa", dpp_het: "#f97316", invitro_k3: "#94a3b8",
    };

    document.addEventListener("biosim:tab", (event) => {
        if (event.detail && event.detail.tab === "embryo") init();
    });

    function $(id) { return document.getElementById(id); }
    function node(tag, cls, text) {
        const n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text != null) n.textContent = String(text);
        return n;
    }
    function fmt(x, digits = 3) {
        const v = Number(x);
        if (!Number.isFinite(v)) return String(x);
        if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5)) return v.toExponential(2);
        return String(Number(v.toPrecision(digits)));
    }
    async function getJson(url, options) {
        if (typeof apiJson === "function") return apiJson(url, options);
        const r = await fetch(url, options);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
    }

    async function init() {
        if (BMP.initialised) { drawSurface(); return; }
        BMP.initialised = true;
        const summary = $("embryo-summary");
        try {
            const [cache, info] = await Promise.all([
                getJson("/data/bmp/cache.json"), getJson("/api/bmp/info"),
            ]);
            BMP.cache = cache;
            BMP.info = info;
        } catch (e) {
            if (summary) summary.innerHTML = "";
            if (summary) summary.appendChild(node("p", "placeholder-text err", "Could not load the embryo model: " + e.message));
            BMP.initialised = false;
            return;
        }
        buildSurfaceControls();
        buildRunControls();
        drawSurface();
        drawCrossSectionChart();
        drawMutantChart();
        renderDocs();
        loadValidation(false);
    }

    // ------------------------------------------------------------------ surface
    function buildSurfaceControls() {
        const sel = $("embryo-surface-pert");
        sel.innerHTML = "";
        Object.keys(BMP.cache.surface || {}).forEach(k => {
            const o = node("option", "", PERT_LABELS[k] || k);
            o.value = k;
            sel.appendChild(o);
        });
        sel.value = "wt";
        sel.addEventListener("change", drawSurface);
        const slider = $("embryo-time");
        slider.addEventListener("input", () => { stopPlay(); drawSurface(); });
        $("embryo-play").addEventListener("click", togglePlay);
        document.querySelectorAll("#embryo-view-toggle .seg-btn").forEach(b => b.addEventListener("click", () => {
            document.querySelectorAll("#embryo-view-toggle .seg-btn").forEach(x => x.classList.remove("active"));
            b.classList.add("active");
            BMP.view = b.getAttribute("data-view");
            drawSurface();
        }));
        window.addEventListener("resize", () => { if ($("tab-embryo").classList.contains("active")) drawSurface(); });
    }

    function togglePlay() {
        if (BMP.playTimer) { stopPlay(); return; }
        const slider = $("embryo-time");
        if (Number(slider.value) >= 60) slider.value = "0";
        $("embryo-play").textContent = "⏸ Pause";
        BMP.playTimer = setInterval(() => {
            const next = Number(slider.value) + 0.5;
            slider.value = String(Math.min(60, next));
            drawSurface();
            if (next >= 60) stopPlay();
        }, 60);
    }
    function stopPlay() {
        if (BMP.playTimer) clearInterval(BMP.playTimer);
        BMP.playTimer = null;
        const b = $("embryo-play");
        if (b) b.textContent = "▶ Play";
    }

    // Perceptually ordered dark->bright map ("magma"-like), matching the video palette.
    const STOPS = [[0, [0, 0, 4]], [0.25, [59, 15, 112]], [0.5, [140, 41, 129]], [0.75, [222, 73, 104]], [1, [252, 253, 191]]];
    function colormap(t) {
        const x = Math.max(0, Math.min(1, t));
        for (let i = 1; i < STOPS.length; i++) {
            if (x <= STOPS[i][0]) {
                const [t0, c0] = STOPS[i - 1], [t1, c1] = STOPS[i];
                const f = (x - t0) / (t1 - t0);
                return `rgb(${Math.round(c0[0] + f * (c1[0] - c0[0]))},${Math.round(c0[1] + f * (c1[1] - c0[1]))},${Math.round(c0[2] + f * (c1[2] - c0[2]))})`;
            }
        }
        return "rgb(252,253,191)";
    }

    function frameAt(run, minutes) {
        const times = run.times;
        let k = times.findIndex(t => t >= minutes);
        if (k < 0) k = times.length - 1;
        if (k === 0 || times[k] === minutes) return run.BR[k];
        const f = (minutes - times[k - 1]) / (times[k] - times[k - 1]);
        const a = run.BR[k - 1], b = run.BR[k];
        return a.map((row, i) => row.map((v, j) => v + f * (b[i][j] - v)));
    }

    function drawSurface() {
        const canvas = $("embryo-surface-canvas");
        if (!canvas || !BMP.cache) return;
        const wrap = canvas.parentElement;
        const cssW = Math.max(320, wrap.clientWidth || 900);
        const cssH = Math.round(cssW * (BMP.view === "both" ? 0.46 : 0.52));
        const dpr = window.devicePixelRatio || 1;
        canvas.style.width = cssW + "px";
        canvas.style.height = cssH + "px";
        canvas.width = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);
        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = "#0b0f1a";
        ctx.fillRect(0, 0, cssW, cssH);

        const pert = $("embryo-surface-pert").value || "wt";
        const run = BMP.cache.surface[pert];
        const g = BMP.cache.surface_grid;
        const minutes = Number($("embryo-time").value);
        $("embryo-time-label").textContent = `${minutes.toFixed(1)} min`;
        const field = frameAt(run, minutes);
        const wt = BMP.cache.surface.wt;
        let vmax = 0;
        wt.BR.forEach(frame => frame.forEach(row => row.forEach(v => { if (v > vmax) vmax = v; })));
        vmax = vmax || 1;

        const views = BMP.view === "both" ? ["dorsal", "lateral"] : [BMP.view];
        const panelW = (cssW - 90) / views.length;
        views.forEach((view, idx) => {
            const ox = 20 + idx * panelW;
            drawView(ctx, g, field, vmax, view, ox, 28, panelW - 20, cssH - 70);
        });
        drawColorbar(ctx, cssW - 58, 36, 14, cssH - 90, vmax);

        const dm = run.dorsal_midline || {};
        const note = $("embryo-surface-note");
        if (note) {
            note.textContent = `${PERT_LABELS[pert] || pert}: solver output on a ${g.nu}×${g.nv} surface grid (400 × 180 µm prolate spheroid). `
                + `Colour scale fixed to the wild-type maximum (${fmt(vmax)} nM) so genotypes compare directly. `
                + (dm.min_max_ratio != null ? `At 60 min the dorsal-midline signal varies along the AP axis (min/max ${fmt(dm.min_max_ratio)}).` : "");
        }
    }

    function drawView(ctx, g, field, vmax, view, ox, oy, w, h) {
        const a = g.a_um, b = g.b_um, nu = g.nu, nv = g.nv;
        const du = Math.PI / nu, dv = Math.PI / nv;
        const scale = Math.min(w / (2.45 * a), h / (2.35 * b));
        const cx = ox + w / 2, cy = oy + h / 2;
        // Anterior (u = 0, x = +a) is drawn on the LEFT; dorsal is UP in the lateral view.
        const P = (u, v, mirror) => {
            const x = a * Math.cos(u), y = b * Math.sin(u) * Math.cos(v), z = b * Math.sin(u) * Math.sin(v);
            if (view === "lateral") return [cx - x * scale, cy - y * scale, z];
            return [cx - x * scale, cy + (mirror ? -z : z) * scale, y];
        };
        const quads = [];
        for (let i = 0; i < nu; i++) {
            const u0 = i * du, u1 = (i + 1) * du;
            for (let j = 0; j < nv; j++) {
                const v0 = j * dv, v1 = (j + 1) * dv;
                const vm = (v0 + v1) / 2;
                const value = field[i][j] / vmax;
                const mirrors = view === "dorsal" ? [false, true] : [false];
                if (view === "dorsal" && Math.cos(vm) <= 0) continue;      // ventral half is hidden
                mirrors.forEach(m => {
                    const pts = [P(u0, v0, m), P(u1, v0, m), P(u1, v1, m), P(u0, v1, m)];
                    const depth = pts.reduce((s, p) => s + p[2], 0) / 4;
                    quads.push({ pts, depth, value });
                });
            }
        }
        quads.sort((p, q) => p.depth - q.depth);
        quads.forEach(q => {
            ctx.beginPath();
            ctx.moveTo(q.pts[0][0], q.pts[0][1]);
            for (let k = 1; k < 4; k++) ctx.lineTo(q.pts[k][0], q.pts[k][1]);
            ctx.closePath();
            ctx.fillStyle = colormap(q.value);
            ctx.fill();
            ctx.strokeStyle = colormap(q.value);
            ctx.lineWidth = 0.6;
            ctx.stroke();
        });
        ctx.strokeStyle = "rgba(0,242,254,0.55)";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.ellipse(cx, cy, a * scale, b * scale, 0, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.fillStyle = "#e5e7eb";
        ctx.font = "600 13px Outfit, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(view === "dorsal" ? "Dorsal view" : "Lateral view (dorsal up)", cx, oy + 4);
        ctx.font = "12px Outfit, sans-serif";
        ctx.fillStyle = "#9ca3af";
        ctx.fillText("A", cx - a * scale - 12, cy + 4);
        ctx.fillText("P", cx + a * scale + 12, cy + 4);
        if (view === "lateral") {
            ctx.fillText("D", cx, cy - b * scale - 8);
            ctx.fillText("V", cx, cy + b * scale + 16);
        } else {
            ctx.fillText("dorsal midline", cx, cy + b * scale + 16);
        }
    }

    function drawColorbar(ctx, x, y, w, h, vmax) {
        for (let k = 0; k < h; k++) {
            ctx.fillStyle = colormap(1 - k / h);
            ctx.fillRect(x, y + k, w, 1);
        }
        ctx.strokeStyle = "#374151";
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = "#9ca3af";
        ctx.font = "11px Outfit, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(`${fmt(vmax)} nM`, x - 4, y - 8);
        ctx.fillText("0", x + w + 4, y + h);
        ctx.save();
        ctx.translate(x + w + 16, y + h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.fillText("BMP-bound receptor (BR)", 0, 0);
        ctx.restore();
    }

    // ------------------------------------------------------------------ charts
    function chartDefaults() {
        return {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: "nearest", intersect: false },
            plugins: { legend: { labels: { color: "#e5e7eb", boxWidth: 14, font: { size: 11 } } } },
            scales: {
                x: { type: "linear", title: { display: true, text: "distance from the dorsal midline (µm)", color: "#9ca3af" },
                     ticks: { color: "#9ca3af" }, grid: { color: "rgba(148,163,184,0.12)" } },
                y: { beginAtZero: true, title: { display: true, text: "BR (nM)", color: "#9ca3af" },
                     ticks: { color: "#9ca3af" }, grid: { color: "rgba(148,163,184,0.12)" } },
            },
        };
    }
    function series(s, values, maxX) {
        const out = [];
        s.forEach((x, i) => { if (maxX == null || x <= maxX) out.push({ x, y: values[i] }); });
        return out;
    }
    function destroyChart(key) {
        if (BMP.charts[key]) { try { BMP.charts[key].destroy(); } catch { /* ignore */ } }
        BMP.charts[key] = null;
    }

    function drawCrossSectionChart() {
        if (typeof Chart === "undefined") return;
        destroyChart("cs");
        const run = BMP.cache.cross_section["sbp/wt"];
        const datasets = [];
        [15, 30, 45, 60].forEach(t => {
            const k = run.times.indexOf(t);
            if (k < 0) return;
            datasets.push({ label: `${t} min (model)`, data: series(run.s_um, run.BR[k], 180), borderColor: TIME_COLORS[t],
                            backgroundColor: TIME_COLORS[t], pointRadius: 0, borderWidth: 2.2, showLine: true });
        });
        const paper = BMP.cache.paper_data.umulis2010_fig4F;
        datasets.push({ label: "paper Fig. 4F, dorsal midline", type: "scatter",
                        data: paper.dm_BR_nM.map(y => ({ x: 0, y })), backgroundColor: "#fbbf24",
                        borderColor: "#0b0f1a", pointRadius: 6, pointHoverRadius: 7 });
        if (BMP.live) {
            const L = BMP.live;
            [15, 30, 45, 60].forEach(t => {
                const k = L.times.indexOf(t);
                if (k < 0) return;
                datasets.push({ label: `${t} min (your run)`, data: series(L.grid.s_um, L.fields.BR[k], 180),
                                borderColor: TIME_COLORS[t], borderDash: [6, 4], pointRadius: 0, borderWidth: 1.8, showLine: true });
            });
            if (L.parameter_set === "umulis2006") {
                const p06 = BMP.cache.paper_data.umulis2006_fig13;
                datasets.push({ label: "2006 paper Fig. 13, dorsal midline", type: "scatter",
                                data: p06.dm_BR_nM.slice(0, 4).map(y => ({ x: 0, y })), backgroundColor: "#34d399",
                                borderColor: "#0b0f1a", pointRadius: 6 });
            }
        }
        const opts = chartDefaults();
        opts.scales.x.max = 180;
        BMP.charts.cs = new Chart($("embryo-cs-chart"), { type: "line", data: { datasets }, options: opts });
    }

    function drawMutantChart() {
        if (typeof Chart === "undefined") return;
        destroyChart("mut");
        const datasets = [];
        ["wt", "sog_het", "sog_null", "tsg_het", "scw_het", "dpp_het", "tld_null", "invitro_k3"].forEach(p => {
            const run = BMP.cache.cross_section[`sbp/${p}`];
            if (!run) return;
            const k = run.times.indexOf(60);
            datasets.push({ label: PERT_LABELS[p], data: series(run.s_um, run.BR[k >= 0 ? k : run.times.length - 1]),
                            borderColor: PERT_COLORS[p], backgroundColor: PERT_COLORS[p], pointRadius: 0,
                            borderWidth: p === "wt" ? 3 : 1.8, showLine: true });
        });
        BMP.charts.mut = new Chart($("embryo-mutant-chart"), { type: "line", data: { datasets }, options: chartDefaults() });
    }

    // ------------------------------------------------------------------ live runs
    const SLIDERS = [
        { key: "phi_S", label: "Sog secretion φS", units: "nM/min" },
        { key: "phi_B", label: "BMP secretion φB", units: "nM/min" },
        { key: "phi_T", label: "Tsg secretion φT", units: "nM/min" },
        { key: "lambda_Tld", label: "Tld processing λ·Tld", units: "1/min" },
        { key: "Lambda", label: "Feedback amplitude Λ", units: "" },
        { key: "Rtot", label: "Total receptor Rtot", units: "nM" },
    ];

    function defaultValue(key, mech, pset) {
        const T = BMP.info.parameter_table;
        const pick = (...names) => { for (const n of names) if (T[n]) return Number(T[n].value); return null; };
        if (pset === "umulis2006") return pick(`umulis2006.${key}`);
        if (key === "phi_S" && pset === "umulis2010_as_printed") return pick("phi_S_as_printed");
        if (pset === "umulis2010_ellipse" && (key === "Rtot" || key === "lambda_Tld")) return pick(`ellipse.${key}`);
        if (key === "Rtot" && mech === "receptor") return pick("receptor.R_basal");
        return pick(`${mech}.${key}`, key);
    }

    function buildRunControls() {
        const pertSel = $("embryo-pert");
        pertSel.innerHTML = "";
        Object.keys(BMP.info.perturbations).forEach(k => {
            const o = node("option", "", PERT_LABELS[k] || k);
            o.value = k;
            o.title = BMP.info.perturbations[k].description;
            pertSel.appendChild(o);
        });
        const refresh = () => {
            const pset = $("embryo-pset").value;
            const sbpOnly = pset === "umulis2006" || pset === "umulis2010_ellipse";
            if (sbpOnly) $("embryo-mech").value = "sbp";
            $("embryo-mech").disabled = sbpOnly;
            renderSliders();
        };
        ["embryo-mech", "embryo-pset"].forEach(id => $(id).addEventListener("change", refresh));
        $("embryo-run").addEventListener("click", runLive);
        $("embryo-reset").addEventListener("click", () => {
            $("embryo-mech").value = "sbp"; $("embryo-pert").value = "wt"; $("embryo-pset").value = "umulis2010";
            BMP.live = null; refresh(); drawCrossSectionChart(); $("embryo-run-readout").innerHTML = "";
        });
        refresh();
    }

    function renderSliders() {
        const box = $("embryo-sliders");
        box.innerHTML = "";
        const mech = $("embryo-mech").value, pset = $("embryo-pset").value;
        SLIDERS.forEach(s => {
            const base = defaultValue(s.key, mech, pset);
            if (base == null || (s.key === "Lambda" && mech === "none")) return;
            const row = node("div", "embryo-slider");
            const label = node("label", "", `${s.label}`);
            const input = document.createElement("input");
            input.type = "range"; input.min = "-2"; input.max = "2"; input.step = "0.05"; input.value = "0";
            input.dataset.key = s.key; input.dataset.base = String(base);
            input.setAttribute("aria-label", s.label);
            const value = node("span", "embryo-slider__value", `${fmt(base)} ${s.units}`);
            input.addEventListener("input", () => {
                const v = base * Math.pow(2, Number(input.value));
                value.textContent = `${fmt(v)} ${s.units}` + (Number(input.value) ? `  (×${fmt(Math.pow(2, Number(input.value)), 2)})` : "");
            });
            row.appendChild(label); row.appendChild(input); row.appendChild(value);
            box.appendChild(row);
        });
        const note = node("p", "hint-text", "Sliders scale the paper value by ¼× to 4×. The genotype is applied on top (e.g. sog +/− halves φS).");
        box.appendChild(note);
    }

    async function runLive() {
        const btn = $("embryo-run");
        const mech = $("embryo-mech").value, pert = $("embryo-pert").value, pset = $("embryo-pset").value;
        const overrides = {};
        const multipliers = (BMP.info.perturbations[pert] || {}).multipliers || {};
        document.querySelectorAll("#embryo-sliders input[type=range]").forEach(inp => {
            const m = Math.pow(2, Number(inp.value));
            if (m !== 1) {
                const key = inp.dataset.key === "Rtot" && mech === "receptor" ? "R_basal" : inp.dataset.key;
                overrides[key] = Number(inp.dataset.base) * m * (multipliers[inp.dataset.key] != null ? multipliers[inp.dataset.key] : 1);
            }
        });
        const body = { mechanism: mech, perturbation: pert, parameter_set: pset, overrides,
                       t_end: 60, save_times: [0, 15, 30, 45, 60], species: ["BR", "B", "S", "IB"] };
        const readout = $("embryo-run-readout");
        btn.disabled = true; btn.textContent = "Solving…";
        try {
            const res = await getJson("/api/bmp/cross-section", {
                method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
            });
            BMP.live = res;
            drawCrossSectionChart();
            const f = res.readouts.final, d = res.diagnostics;
            readout.innerHTML = "";
            const rows = [
                ["Dorsal-midline BR at 60 min", `${fmt(f.dm_value)} nM`],
                ["High-signal width (FWHM)", `${fmt(f.fwhm_percent_circumference)}% of the circumference`],
                ["Peak position", `${fmt(f.peak_location_um_from_dm)} µm from the dorsal midline`],
                ["Ligand mass balance", `relative error ${fmt(d.ligand_mass_balance_relative_error)}`],
                ["Solve time", `${fmt(d.wall_time_s)} s (BDF, analytic sparse Jacobian)`],
            ];
            rows.forEach(([k, v]) => { const r = node("div", "readout-row"); r.appendChild(node("span", "", k)); r.appendChild(node("strong", "", v)); readout.appendChild(r); });
        } catch (e) {
            readout.innerHTML = "";
            readout.appendChild(node("p", "placeholder-text err", "The run failed: " + e.message));
        } finally {
            btn.disabled = false; btn.textContent = "Solve cross-section";
        }
    }

    // ------------------------------------------------------------------ validation
    async function loadValidation(fresh) {
        const table = $("embryo-validation-table");
        const source = $("embryo-validation-source");
        const btn = $("embryo-revalidate");
        if (fresh) { btn.disabled = true; btn.textContent = "Running…"; }
        try {
            const report = await getJson(`/api/bmp/validation${fresh ? "?fresh=true" : ""}`);
            BMP.validation = report;
            source.textContent = `${report.passed} of ${report.total} checks pass · ${report.source || ""}. `
                + "Nothing below was tuned to pass; failures are kept and explained.";
            renderValidation(report);
            renderSummary(report);
        } catch (e) {
            source.textContent = "Could not load the validation report: " + e.message;
        } finally {
            if (!btn.dataset.wired) { btn.addEventListener("click", () => loadValidation(true)); btn.dataset.wired = "1"; }
            btn.disabled = false; btn.textContent = "Re-run checks live";
        }
    }

    function measuredText(measured) {
        const parts = [];
        const add = (k, v) => {
            if (v == null) return;
            if (typeof v === "number") parts.push(`${k} = ${fmt(v)}`);
            else if (typeof v === "boolean") parts.push(`${k} = ${v}`);
            else if (Array.isArray(v)) parts.push(`${k} = [${v.map(x => fmt(x)).join(", ")}]`);
            else if (typeof v === "object") parts.push(`${k}: ` + Object.entries(v).map(([a, b]) => `${a}→${fmt(b)}`).join(", "));
            else parts.push(`${k}: ${v}`);
        };
        Object.entries(measured || {}).forEach(([k, v]) => add(k, v));
        return parts.join(" · ");
    }

    function renderValidation(report) {
        const table = $("embryo-validation-table");
        table.innerHTML = "";
        const head = document.createElement("thead");
        const hr = document.createElement("tr");
        ["", "Check", "What the paper says / how it was measured", "Measured"].forEach(h => hr.appendChild(node("th", "", h)));
        head.appendChild(hr);
        table.appendChild(head);
        const body = document.createElement("tbody");
        report.checks.forEach(row => {
            const tr = document.createElement("tr");
            tr.className = row.pass === true ? "pass" : row.pass === false ? (row.kind === "reproduction_gap" ? "gap" : "fail") : "";
            const badge = node("td", "result");
            badge.appendChild(node("span", "result-badge", row.pass === true ? "PASS" : row.pass === false ? "FAIL" : "—"));
            tr.appendChild(badge);
            const id = node("td", "check-id");
            id.appendChild(node("strong", "", row.id));
            id.appendChild(node("span", "kind", KIND_LABELS[row.kind] || row.kind));
            tr.appendChild(id);
            const claim = node("td", "");
            claim.appendChild(node("div", "claim", row.claim));
            claim.appendChild(node("div", "cite", row.citation));
            claim.appendChild(node("div", "how", row.how_measured));
            tr.appendChild(claim);
            const m = node("td", "measured", measuredText(row.measured));
            tr.appendChild(m);
            body.appendChild(tr);
        });
        table.appendChild(body);
    }

    function renderSummary(report) {
        const box = $("embryo-summary");
        if (!box) return;
        box.innerHTML = "";
        const by = {};
        report.checks.forEach(r => { by[r.id] = r; });
        const stat = (value, label, cls) => {
            const d = node("div", "stat " + (cls || ""));
            d.appendChild(node("span", "stat__value", value));
            d.appendChild(node("span", "stat__label", label));
            box.appendChild(d);
        };
        stat(`${report.passed}/${report.total}`, "checks against the papers pass", report.passed === report.total ? "ok" : "");
        if (by.S1) {
            const errs = Object.values(by.S1.measured.relative_error_BR || {}).map(Math.abs);
            if (errs.length) stat(`≤ ${fmt(100 * Math.max(...errs), 2)}%`, "solver error vs the fully specified 2006 model (30 min → steady state)", by.S1.pass ? "ok" : "bad");
        }
        if (by.V2) stat(`${fmt(by.V2.measured.fwhm_percent_circumference, 2)}%`, "of the circumference is high-signal at 60 min (paper ≈ 10%)", by.V2.pass ? "ok" : "bad");
        if (by.V4) stat(fmt(by.V4.measured.peak_ratio, 2), "tld −/− peak ÷ wild-type peak", by.V4.pass ? "ok" : "bad");
    }

    // ------------------------------------------------------------------ docs
    function renderDocs() {
        const eqBox = $("embryo-equations");
        eqBox.innerHTML = "";
        Object.entries(BMP.info.equations_latex || {}).forEach(([k, tex]) => {
            if (k.includes(".")) return;
            const row = node("div", "eq-row");
            try { katex.render(tex, row, { throwOnError: false, displayMode: true }); }
            catch { row.textContent = tex; }
            eqBox.appendChild(row);
        });
        const table = $("embryo-params");
        table.innerHTML = "";
        const head = document.createElement("thead");
        const hr = document.createElement("tr");
        ["Parameter", "Value", "Units", "Source"].forEach(h => hr.appendChild(node("th", "", h)));
        head.appendChild(hr);
        table.appendChild(head);
        const body = document.createElement("tbody");
        Object.entries(BMP.info.parameter_table || {}).forEach(([name, p]) => {
            if (name.startsWith("umulis2006.") || name.startsWith("prepattern.smooth")) return;
            const tr = document.createElement("tr");
            tr.appendChild(node("td", "mono", name));
            tr.appendChild(node("td", "mono", fmt(p.value, 4)));
            tr.appendChild(node("td", "", p.units));
            const src = node("td", "");
            src.appendChild(node("div", "", p.source));
            if (p.interpretation) src.appendChild(node("div", "muted", p.interpretation));
            tr.appendChild(src);
            body.appendChild(tr);
        });
        table.appendChild(body);
        const list = $("embryo-assumptions");
        list.innerHTML = "";
        (BMP.info.assumptions || []).forEach(a => list.appendChild(node("li", "", a)));
    }
})();
