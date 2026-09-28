// ==========================================================================
// GUIDES TAB - motion-graphic explainers rendered by tools/make_videos.py from the
// app's own outputs. Listed by static/videos/manifest.json.
// ==========================================================================
(function () {
    "use strict";
    let loaded = false;

    document.addEventListener("biosim:tab", (event) => {
        if (event.detail && event.detail.tab === "guides") load();
    });

    function node(tag, cls, text) {
        const n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text != null) n.textContent = String(text);
        return n;
    }

    async function load() {
        if (loaded) return;
        const grid = document.getElementById("guides-grid");
        if (!grid) return;
        try {
            const response = await fetch("/videos/manifest.json");
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const manifest = await response.json();
            grid.innerHTML = "";
            manifest.forEach(item => grid.appendChild(card(item)));
            loaded = true;
        } catch (e) {
            grid.innerHTML = "";
            grid.appendChild(node("p", "placeholder-text err", "Could not load the guides: " + e.message));
        }
    }

    function card(item) {
        const c = node("article", "card guide-card");
        const video = document.createElement("video");
        video.controls = true;
        video.preload = "metadata";
        video.playsInline = true;
        if (item.poster) video.poster = `/videos/${item.poster}`;
        video.setAttribute("aria-label", item.title);
        const source = document.createElement("source");
        source.src = `/videos/${item.file}`;
        source.type = "video/mp4";
        video.appendChild(source);
        if (item.captions) {
            const track = document.createElement("track");
            track.kind = "captions";
            track.src = `/videos/${item.captions}`;
            track.srclang = "en";
            track.label = "English";
            track.default = true;
            video.appendChild(track);
        }
        c.appendChild(video);
        const body = node("div", "guide-card__body");
        const title = node("h4", "guide-card__title", item.title);
        if (item.duration_s) title.appendChild(node("span", "guide-card__duration", `${Math.round(item.duration_s)} s`));
        body.appendChild(title);
        body.appendChild(node("p", "guide-card__desc", item.description));
        if (item.source_note) body.appendChild(node("p", "guide-card__source", "Rendered from: " + item.source_note));
        c.appendChild(body);
        return c;
    }
})();
