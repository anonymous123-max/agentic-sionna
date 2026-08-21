/**
 * pane-router.js — render typed result panes returned by /api/agent/step.
 *
 * Expected payload shape:
 *   {
 *     text: "...",
 *     layout: { cols: 3 } | { auto: true },
 *     panes: [
 *       { type: "text",      data: { markdown: "..." }, title: "Reply" },
 *       { type: "ber_curve", data: { snr_db: [...], ber: [...] }, title: "BER" },
 *       { type: "image",     data: { url: "/api/agent/file/..." }, title: "..." },
 *       { type: "scene",     data: { url: "/api/agent/file/...glb" } },
 *       { type: "table",     data: { rows: [...], columns: [...] } },
 *       { type: "code",      data: { src: "...", language: "py" } },
 *       { type: "json",      data: { ... } },
 *     ],
 *     files_produced: [...]
 *   }
 *
 * Dependencies: Plotly (loaded in dashboard.html), THREE + GLTFLoader
 * (importmap in dashboard.html). All other rendering is plain DOM.
 */

const PANE_RENDERERS = {
    text: renderText,
    image: renderImage,
    ber_curve: renderBerCurve,
    scene: renderScene,
    scene_iframe: renderSceneIframe,
    table: renderTable,
    code: renderCode,
    json: renderJson,
};


/** Render a self-contained scene viewer as an iframe.
 *  Used for agent-generated viewer.html (produced by template_viewer.py
 *  alongside scene_state.json), so the 3D scene shows up immediately
 *  without a separate GLB export step. */
function renderSceneIframe(body, pane) {
    const url = pane.data && pane.data.url;
    if (!url) { body.appendChild(errorMsg("No viewer URL")); return; }
    const frame = document.createElement("iframe");
    frame.src = url;
    frame.style.cssText = "width:100%;height:100%;min-height:420px;border:0;background:#050505;border-radius:8px;";
    frame.title = pane.title || "scene";
    frame.allow = "fullscreen";
    body.appendChild(frame);
}

/** Public entry: render the response into #panes-root. */
export function renderAgentResponse(payload) {
    const root = ensureRoot();
    const grid = root.querySelector(".panes-grid");
    grid.innerHTML = "";

    const panes = (payload && payload.panes) || [];
    if (panes.length === 0) {
        root.classList.remove("has-panes");
        return;
    }

    // Apply layout hint.
    const cols = payload && payload.layout && payload.layout.cols;
    if (cols && cols > 0) {
        grid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
    } else {
        grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(360px, 1fr))";
    }

    for (const pane of panes) {
        const el = buildPaneShell(pane);
        const body = el.querySelector(".pane-body");
        const renderer = PANE_RENDERERS[pane.type] || renderUnknown;
        try {
            renderer(body, pane);
        } catch (err) {
            body.innerHTML = "";
            body.appendChild(errorMsg(`Render error: ${err.message || err}`));
        }
        grid.appendChild(el);
    }

    root.classList.add("has-panes");
}

/** Clear all panes and hide the drawer. */
export function clearPanes() {
    const root = document.getElementById("panes-root");
    if (!root) return;
    const grid = root.querySelector(".panes-grid");
    if (grid) grid.innerHTML = "";
    root.classList.remove("has-panes");
}

// ─── Shell ────────────────────────────────────────────────────────────────

function ensureRoot() {
    let root = document.getElementById("panes-root");
    if (root && root.querySelector(".panes-grid")) return root;
    if (!root) {
        // Template should provide it, but be defensive.
        root = document.createElement("div");
        root.id = "panes-root";
        document.body.appendChild(root);
    }
    root.innerHTML = `
        <div class="panes-header">
            <span class="panes-title">Agent Output</span>
            <button class="panes-close" type="button">Dismiss</button>
        </div>
        <div class="panes-grid"></div>
    `;
    root.querySelector(".panes-close").addEventListener("click", clearPanes);
    return root;
}

function buildPaneShell(pane) {
    const el = document.createElement("div");
    el.className = "pane";
    el.dataset.paneType = pane.type;

    const head = document.createElement("div");
    head.className = "pane-head";
    head.innerHTML = `
        <span class="pane-title">${escapeHtml(pane.title || pane.type)}</span>
        <span class="pane-type-chip">${escapeHtml(pane.type)}</span>
    `;

    const body = document.createElement("div");
    body.className = "pane-body";

    el.appendChild(head);
    el.appendChild(body);
    return el;
}

// ─── Renderers ────────────────────────────────────────────────────────────

function renderText(body, pane) {
    const data = pane.data || {};
    const text = data.markdown || data.text || "";
    body.textContent = text;
}

function renderImage(body, pane) {
    const url = pane.data && pane.data.url;
    if (!url) {
        body.appendChild(errorMsg("No image URL"));
        return;
    }
    const img = document.createElement("img");
    img.src = url;
    img.alt = pane.title || "agent output";
    body.appendChild(img);
}

function renderBerCurve(body, pane) {
    if (typeof window.Plotly === "undefined") {
        body.appendChild(errorMsg("Plotly not loaded"));
        return;
    }
    const data = pane.data || {};
    // Accept either {snr_db: [], ber: []} or {ber_curve: [{snr,ber}, ...]}
    let x = data.snr_db || data.snr;
    let y = data.ber;
    if (!x && Array.isArray(data.ber_curve)) {
        x = data.ber_curve.map((p) => p.snr_db ?? p.snr);
        y = data.ber_curve.map((p) => p.ber);
    }
    if (!x || !y || x.length === 0 || y.length === 0) {
        // Fallback to JSON dump.
        renderJson(body, pane);
        return;
    }
    const trace = {
        x, y,
        mode: "lines+markers",
        line: { color: "#4CD964", width: 2 },
        marker: { color: "#E6E0D4", size: 5 },
        name: pane.title || "BER",
    };
    const layout = {
        paper_bgcolor: "#111111",
        plot_bgcolor: "#050505",
        font: { family: "Inter, sans-serif", color: "#9E9E9E", size: 11 },
        xaxis: { title: "SNR (dB)", gridcolor: "#262626" },
        yaxis: { title: "BER", type: "log", gridcolor: "#262626" },
        margin: { l: 50, r: 20, t: 20, b: 40 },
    };
    window.Plotly.newPlot(body, [trace], layout, {
        displayModeBar: false,
        responsive: true,
    });
}

function renderScene(body, pane) {
    const url = pane.data && (pane.data.url || pane.data.glb_url);
    if (!url) {
        body.appendChild(errorMsg("No scene URL"));
        return;
    }
    const canvasWrap = document.createElement("div");
    canvasWrap.className = "pane-scene-canvas";
    body.appendChild(canvasWrap);

    // Lazy-import three so the bundle stays small if no scene pane shows up.
    Promise.all([
        import("three"),
        import("three/addons/loaders/GLTFLoader.js"),
        import("three/addons/controls/OrbitControls.js"),
    ]).then(([THREE, gltfMod, orbitMod]) => {
        const GLTFLoader = gltfMod.GLTFLoader;
        const OrbitControls = orbitMod.OrbitControls;

        const w = canvasWrap.clientWidth || 360;
        const h = canvasWrap.clientHeight || 260;
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x050505);

        const camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 5000);
        camera.position.set(5, 3, 5);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(w, h);
        renderer.setPixelRatio(window.devicePixelRatio);
        canvasWrap.appendChild(renderer.domElement);

        const amb = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(amb);
        const dir = new THREE.DirectionalLight(0xffffff, 0.8);
        dir.position.set(5, 10, 5);
        scene.add(dir);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 0, 0);

        new GLTFLoader().load(url, (gltf) => {
            scene.add(gltf.scene);
            const box = new THREE.Box3().setFromObject(gltf.scene);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3()).length();
            controls.target.copy(center);
            camera.position.copy(center).add(new THREE.Vector3(size, size * 0.6, size));
            controls.update();
        }, undefined, (err) => {
            body.appendChild(errorMsg(`GLB load failed: ${err && err.message}`));
        });

        function tick() {
            requestAnimationFrame(tick);
            controls.update();
            renderer.render(scene, camera);
        }
        tick();
    }).catch((err) => {
        body.appendChild(errorMsg(`three.js import failed: ${err.message || err}`));
    });
}

function renderTable(body, pane) {
    const data = pane.data || {};
    let rows = [];
    let columns = data.columns || null;
    if (Array.isArray(data.rows)) {
        rows = data.rows;
    } else if (Array.isArray(data)) {
        rows = data;
    } else if (data && typeof data === "object") {
        // Convert flat object into key/value rows.
        rows = Object.entries(data).map(([k, v]) => ({ key: k, value: v }));
        columns = columns || ["key", "value"];
    }

    if (rows.length === 0) {
        body.appendChild(emptyMsg("No rows"));
        return;
    }
    if (!columns) {
        columns = Object.keys(rows[0]);
    }

    const tbl = document.createElement("table");
    tbl.className = "pane-table";
    const thead = document.createElement("thead");
    const trh = document.createElement("tr");
    columns.forEach((c) => {
        const th = document.createElement("th");
        th.textContent = c;
        trh.appendChild(th);
    });
    thead.appendChild(trh);
    tbl.appendChild(thead);

    const tbody = document.createElement("tbody");
    rows.forEach((r) => {
        const tr = document.createElement("tr");
        columns.forEach((c) => {
            const td = document.createElement("td");
            const v = r[c];
            td.textContent = typeof v === "object" && v !== null
                ? JSON.stringify(v)
                : (v === undefined || v === null ? "" : String(v));
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    tbl.appendChild(tbody);
    body.appendChild(tbl);
}

function renderCode(body, pane) {
    const data = pane.data || {};
    const src = data.src || data.code || "";
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    code.textContent = src;
    if (data.language) code.className = `language-${data.language}`;
    pre.appendChild(code);
    body.appendChild(pre);
}

function renderJson(body, pane) {
    const pre = document.createElement("pre");
    try {
        pre.textContent = JSON.stringify(pane.data, null, 2);
    } catch (err) {
        pre.textContent = String(pane.data);
    }
    body.appendChild(pre);
}

function renderUnknown(body, pane) {
    body.appendChild(errorMsg(`Unknown pane type: ${pane.type}`));
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(pane.data, null, 2);
    body.appendChild(pre);
}

// ─── Helpers ──────────────────────────────────────────────────────────────

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
}

function errorMsg(msg) {
    const div = document.createElement("div");
    div.className = "pane-empty";
    div.style.color = "var(--sim-heat-1)";
    div.textContent = msg;
    return div;
}

function emptyMsg(msg) {
    const div = document.createElement("div");
    div.className = "pane-empty";
    div.textContent = msg;
    return div;
}

// Expose on window for non-module callers (dashboard.js is a module too but
// keeps most state on window).
window._renderAgentResponse = renderAgentResponse;
window._clearPanes = clearPanes;
