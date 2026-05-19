// Web port of classifier.py.
// Expects MIRRORED landmarks (caller is responsible for x → 1-x).

import {
    WRIST,
    THUMB_TIP, THUMB_IP,
    INDEX_MCP, INDEX_PIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
    RING_MCP, RING_PIP, RING_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_TIP,
    featureVector, normalizedPoints, rawArray,
} from './landmarks.js';

const TEMPLATES_KEY = 'asl_templates_v1';

// ---------- KNN classifier ----------

export class KnnClassifier {
    constructor(X, y, k = 7, maxDistance = 0.9) {
        this.X = X; // array of Float32Array(63)
        this.y = y; // array of strings
        this.k = Math.min(k, y.length);
        this.maxDistance = maxDistance;
    }

    static load() {
        try {
            const raw = localStorage.getItem(TEMPLATES_KEY);
            if (!raw) return null;
            const data = JSON.parse(raw);
            if (!data || !data.X || !data.y || data.X.length === 0) return null;
            const X = data.X.map(a => Float32Array.from(a));
            return new KnnClassifier(X, data.y);
        } catch {
            return null;
        }
    }

    static save(X, y) {
        const data = {
            X: X.map(a => Array.from(a)),
            y: y,
        };
        localStorage.setItem(TEMPLATES_KEY, JSON.stringify(data));
    }

    static clear() {
        localStorage.removeItem(TEMPLATES_KEY);
    }

    predict(landmarks) {
        const v = featureVector(landmarks);
        if (!v) return null;
        const n = this.X.length;
        const dists = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            const x = this.X[i];
            let s = 0;
            for (let j = 0; j < v.length; j++) {
                const d = x[j] - v[j];
                s += d * d;
            }
            dists[i] = Math.sqrt(s);
        }
        const idx = Array.from(dists.keys())
            .sort((a, b) => dists[a] - dists[b])
            .slice(0, this.k);
        if (dists[idx[0]] > this.maxDistance) return null;
        const votes = new Map();
        for (const i of idx) {
            const lbl = this.y[i];
            votes.set(lbl, (votes.get(lbl) || 0) + 1);
        }
        let best = null, bestCount = 0;
        for (const [lbl, c] of votes) {
            if (c > bestCount) { best = lbl; bestCount = c; }
        }
        if (bestCount < Math.floor(this.k / 2) + 1) return null;
        return best;
    }
}

// ---------- Rule-based fallback ----------

function dist(a, b) {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function ext(pts, tip, pip, mcp, ratio = 1.3) {
    return dist(pts[tip], pts[mcp]) > dist(pts[pip], pts[mcp]) * ratio;
}

function handScale(pts) {
    return dist(pts[WRIST], pts[MIDDLE_MCP]);
}

function partiallyCurled(pts) {
    const s = handScale(pts);
    const pairs = [
        [INDEX_TIP, INDEX_MCP],
        [MIDDLE_TIP, MIDDLE_MCP],
        [RING_TIP, RING_MCP],
        [PINKY_TIP, PINKY_MCP],
    ];
    return pairs.map(([tip, mcp]) => {
        const d = dist(pts[tip], pts[mcp]) / Math.max(s, 1e-6);
        return d > 0.55 && d < 1.15;
    });
}

export function classifyRules(landmarks) {
    const pts = rawArray(landmarks);
    const s = handScale(pts);
    if (s < 1e-4) return null;

    const idx = ext(pts, INDEX_TIP, INDEX_PIP, INDEX_MCP);
    const mid = ext(pts, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP);
    const ring = ext(pts, RING_TIP, RING_PIP, RING_MCP);
    const pky = ext(pts, PINKY_TIP, PINKY_PIP, PINKY_MCP);
    const thb = dist(pts[THUMB_TIP], pts[INDEX_MCP]) > s * 0.75;

    const key = `${idx?1:0}${mid?1:0}${ring?1:0}${pky?1:0}`;

    if (key === '0000') {
        const thumbIndex = dist(pts[THUMB_TIP], pts[INDEX_TIP]) / s;
        if (thumbIndex < 0.45) return 'O';
        const curls = partiallyCurled(pts);
        if (curls.every(x => x) && thumbIndex > 0.6) return 'C';
        if (pts[THUMB_TIP][1] <= pts[INDEX_PIP][1] + s * 0.05) return 'A';
        return 'S';
    }
    if (key === '1111') return 'B';
    if (key === '0001') return thb ? 'Y' : 'I';
    if (key === '1000') {
        if (thb) return 'L';
        const d = dist(pts[INDEX_TIP], pts[INDEX_MCP]) / s;
        return d < 1.1 ? 'X' : 'D';
    }
    if (key === '1100') {
        const tipGap = dist(pts[INDEX_TIP], pts[MIDDLE_TIP]);
        const mcpGap = dist(pts[INDEX_MCP], pts[MIDDLE_MCP]);
        const idxLeftMcp = pts[INDEX_MCP][0] < pts[MIDDLE_MCP][0];
        const idxLeftTip = pts[INDEX_TIP][0] < pts[MIDDLE_TIP][0];
        if (idxLeftMcp !== idxLeftTip && tipGap < mcpGap * 1.3) return 'R';
        if (tipGap > mcpGap * 2.0) return 'V';
        if (thb && pts[THUMB_TIP][1] < pts[THUMB_IP][1]) return 'K';
        return 'U';
    }
    if (key === '1110') return 'W';
    if (key === '0111') return 'F';
    return null;
}

// ---------- Top-level (adds J/Z motion detection) ----------

export class Classifier {
    constructor(knn = null, historyLen = 22) {
        this.knn = knn;
        this.historyLen = historyLen;
        this.history = [];
        this.labelHistory = [];
    }

    resetHistory() {
        this.history = [];
        this.labelHistory = [];
    }

    predict(landmarks) {
        const pts = normalizedPoints(landmarks);
        if (!pts) { this.resetHistory(); return null; }
        this.history.push(pts);
        if (this.history.length > this.historyLen) this.history.shift();

        const stat = this.knn ? this.knn.predict(landmarks) : classifyRules(landmarks);
        this.labelHistory.push(stat);
        if (this.labelHistory.length > this.historyLen) this.labelHistory.shift();

        const motion = this._detectMotion();
        return motion || stat;
    }

    _detectMotion() {
        if (this.history.length < this.historyLen) return null;
        if (this._mostlyLabel('D') && this._zigzag(INDEX_TIP, 2, 1.8)) return 'Z';
        if (this._mostlyLabel('I') && this._jSweep(PINKY_TIP, 1.4)) return 'J';
        return null;
    }

    _mostlyLabel(label) {
        let c = 0;
        for (const l of this.labelHistory) if (l === label) c++;
        return c >= 0.6 * this.labelHistory.length;
    }

    _tipPath(tipIdx) {
        let total = 0;
        for (let i = 1; i < this.history.length; i++) {
            const a = this.history[i][tipIdx], b = this.history[i-1][tipIdx];
            total += Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]);
        }
        return total;
    }

    _zigzag(tipIdx, minChanges, minPath) {
        if (this._tipPath(tipIdx) < minPath) return false;
        let prev = 0, changes = 0;
        for (let i = 1; i < this.history.length; i++) {
            const dx = this.history[i][tipIdx][0] - this.history[i-1][tipIdx][0];
            const cur = dx > 0.04 ? 1 : (dx < -0.04 ? -1 : 0);
            if (cur !== 0 && prev !== 0 && cur !== prev) changes++;
            if (cur !== 0) prev = cur;
        }
        return changes >= minChanges;
    }

    _jSweep(tipIdx, minPath) {
        if (this._tipPath(tipIdx) < minPath) return false;
        const ys = this.history.map(p => p[tipIdx][1]);
        const xs = this.history.map(p => p[tipIdx][0]);
        const n = ys.length;
        const dy = ys[Math.floor(n / 2)] - ys[0];
        const dx = xs[n - 1] - xs[Math.floor(n / 2)];
        return dy > 0.4 && Math.abs(dx) > 0.25;
    }
}
