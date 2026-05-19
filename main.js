import { HandLandmarker, FilesetResolver }
    from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/+esm';

import { Classifier, KnnClassifier } from './classifier.js';
import { HAND_CONNECTIONS, mirrorLandmarks } from './landmarks.js';
import {
    say, toggleMute, installPrimer, unlockAudio,
    onAudioState, isAvailable,
} from './speech.js';

const STABLE_FRAMES = 12;
const COOLDOWN_FRAMES = 8;

const els = {
    video:        document.getElementById('video'),
    canvas:       document.getElementById('overlay'),
    detected:     document.getElementById('detected'),
    progress:     document.getElementById('progress'),
    text:         document.getElementById('text'),
    status:       document.getElementById('status'),
    audioBanner:  document.getElementById('audio-banner'),
    btnSpace:     document.getElementById('btn-space'),
    btnBack:      document.getElementById('btn-back'),
    btnClear:     document.getElementById('btn-clear'),
    btnMute:      document.getElementById('btn-mute'),
    btnSwitch:    document.getElementById('btn-switch'),
    btnCalibrate: document.getElementById('btn-calibrate'),
};

let handLandmarker = null;
let classifier = null;
let stream = null;
let facingMode = 'user';
let lastFrameTime = -1;

let textValue = '';
let currentWord = '';
let lastLetter = null;
let stable = 0;
let cooldown = 0;
let lastCommitted = null;

async function setupLandmarker() {
    els.status.textContent = 'Loading hand model…';
    const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm'
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 1,
    });
    refreshStatus();
}

async function startCamera() {
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
    });
    els.video.srcObject = stream;
    await els.video.play();
}

function loadClassifier() {
    const knn = KnnClassifier.load();
    classifier = new Classifier(knn);
}

function refreshStatus() {
    const knn = classifier && classifier.knn;
    if (knn) {
        const letters = [...new Set(knn.y)].sort().join(' ');
        els.status.textContent = `Calibrated: ${letters}  (${knn.X.length} samples)`;
    } else {
        els.status.textContent = 'No calibration — using rule-based fallback. Tap Calibrate for full accuracy.';
    }
}

function syncCanvasToVideo() {
    const v = els.video;
    const wrap = v.parentElement;
    const vw = v.videoWidth, vh = v.videoHeight;
    const cw = wrap.clientWidth, ch = wrap.clientHeight;
    if (!vw || !vh || !cw || !ch) return;
    const vAspect = vw / vh;
    const cAspect = cw / ch;
    let dispW, dispH, offX = 0, offY = 0;
    if (vAspect > cAspect) {
        dispW = cw;
        dispH = cw / vAspect;
        offY = (ch - dispH) / 2;
    } else {
        dispH = ch;
        dispW = ch * vAspect;
        offX = (cw - dispW) / 2;
    }
    const c = els.canvas;
    c.style.left = offX + 'px';
    c.style.top = offY + 'px';
    c.style.width = dispW + 'px';
    c.style.height = dispH + 'px';
    if (c.width !== vw) c.width = vw;
    if (c.height !== vh) c.height = vh;
}

function drawLandmarks(landmarks, w, h) {
    const ctx = els.canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#4ade80';
    ctx.fillStyle = '#f472b6';
    for (const [a, b] of HAND_CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
        ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
        ctx.stroke();
    }
    for (const lm of landmarks) {
        ctx.beginPath();
        ctx.arc(lm.x * w, lm.y * h, 4, 0, Math.PI * 2);
        ctx.fill();
    }
}

function clearCanvas(w, h) {
    const ctx = els.canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
}

function loop() {
    if (!handLandmarker || els.video.readyState < 2) {
        requestAnimationFrame(loop);
        return;
    }

    const vw = els.video.videoWidth, vh = els.video.videoHeight;

    const now = performance.now();
    let results = null;
    if (now !== lastFrameTime) {
        lastFrameTime = now;
        try {
            results = handLandmarker.detectForVideo(els.video, now);
        } catch (e) {
            console.warn(e);
        }
    }

    let current = null;
    if (results && results.landmarks && results.landmarks.length > 0) {
        const lms = results.landmarks[0];
        drawLandmarks(lms, vw, vh);
        current = classifier.predict(mirrorLandmarks(lms));
    } else {
        clearCanvas(vw, vh);
        classifier.resetHistory();
    }

    if (current && current === lastLetter) stable++;
    else { stable = 1; lastLetter = current; }
    if (cooldown > 0) cooldown--;

    const commitThreshold = (current === 'J' || current === 'Z') ? 5 : STABLE_FRAMES;
    if (current && stable === commitThreshold && cooldown === 0 && current !== lastCommitted) {
        textValue += current;
        currentWord += current;
        say(current);
        lastCommitted = current;
        cooldown = COOLDOWN_FRAMES;
        if (current === 'J' || current === 'Z') classifier.resetHistory();
    } else if (!current) {
        lastCommitted = null;
    }

    els.detected.textContent = current || '–';
    els.progress.style.width = current
        ? Math.min(100, (stable / commitThreshold) * 100) + '%'
        : '0%';
    els.text.textContent = textValue || ' ';

    requestAnimationFrame(loop);
}

function wireButtons() {
    els.btnSpace.addEventListener('click', () => {
        if (currentWord) say(currentWord, { interrupt: true });
        textValue += ' ';
        currentWord = '';
        lastCommitted = ' ';
    });
    els.btnBack.addEventListener('click', () => {
        textValue = textValue.slice(0, -1);
        currentWord = currentWord.slice(0, -1);
        lastCommitted = null;
    });
    els.btnClear.addEventListener('click', () => {
        textValue = '';
        currentWord = '';
        lastCommitted = null;
    });
    els.btnMute.addEventListener('click', () => {
        const m = toggleMute();
        els.btnMute.textContent = m ? '🔇' : '🔊';
    });
    if (els.audioBanner) {
        els.audioBanner.addEventListener('click', () => unlockAudio());
    }
    els.btnSwitch.addEventListener('click', async () => {
        facingMode = facingMode === 'user' ? 'environment' : 'user';
        try { await startCamera(); } catch (e) { console.warn(e); }
    });
    els.btnCalibrate.addEventListener('click', () => {
        window.location.href = 'calibrate.html';
    });

    document.addEventListener('keydown', (e) => {
        if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
        if (e.key === ' ')              { e.preventDefault(); els.btnSpace.click(); }
        else if (e.key === 'Backspace') { els.btnBack.click(); }
        else if (e.key === 'c' || e.key === 'C') { els.btnClear.click(); }
        else if (e.key === 'm' || e.key === 'M') { els.btnMute.click(); }
    });
}

(async () => {
    installPrimer();
    wireButtons();
    loadClassifier();
    refreshStatus();

    els.video.addEventListener('loadedmetadata', syncCanvasToVideo);
    window.addEventListener('resize', syncCanvasToVideo);
    window.addEventListener('orientationchange', syncCanvasToVideo);
    if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(syncCanvasToVideo).observe(els.video.parentElement);
    }

    if (!isAvailable()) {
        els.btnMute.style.display = 'none';
        if (els.audioBanner) els.audioBanner.hidden = true;
    } else {
        onAudioState((s) => {
            if (!els.audioBanner) return;
            els.audioBanner.hidden = (s !== 'locked');
            els.btnMute.textContent = s === 'muted' ? '🔇' : '🔊';
        });
    }

    try {
        await startCamera();
    } catch (e) {
        els.status.textContent = 'Camera access denied. Reload and allow camera permission.';
        return;
    }

    try {
        await setupLandmarker();
    } catch (e) {
        els.status.textContent = 'Could not load hand model: ' + e.message;
        return;
    }

    requestAnimationFrame(loop);
})();
