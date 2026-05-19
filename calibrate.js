import { HandLandmarker, FilesetResolver }
    from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/+esm';

import { KnnClassifier } from './classifier.js';
import { HAND_CONNECTIONS, mirrorLandmarks, featureVector } from './landmarks.js';

const SAMPLES_PER_LETTER = 40;

const LETTERS = [
    ['A', 'Closed fist, thumb resting on the side of the index.'],
    ['B', 'Flat hand, all four fingers up, thumb tucked across palm.'],
    ['C', "Curve hand into a 'C' — fingers and thumb form an open curve."],
    ['D', 'Index finger straight up; other fingers curl to touch thumb.'],
    ['E', 'All fingers curl down; thumb tucks across in front.'],
    ['F', 'Index curls to thumb (small circle); middle/ring/pinky up.'],
    ['G', 'Index points sideways (horizontal); thumb parallel below it.'],
    ['H', 'Index AND middle point sideways together (horizontal).'],
    ['I', 'Pinky up only; other fingers curled, thumb tucked.'],
    ['K', 'Index up, middle out at an angle; thumb between them.'],
    ['L', 'Index up + thumb out — clear right-angle L shape.'],
    ['M', 'Fist with thumb tucked under index, middle, AND ring.'],
    ['N', 'Fist with thumb tucked under index AND middle (only).'],
    ['O', 'All fingers curl to meet thumb — round O shape.'],
    ['P', 'Like K but rotated to point downward.'],
    ['Q', 'Like G but rotated to point downward.'],
    ['R', 'Index and middle up, with index CROSSED in front of middle.'],
    ['S', 'Closed fist, thumb wrapped over the front of the fingers.'],
    ['T', 'Fist with thumb poking out between index and middle.'],
    ['U', 'Index and middle up, held together (touching).'],
    ['V', 'Index and middle up, spread apart in a V.'],
    ['W', 'Index, middle, and ring up; pinky and thumb tucked.'],
    ['X', 'Index hooked (bent at middle joint); other fingers curled.'],
    ['Y', 'Thumb and pinky out (hang-loose shape); other 3 curled.'],
];

const els = {
    video:       document.getElementById('video'),
    canvas:      document.getElementById('overlay'),
    letter:      document.getElementById('letter'),
    signImg:     document.getElementById('sign-img'),
    counter:     document.getElementById('counter'),
    hint:        document.getElementById('hint'),
    progress:    document.getElementById('progress'),
    status:      document.getElementById('status'),
    btnRecord:   document.getElementById('btn-record'),
    btnSkip:     document.getElementById('btn-skip'),
    btnBack:     document.getElementById('btn-back'),
    btnFinish:   document.getElementById('btn-finish'),
    btnClearAll: document.getElementById('btn-clear-all'),
};

// Maps letter → "signs/ASL SHEET-NN.png" (A=01 … Z=26).
function loadSignImage(letter) {
    if (!els.signImg) return;
    els.signImg.classList.add('missing');
    const n = letter.charCodeAt(0) - 'A'.charCodeAt(0) + 1;
    if (n < 1 || n > 26) return;
    const padded = String(n).padStart(2, '0');
    const url = `signs/ASL%20SHEET-${padded}.png`;
    const probe = new Image();
    probe.onload = () => {
        els.signImg.src = url;
        els.signImg.classList.remove('missing');
    };
    probe.onerror = () => {
        els.signImg.removeAttribute('src');
        els.signImg.classList.add('missing');
    };
    probe.src = url;
}

let handLandmarker = null;
let stream = null;
let i = 0;
let recording = false;
let recorded = [];
let lastFrameTime = -1;

let samples = (() => {
    const existing = KnnClassifier.load();
    if (existing) return { X: [...existing.X], y: [...existing.y] };
    return { X: [], y: [] };
})();

async function setup() {
    stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
    });
    els.video.srcObject = stream;
    await els.video.play();

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
    els.video.addEventListener('loadedmetadata', syncCanvasToVideo);
    window.addEventListener('resize', syncCanvasToVideo);
    window.addEventListener('orientationchange', syncCanvasToVideo);
    if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(syncCanvasToVideo).observe(els.video.parentElement);
    }

    showLetter();
    requestAnimationFrame(loop);
}

function showLetter() {
    const [letter, hint] = LETTERS[i];
    els.letter.textContent = letter;
    els.hint.textContent = hint;
    els.counter.textContent = `${i + 1} / ${LETTERS.length}`;
    loadSignImage(letter);
    const already = samples.y.includes(letter);
    els.status.textContent = already
        ? '(already calibrated — recording will overwrite)'
        : '';
    recording = false;
    recorded = [];
    els.progress.style.width = '0%';
    els.btnRecord.textContent = 'Record';
    els.btnRecord.classList.remove('recording');
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

    if (results && results.landmarks && results.landmarks.length > 0) {
        const lms = results.landmarks[0];
        drawLandmarks(lms, vw, vh);
        if (recording) {
            const v = featureVector(mirrorLandmarks(lms));
            if (v) recorded.push(v);
            els.progress.style.width = Math.min(100, (recorded.length / SAMPLES_PER_LETTER) * 100) + '%';
            els.btnRecord.textContent = `${recorded.length}/${SAMPLES_PER_LETTER}`;
            if (recorded.length >= SAMPLES_PER_LETTER) {
                commitLetter();
                advance(1);
            }
        }
    } else {
        clearCanvas(vw, vh);
    }

    requestAnimationFrame(loop);
}

function commitLetter() {
    const letter = LETTERS[i][0];
    const keepX = [], keepY = [];
    for (let j = 0; j < samples.y.length; j++) {
        if (samples.y[j] !== letter) {
            keepX.push(samples.X[j]);
            keepY.push(samples.y[j]);
        }
    }
    for (const v of recorded) {
        keepX.push(v);
        keepY.push(letter);
    }
    samples = { X: keepX, y: keepY };
    KnnClassifier.save(samples.X, samples.y);
}

function advance(delta) {
    let next = i + delta;
    if (next < 0) next = 0;
    if (next >= LETTERS.length) next = LETTERS.length - 1;
    i = next;
    showLetter();
}

function wireButtons() {
    els.btnRecord.addEventListener('click', () => {
        if (recording) {
            recording = false;
            recorded = [];
            els.btnRecord.textContent = 'Record';
            els.btnRecord.classList.remove('recording');
            els.progress.style.width = '0%';
            return;
        }
        recording = true;
        recorded = [];
        els.btnRecord.classList.add('recording');
    });
    els.btnSkip.addEventListener('click', () => advance(1));
    els.btnBack.addEventListener('click', () => advance(-1));
    els.btnFinish.addEventListener('click', () => {
        window.location.href = 'index.html';
    });
    els.btnClearAll.addEventListener('click', () => {
        if (confirm('Erase ALL calibration samples? This cannot be undone.')) {
            KnnClassifier.clear();
            samples = { X: [], y: [] };
            showLetter();
        }
    });
}

wireButtons();
setup().catch(e => {
    els.status.textContent = 'Setup failed: ' + e.message;
});
