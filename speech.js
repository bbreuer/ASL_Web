// Web Speech API wrapper with a 3-state model:
//   'locked' — needs a user gesture before speak() will produce audio (iOS rule)
//   'ready'  — audio is unlocked and active
//   'muted'  — user has muted; speak() is suppressed
//
// Subscribe with onAudioState(cb) to drive UI (e.g. show "tap to enable" banner).

let state = 'locked';
let preferredVoice = null;
let voicesReady = false;
const listeners = new Set();

const PREFERRED = [
    'Samantha', 'Aria', 'Jenny', 'Ava',
    'Google US English', 'Google UK English Female',
    'Microsoft Aria', 'Microsoft Zira',
    'Karen', 'Tessa',
];

function chooseVoice() {
    const voices = speechSynthesis.getVoices();
    if (!voices.length) return null;
    for (const name of PREFERRED) {
        const v = voices.find(x => x.name.includes(name));
        if (v) return v;
    }
    return voices.find(v => v.lang && v.lang.startsWith('en')) || voices[0];
}

function ensureVoices() {
    preferredVoice = chooseVoice();
    voicesReady = true;
}

if (typeof speechSynthesis !== 'undefined') {
    if (speechSynthesis.getVoices().length) ensureVoices();
    speechSynthesis.onvoiceschanged = ensureVoices;
}

function setState(s) {
    if (state === s) return;
    state = s;
    for (const cb of listeners) {
        try { cb(state); } catch {}
    }
}

export function onAudioState(cb) {
    listeners.add(cb);
    try { cb(state); } catch {}
    return () => listeners.delete(cb);
}

export function audioState() { return state; }

// Must be called from a user gesture handler. Speaks a brief, mostly
// inaudible utterance to unlock iOS Safari's speech API for the session.
export function unlockAudio() {
    if (state !== 'locked') return;
    if (typeof speechSynthesis === 'undefined') return;
    try {
        if (!voicesReady) ensureVoices();
        try { speechSynthesis.resume(); } catch {}
        const u = new SpeechSynthesisUtterance(' ');
        if (preferredVoice) u.voice = preferredVoice;
        u.volume = 1;
        u.rate = 1;
        let settled = false;
        const ready = () => { if (!settled) { settled = true; setState('ready'); } };
        u.onstart = ready;
        u.onend = ready;
        u.onerror = ready;
        speechSynthesis.speak(u);
        // Safety net — some browsers don't fire events for empty utterances
        setTimeout(ready, 1200);
    } catch {
        setState('ready');
    }
}

// Auto-unlock on the first real user gesture anywhere on the page.
export function installPrimer() {
    if (typeof document === 'undefined') return;
    const handler = () => unlockAudio();
    document.addEventListener('click', handler, true);
    document.addEventListener('touchstart', handler, true);
    document.addEventListener('keydown', handler, true);
}

export function say(text, { interrupt = false } = {}) {
    if (!text) return;
    if (typeof speechSynthesis === 'undefined') return;
    if (state !== 'ready') return; // 'locked' or 'muted' → drop
    try { speechSynthesis.resume(); } catch {}
    if (interrupt) speechSynthesis.cancel();
    if (!voicesReady) ensureVoices();
    const u = new SpeechSynthesisUtterance(String(text));
    if (preferredVoice) u.voice = preferredVoice;
    u.rate = 1.0;
    u.pitch = 1.0;
    speechSynthesis.speak(u);
}

export function toggleMute() {
    if (state === 'locked') {
        unlockAudio();
        // After unlock, leave as 'ready' (user clearly wants audio).
        return false;
    }
    if (state === 'ready') {
        try { speechSynthesis.cancel(); } catch {}
        setState('muted');
        return true;
    }
    // muted → ready
    setState('ready');
    return false;
}

export function isAvailable() { return typeof speechSynthesis !== 'undefined'; }
export function isMuted() { return state === 'muted'; }
