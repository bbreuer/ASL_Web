// Web Speech API wrapper. iOS Safari requires a user gesture to unlock
// speechSynthesis — we prime it on the first user interaction.

let muted = false;
let preferredVoice = null;
let voicesReady = false;
let primed = false;

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

export function primeSpeech() {
    if (primed) return;
    if (typeof speechSynthesis === 'undefined') return;
    try {
        const u = new SpeechSynthesisUtterance(' ');
        u.volume = 0;
        speechSynthesis.speak(u);
        primed = true;
    } catch {
        // ignore
    }
}

export function installPrimer() {
    if (typeof document === 'undefined') return;
    const handler = () => {
        primeSpeech();
        document.removeEventListener('click', handler);
        document.removeEventListener('touchstart', handler);
        document.removeEventListener('keydown', handler);
    };
    document.addEventListener('click', handler, { once: true });
    document.addEventListener('touchstart', handler, { once: true });
    document.addEventListener('keydown', handler, { once: true });
}

export function say(text, { interrupt = false } = {}) {
    if (muted || !text) return;
    if (typeof speechSynthesis === 'undefined') return;
    if (interrupt) speechSynthesis.cancel();
    if (!voicesReady) ensureVoices();
    const u = new SpeechSynthesisUtterance(String(text));
    if (preferredVoice) u.voice = preferredVoice;
    u.rate = 1.0;
    u.pitch = 1.0;
    speechSynthesis.speak(u);
}

export function toggleMute() {
    muted = !muted;
    if (muted && typeof speechSynthesis !== 'undefined') speechSynthesis.cancel();
    return muted;
}

export function isMuted() { return muted; }
export function isAvailable() { return typeof speechSynthesis !== 'undefined'; }
