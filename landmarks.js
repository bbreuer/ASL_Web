// MediaPipe hand landmark indices (match Python classifier.py).
export const WRIST = 0;
export const THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4;
export const INDEX_MCP = 5, INDEX_PIP = 6, INDEX_DIP = 7, INDEX_TIP = 8;
export const MIDDLE_MCP = 9, MIDDLE_PIP = 10, MIDDLE_DIP = 11, MIDDLE_TIP = 12;
export const RING_MCP = 13, RING_PIP = 14, RING_DIP = 15, RING_TIP = 16;
export const PINKY_MCP = 17, PINKY_PIP = 18, PINKY_DIP = 19, PINKY_TIP = 20;

export const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],
    [0,17],
];

// MediaPipe returns landmarks in original (unflipped) image space.
// The Python desktop version ran cv2.flip(frame, 1) BEFORE detection,
// so all its math assumes mirrored coords. To keep the classifier
// behaving identically here, we mirror x once before classification.
export function mirrorLandmarks(landmarks) {
    return landmarks.map(lm => ({ x: 1 - lm.x, y: lm.y, z: lm.z }));
}

function landmarksToArray(landmarks) {
    return landmarks.map(lm => [lm.x, lm.y, lm.z]);
}

// Subtract wrist, scale so wrist→middle_mcp = 1. Preserves orientation.
// Returns Float32Array(63) or null if hand is too small.
export function featureVector(landmarks) {
    const pts = landmarksToArray(landmarks);
    const wrist = pts[WRIST];
    for (let i = 0; i < pts.length; i++) {
        pts[i] = [pts[i][0] - wrist[0], pts[i][1] - wrist[1], pts[i][2] - wrist[2]];
    }
    const mid = pts[MIDDLE_MCP];
    const scale = Math.hypot(mid[0], mid[1], mid[2]);
    if (scale < 1e-6) return null;
    const out = new Float32Array(pts.length * 3);
    for (let i = 0; i < pts.length; i++) {
        out[i*3]     = pts[i][0] / scale;
        out[i*3 + 1] = pts[i][1] / scale;
        out[i*3 + 2] = pts[i][2] / scale;
    }
    return out;
}

// Returns normalized 3D points (array of [x,y,z]) for motion tracking.
export function normalizedPoints(landmarks) {
    const pts = landmarksToArray(landmarks);
    const wrist = pts[WRIST];
    for (let i = 0; i < pts.length; i++) {
        pts[i] = [pts[i][0] - wrist[0], pts[i][1] - wrist[1], pts[i][2] - wrist[2]];
    }
    const mid = pts[MIDDLE_MCP];
    const scale = Math.hypot(mid[0], mid[1], mid[2]);
    if (scale < 1e-6) return null;
    for (let i = 0; i < pts.length; i++) {
        pts[i] = [pts[i][0]/scale, pts[i][1]/scale, pts[i][2]/scale];
    }
    return pts;
}

export function rawArray(landmarks) {
    return landmarksToArray(landmarks);
}
