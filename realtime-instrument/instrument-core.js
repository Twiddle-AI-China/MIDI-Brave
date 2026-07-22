export const clamp = (value, low, high) => (
  Math.max(low, Math.min(high, Number(value)))
);

export function xyFromPointer(clientX, clientY, rect) {
  return {
    x: clamp(((clientX - rect.left) / rect.width) * 2 - 1, -1, 1),
    y: clamp(1 - ((clientY - rect.top) / rect.height) * 2, -1, 1),
  };
}

export function controlFrame(seq, state) {
  return {
    type: 'control',
    seq,
    x: clamp(state.x, -1, 1),
    y: clamp(state.y, -1, 1),
    note: Math.round(clamp(state.note, 21, 109)),
    velocity: clamp(state.velocity, 0, 1),
  };
}

export function noteName(note) {
  const names = [
    'C', 'C#', 'D', 'D#', 'E', 'F',
    'F#', 'G', 'G#', 'A', 'A#', 'B',
  ];
  return `${names[note % 12]}${Math.floor(note / 12) - 1}`;
}
