import torch
import sounddevice as sd
from pesto import load_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# === Audio and Model Config ===
RATE = 48000
BUFFER_SIZE = 2048
HISTORY_LEN = 200
SAMPLE_SCALE = 4

pesto_model = load_model(
    "mir-1k_g7",
    step_size=5.0,
    sampling_rate=RATE * SAMPLE_SCALE,
    streaming=True,
    max_batch_size=1,
)

# === Pitch + Confidence History ===
pitch_history = [0.0] * HISTORY_LEN
conf_history = [0.1] * HISTORY_LEN  # Start with low confidence values

# === Plot Setup ===
fig, ax = plt.subplots()
ax.set_ylim(50, 600)
ax.set_xlim(0, HISTORY_LEN - 1)
ax.set_title("Pitch Trace (Confidence = Darkness)")
ax.set_ylabel("Pitch (Hz)")
ax.set_xlabel("Frame")
ax.grid(True)

# Create empty LineCollection
segments = []
for i in range(HISTORY_LEN - 1):
    segments.append([[i, 0], [i + 1, 0]])
line_collection = LineCollection(segments, linewidth=2)
ax.add_collection(line_collection)


# === Update Function ===
def update(frame):
    global pitch_history, conf_history

    # Record and process
    buffer = sd.rec(BUFFER_SIZE, samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    buffer_tensor = torch.tensor(buffer.T, dtype=torch.float32)
    pitch, conf, amp = pesto_model(
        buffer_tensor, return_activations=False, convert_to_freq=True
    )

    pitch_val = pitch.mean().item() / SAMPLE_SCALE if pitch.numel() > 0 else 0.0
    conf_val = conf.mean().item() if conf.numel() > 0 else 0.1

    if not torch.isfinite(pitch.mean()):
        pitch_val = 0.0
    if not torch.isfinite(conf.mean()):
        conf_val = 0.1

    # Update history
    pitch_history = pitch_history[1:] + [pitch_val]
    conf_history = conf_history[1:] + [conf_val]

    # Create new segments
    new_segments = []
    new_alphas = []

    for i in range(HISTORY_LEN - 1):
        new_segments.append([[i, pitch_history[i]], [i + 1, pitch_history[i + 1]]])
        alpha = max(0.05, min(conf_history[i], 1.0))  # Clamp alpha to [0.05, 1.0]
        new_alphas.append(alpha)

    # Map alphas to RGBA colors
    colors = [(0, 0, 1, a) for a in new_alphas]  # Blue line, variable alpha

    line_collection.set_segments(new_segments)
    line_collection.set_color(colors)

    return (line_collection,)


# === Animate ===
ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
plt.tight_layout()
plt.show()
