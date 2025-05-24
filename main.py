import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt6 import QtGui

# === Settings ===
samplerate = 44100
blocksize = 2048

# === Shared buffers ===
latest_waveform = np.zeros(blocksize)
latest_fft = np.zeros(blocksize // 2 + 1)
fft_smooth = np.zeros(blocksize // 2 + 1)
fft_freqs = np.fft.rfftfreq(blocksize, d=1/samplerate)

# === Create GUI ===
app = pg.mkQApp()
pg.setConfigOptions(antialias=True)
win = pg.GraphicsLayoutWidget(title="Live Audio Visualizer")
win.resize(800, 600)
win.show()

# === Waveform Plot ===
waveform_plot = win.addPlot(title="Waveform (Time Domain)")
waveform_plot.setYRange(-1, 1)
waveform_plot.setMouseEnabled(x=False, y=False)
waveform_plot.hideButtons()

# ✅ Smooth line
curve_waveform = waveform_plot.plot(pen=pg.mkPen(color='c', width=2))  # Cyan line

# === FFT Plot ===
win.nextRow()
fft_plot = win.addPlot(title="FFT (Frequency Domain)")
fft_plot.setLogMode(False, False)
fft_plot.setYRange(0, 1)
fft_plot.setXRange(0, 2000)
fft_plot.setMouseEnabled(x=False, y=False)
fft_plot.hideButtons()

# ✅ Smooth line
curve_fft = fft_plot.plot(pen=pg.mkPen(color='m', width=2))  # Magenta line

# === Audio Callback ===
def audio_callback(indata, frames, time, status):
    global latest_waveform, latest_fft, fft_smooth
    if status:
        print(status)

    samples = indata[:, 0].copy()
    latest_waveform = samples

    # Apply Hann window
    windowed = samples * np.hanning(len(samples))

    # FFT magnitude
    fft_vals = np.abs(np.fft.rfft(windowed))
    max_val = np.max(fft_vals)

    # Normalize safely
    if max_val > 1e-6:
        fft_vals /= max_val
    else:
        fft_vals[:] = 0

    # Exponential smoothing
    smoothing = 0.8
    fft_smooth = (smoothing * fft_smooth) + ((1 - smoothing) * fft_vals)
    latest_fft = fft_smooth.copy()

# === GUI Update Timer ===
def update_plot():
    global latest_waveform, latest_fft

    # --- Waveform ---
    t = np.linspace(0, blocksize / samplerate, blocksize)
    curve_waveform.setData(t, latest_waveform)

    # --- FFT with interpolation ---
    mask = fft_freqs <= 2000
    x = fft_freqs[mask]
    y = latest_fft[mask]

    # Interpolate to more points
    x_interp = np.linspace(x[0], x[-1], 500)  # 500 smoother points
    y_interp = np.interp(x_interp, x, y)

    curve_fft.setData(x_interp, y_interp)

# === Start Audio Stream ===
stream = sd.InputStream(callback=audio_callback, channels=1,
                        samplerate=samplerate, blocksize=blocksize)
stream.start()

# === Refresh Timer (~60fps) ===
timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(16)

pg.exec()
