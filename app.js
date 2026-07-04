/**
 * Click Tally — Syringe Dial Click Counter
 *
 * Detects clicks in audio recordings of syringe dial turns using
 * amplitude envelope analysis and peak finding.
 */

// ============================================================
// DOM elements
// ============================================================

const fileInput = document.getElementById("file-input");
const recordBtn = document.getElementById("record-btn");
const sensitivitySlider = document.getElementById("sensitivity");
const sensitivityValue = document.getElementById("sensitivity-value");
const statusSection = document.getElementById("status");
const statusText = document.getElementById("status-text");
const resultsSection = document.getElementById("results");
const clickCountEl = document.getElementById("click-count");
const chartContainer = document.getElementById("chart-container");
const waveformCanvas = document.getElementById("waveform-canvas");
const cumulativeCanvas = document.getElementById("cumulative-canvas");

// ============================================================
// State
// ============================================================

let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let lastAudioBuffer = null;

// ============================================================
// Event listeners
// ============================================================

fileInput.addEventListener("change", handleFileUpload);
recordBtn.addEventListener("click", toggleRecording);
sensitivitySlider.addEventListener("input", handleSensitivityChange);

function handleSensitivityChange() {
    sensitivityValue.textContent = sensitivitySlider.value;
    // Re-process if we have audio loaded
    if (lastAudioBuffer) {
        processAudio(lastAudioBuffer);
    }
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    showStatus("Decoding audio...");

    try {
        const arrayBuffer = await file.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        audioCtx.close();

        lastAudioBuffer = audioBuffer;
        processAudio(audioBuffer);
    } catch (err) {
        showStatus("Error: Could not decode audio file. " + err.message);
    }
}

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        recordedChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach((track) => track.stop());
            const blob = new Blob(recordedChunks, { type: "audio/webm" });
            showStatus("Decoding recorded audio...");

            try {
                const arrayBuffer = await blob.arrayBuffer();
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
                audioCtx.close();

                lastAudioBuffer = audioBuffer;
                processAudio(audioBuffer);
            } catch (err) {
                showStatus("Error: Could not process recording. " + err.message);
            }
        };

        mediaRecorder.start();
        isRecording = true;
        recordBtn.textContent = "⏹️ Stop";
        recordBtn.classList.add("recording");
        showStatus("Recording... click Stop when done.");
    } catch (err) {
        showStatus("Error: Microphone access denied. " + err.message);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    isRecording = false;
    recordBtn.textContent = "🎙️ Record";
    recordBtn.classList.remove("recording");
}

// ============================================================
// Audio processing
// ============================================================

function processAudio(audioBuffer) {
    showStatus("Detecting clicks...");

    // Get mono channel data
    const samples = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;
    const duration = audioBuffer.duration;

    // Detect clicks
    const sensitivity = parseFloat(sensitivitySlider.value);
    const clickIndices = detectClicks(samples, sampleRate, sensitivity);

    // Show results
    showResults(clickIndices.length);
    drawVisualization(samples, sampleRate, duration, clickIndices);

    hideStatus();
}

/**
 * Detect click events in audio samples using amplitude envelope + peak finding.
 *
 * Algorithm:
 * 1. Compute amplitude envelope using a 1ms sliding window (uniform filter)
 * 2. Find peaks with minimum 30ms spacing and adaptive prominence threshold
 *
 * @param {Float32Array} samples - Normalized audio samples [-1, 1]
 * @param {number} sampleRate - Sample rate in Hz
 * @param {number} sensitivity - Prominence multiplier (lower = more sensitive)
 * @returns {number[]} Array of sample indices where clicks were detected
 */
function detectClicks(samples, sampleRate, sensitivity) {
    // 1. Compute amplitude envelope with ~1ms rectangular window
    const windowDurationMs = 1;
    const windowSize = Math.max(1, Math.round(sampleRate * windowDurationMs / 1000));
    const envelope = uniformFilter1d(samples, windowSize);

    // 2. Compute envelope statistics for adaptive thresholding
    const stats = computeStats(envelope);

    // 3. Find peaks with constraints
    const minDistanceMs = 30;
    const minDistanceSamples = Math.round(sampleRate * minDistanceMs / 1000);
    const prominenceThreshold = stats.median + sensitivity * stats.std;

    const clickIndices = findPeaks(envelope, minDistanceSamples, prominenceThreshold);

    return clickIndices;
}

/**
 * Compute a uniform (moving average) filter of the absolute signal.
 * Equivalent to scipy.ndimage.uniform_filter1d(np.abs(samples), size=windowSize)
 */
function uniformFilter1d(samples, windowSize) {
    const n = samples.length;
    const result = new Float32Array(n);
    const halfWin = Math.floor(windowSize / 2);

    // Compute prefix sum of absolute values for O(1) range queries
    const prefixSum = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) {
        prefixSum[i + 1] = prefixSum[i] + Math.abs(samples[i]);
    }

    // Compute centered moving average using prefix sums
    for (let i = 0; i < n; i++) {
        const winStart = Math.max(0, i - halfWin);
        const winEnd = Math.min(n - 1, i + halfWin);
        const winLen = winEnd - winStart + 1;
        const sum = prefixSum[winEnd + 1] - prefixSum[winStart];
        result[i] = sum / winLen;
    }

    return result;
}

/**
 * Compute median and standard deviation of an array.
 */
function computeStats(arr) {
    const n = arr.length;

    // For large arrays, sample for median computation (performance)
    let sorted;
    if (n > 100000) {
        // Sample every Nth element
        const step = Math.floor(n / 50000);
        const sampled = [];
        for (let i = 0; i < n; i += step) {
            sampled.push(arr[i]);
        }
        sampled.sort((a, b) => a - b);
        sorted = sampled;
    } else {
        sorted = Array.from(arr).sort((a, b) => a - b);
    }

    const median = sorted[Math.floor(sorted.length / 2)];

    // Standard deviation
    let sumSq = 0;
    let mean = 0;
    for (let i = 0; i < n; i++) {
        mean += arr[i];
    }
    mean /= n;
    for (let i = 0; i < n; i++) {
        const diff = arr[i] - mean;
        sumSq += diff * diff;
    }
    const std = Math.sqrt(sumSq / n);

    return { median, std, mean };
}

/**
 * Find peaks in a signal with minimum distance and prominence constraints.
 * Simplified version of scipy.signal.find_peaks.
 *
 * @param {Float32Array} signal - Input signal
 * @param {number} minDistance - Minimum samples between peaks
 * @param {number} prominenceThreshold - Minimum prominence required
 * @returns {number[]} Indices of detected peaks
 */
function findPeaks(signal, minDistance, prominenceThreshold) {
    const n = signal.length;

    // Step 1: Find all local maxima
    const candidates = [];
    for (let i = 1; i < n - 1; i++) {
        if (signal[i] > signal[i - 1] && signal[i] >= signal[i + 1]) {
            candidates.push(i);
        }
    }

    // Step 2: Filter by prominence
    const prominent = [];
    for (const idx of candidates) {
        const prom = computeProminence(signal, idx, n);
        if (prom >= prominenceThreshold) {
            prominent.push({ idx, height: signal[idx] });
        }
    }

    // Step 3: Apply minimum distance constraint (keep tallest peaks)
    // Sort by height descending
    prominent.sort((a, b) => b.height - a.height);

    const selected = [];
    const excluded = new Set();

    for (const peak of prominent) {
        if (excluded.has(peak.idx)) continue;
        selected.push(peak.idx);

        // Exclude nearby peaks
        for (let j = peak.idx - minDistance; j <= peak.idx + minDistance; j++) {
            if (j !== peak.idx) {
                excluded.add(j);
            }
        }
    }

    // Return sorted by time
    selected.sort((a, b) => a - b);
    return selected;
}

/**
 * Compute the prominence of a peak.
 * Prominence = peak height - max of the two minimum values
 * found by descending left and right until a higher peak is found.
 */
function computeProminence(signal, peakIdx, n) {
    const peakHeight = signal[peakIdx];

    // Search left for the lowest point before a higher peak
    let leftMin = peakHeight;
    for (let i = peakIdx - 1; i >= 0; i--) {
        if (signal[i] > peakHeight) break;
        if (signal[i] < leftMin) leftMin = signal[i];
    }

    // Search right for the lowest point before a higher peak
    let rightMin = peakHeight;
    for (let i = peakIdx + 1; i < n; i++) {
        if (signal[i] > peakHeight) break;
        if (signal[i] < rightMin) rightMin = signal[i];
    }

    // Prominence is height above the higher of the two reference levels
    const referenceLevel = Math.max(leftMin, rightMin);
    return peakHeight - referenceLevel;
}

// ============================================================
// Visualization
// ============================================================

function drawVisualization(samples, sampleRate, duration, clickIndices) {
    chartContainer.hidden = false;

    drawWaveform(samples, sampleRate, duration, clickIndices);
    drawCumulative(duration, sampleRate, clickIndices);
}

function drawWaveform(samples, sampleRate, duration, clickIndices) {
    const canvas = waveformCanvas;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 30, bottom: 25, left: 50, right: 20 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Clear
    ctx.fillStyle = "#16213e";
    ctx.fillRect(0, 0, width, height);

    // Title
    ctx.fillStyle = "#90a4ae";
    ctx.font = "12px -apple-system, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Waveform with detected clicks", padding.left, 18);

    // Draw click markers first (behind waveform)
    ctx.strokeStyle = "rgba(239, 83, 80, 0.4)";
    ctx.lineWidth = 1;
    for (const idx of clickIndices) {
        const t = idx / sampleRate;
        const x = padding.left + (t / duration) * plotWidth;
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, padding.top + plotHeight);
        ctx.stroke();
    }

    // Downsample waveform for drawing
    const samplesPerPixel = Math.ceil(samples.length / plotWidth);
    ctx.strokeStyle = "#4fc3f7";
    ctx.lineWidth = 0.8;
    ctx.beginPath();

    for (let px = 0; px < plotWidth; px++) {
        const startSample = Math.floor((px / plotWidth) * samples.length);
        const endSample = Math.min(startSample + samplesPerPixel, samples.length);

        let min = Infinity;
        let max = -Infinity;
        for (let i = startSample; i < endSample; i++) {
            if (samples[i] < min) min = samples[i];
            if (samples[i] > max) max = samples[i];
        }

        const x = padding.left + px;
        const yMin = padding.top + (1 - (min + 1) / 2) * plotHeight;
        const yMax = padding.top + (1 - (max + 1) / 2) * plotHeight;

        if (px === 0) {
            ctx.moveTo(x, yMax);
        }
        ctx.lineTo(x, yMax);
        ctx.lineTo(x, yMin);
    }
    ctx.stroke();

    // Y-axis
    ctx.strokeStyle = "#455a64";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + plotHeight);
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle = "#607d8b";
    ctx.font = "10px -apple-system, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("1.0", padding.left - 5, padding.top + 4);
    ctx.fillText("0", padding.left - 5, padding.top + plotHeight / 2 + 4);
    ctx.fillText("-1.0", padding.left - 5, padding.top + plotHeight + 4);

    // X-axis (time)
    drawTimeAxis(ctx, padding, plotWidth, plotHeight, duration);
}

function drawCumulative(duration, sampleRate, clickIndices) {
    const canvas = cumulativeCanvas;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 30, bottom: 25, left: 50, right: 20 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Clear
    ctx.fillStyle = "#16213e";
    ctx.fillRect(0, 0, width, height);

    // Title
    ctx.fillStyle = "#90a4ae";
    ctx.font = "12px -apple-system, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Cumulative click count", padding.left, 18);

    // Grid lines
    const totalClicks = clickIndices.length;
    const gridSteps = Math.min(5, totalClicks);
    ctx.strokeStyle = "#263238";
    ctx.lineWidth = 0.5;
    ctx.fillStyle = "#607d8b";
    ctx.font = "10px -apple-system, sans-serif";
    ctx.textAlign = "right";

    for (let i = 0; i <= gridSteps; i++) {
        const value = Math.round((i / gridSteps) * totalClicks);
        const y = padding.top + plotHeight - (value / totalClicks) * plotHeight;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(padding.left + plotWidth, y);
        ctx.stroke();
        ctx.fillText(value.toString(), padding.left - 5, y + 4);
    }

    // Draw cumulative step function
    if (clickIndices.length > 0) {
        ctx.strokeStyle = "#66bb6a";
        ctx.lineWidth = 2;
        ctx.beginPath();

        // Start at 0
        const firstX = padding.left + (clickIndices[0] / sampleRate / duration) * plotWidth;
        ctx.moveTo(padding.left, padding.top + plotHeight);
        ctx.lineTo(firstX, padding.top + plotHeight);

        for (let i = 0; i < clickIndices.length; i++) {
            const t = clickIndices[i] / sampleRate;
            const x = padding.left + (t / duration) * plotWidth;
            const y = padding.top + plotHeight - ((i + 1) / totalClicks) * plotHeight;

            // Step up
            ctx.lineTo(x, y);

            // Horizontal to next click (or end)
            if (i < clickIndices.length - 1) {
                const nextT = clickIndices[i + 1] / sampleRate;
                const nextX = padding.left + (nextT / duration) * plotWidth;
                ctx.lineTo(nextX, y);
            } else {
                // Extend to end
                ctx.lineTo(padding.left + plotWidth, y);
            }
        }
        ctx.stroke();
    }

    // Y-axis
    ctx.strokeStyle = "#455a64";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + plotHeight);
    ctx.stroke();

    // X-axis (time)
    drawTimeAxis(ctx, padding, plotWidth, plotHeight, duration);
}

function drawTimeAxis(ctx, padding, plotWidth, plotHeight, duration) {
    ctx.strokeStyle = "#455a64";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top + plotHeight);
    ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
    ctx.stroke();

    // Time tick marks
    const tickInterval = duration <= 5 ? 1 : duration <= 20 ? 2 : 5;
    ctx.fillStyle = "#607d8b";
    ctx.font = "10px -apple-system, sans-serif";
    ctx.textAlign = "center";

    for (let t = 0; t <= duration; t += tickInterval) {
        const x = padding.left + (t / duration) * plotWidth;
        ctx.beginPath();
        ctx.moveTo(x, padding.top + plotHeight);
        ctx.lineTo(x, padding.top + plotHeight + 5);
        ctx.stroke();
        ctx.fillText(t + "s", x, padding.top + plotHeight + 17);
    }
}

// ============================================================
// UI helpers
// ============================================================

function showStatus(msg) {
    statusSection.hidden = false;
    statusText.textContent = msg;
    resultsSection.hidden = true;
}

function hideStatus() {
    statusSection.hidden = true;
}

function showResults(count) {
    resultsSection.hidden = false;
    clickCountEl.textContent = count;
}

// Handle window resize — redraw if we have data
window.addEventListener("resize", () => {
    if (lastAudioBuffer) {
        const sensitivity = parseFloat(sensitivitySlider.value);
        const samples = lastAudioBuffer.getChannelData(0);
        const clickIndices = detectClicks(samples, lastAudioBuffer.sampleRate, sensitivity);
        drawVisualization(samples, lastAudioBuffer.sampleRate, lastAudioBuffer.duration, clickIndices);
    }
});
