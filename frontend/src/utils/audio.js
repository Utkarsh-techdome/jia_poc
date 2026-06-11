/**
 * Audio recording (MediaRecorder) + queued WAV playback (Web Audio API)
 * + a microphone Voice Activity Detector (VAD) for hands-free turn-taking.
 *
 * The player exposes an `analyser` so the avatar can react to amplitude.
 */

// --------- Recorder ---------
// Holds a single persistent mic stream. The VAD watches that stream's level
// continuously; recording is started/stopped around detected speech so we
// never have to re-request mic permission per turn.
export class AudioRecorder {
  constructor() {
    this.mediaRecorder = null
    this.chunks = []
    this.stream = null
    this.mime = null
  }

  /** Acquire the mic once and keep it open for the whole session. */
  async acquire() {
    if (this.stream) return this.stream
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    })
    this.mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm'
    return this.stream
  }

  /** Begin capturing a single utterance. Requires acquire() first. */
  startCapture() {
    if (!this.stream) throw new Error('AudioRecorder.acquire() must be called first')
    this.chunks = []
    this.mediaRecorder = new MediaRecorder(this.stream, { mimeType: this.mime })
    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.chunks.push(e.data)
    }
    this.mediaRecorder.start()
  }

  /** Stop the current utterance; resolves with the recorded Blob. */
  stopCapture() {
    return new Promise((resolve) => {
      const mr = this.mediaRecorder
      if (!mr || mr.state === 'inactive') return resolve(null)
      mr.onstop = () => {
        const blob = new Blob(this.chunks, { type: mr.mimeType })
        this.mediaRecorder = null
        resolve(blob)
      }
      mr.stop()
    })
  }

  get isCapturing() {
    return !!this.mediaRecorder && this.mediaRecorder.state === 'recording'
  }

  /** Fully release the mic (end of session). */
  release() {
    try { this.mediaRecorder?.stop() } catch { /* ignore */ }
    this.mediaRecorder = null
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop())
      this.stream = null
    }
  }
}

// --------- Voice Activity Detector ---------
// Watches a mic MediaStream's RMS level and fires callbacks when the user
// starts and stops talking. Pure Web Audio API — no external deps.
export class VoiceActivityDetector {
  /**
   * @param {MediaStream} stream  the live mic stream
   * @param {object} opts
   *   onSpeechStart()  - fired when sustained speech begins
   *   onSpeechEnd()    - fired after `silenceMs` of quiet following speech
   *   onLevel(level)   - fired every frame with the current 0..1 level
   *   speechThreshold  - RMS level (0..1) that counts as "talking"
   *   bargeInThreshold - higher RMS required to trigger while the AI is speaking
   *                      (so the AI's own playback leaking into the mic can't
   *                      self-trigger; a real, louder human voice still can)
   *   silenceMs        - how long to wait after speech before ending the turn
   *   minSpeechMs      - ignore blips shorter than this (avoids false triggers)
   *   startSpeechMs    - sustained speech (ms) required before a turn actually
   *                      begins — debounces brief speaker bleed / keyboard taps
   */
  constructor(stream, opts = {}) {
    this.stream = stream
    this.onSpeechStart = opts.onSpeechStart || (() => {})
    this.onSpeechEnd = opts.onSpeechEnd || (() => {})
    this.onLevel = opts.onLevel || (() => {})
    this.speechThreshold = opts.speechThreshold ?? 0.04
    this.bargeInThreshold = opts.bargeInThreshold ?? 0.16
    this.silenceMs = opts.silenceMs ?? 1100
    this.minSpeechMs = opts.minSpeechMs ?? 250
    this.startSpeechMs = opts.startSpeechMs ?? 180

    this.ctx = null
    this.analyser = null
    this.source = null
    this._rafId = null
    this._active = false       // currently inside a speech segment
    this._enabled = false      // VAD is armed and watching
    this._aiSpeaking = false   // is the AI currently playing audio?
    this._candidateAt = 0      // when the current above-threshold run began
    this._speechStartAt = 0
    this._lastVoiceAt = 0
  }

  /**
   * Tell the VAD whether the AI is currently speaking. While true, the bar to
   * trigger is raised to `bargeInThreshold` so the AI's own voice (echoed back
   * through the mic) does not flip us into the listening state.
   */
  setAiSpeaking(speaking) {
    this._aiSpeaking = speaking
  }

  start() {
    if (this._enabled) return
    this.ctx = new (window.AudioContext || window.webkitAudioContext)()
    this.source = this.ctx.createMediaStreamSource(this.stream)
    this.analyser = this.ctx.createAnalyser()
    this.analyser.fftSize = 512
    this.analyser.smoothingTimeConstant = 0.6
    this.source.connect(this.analyser)
    this._enabled = true
    this._active = false
    this._loop()
  }

  _loop() {
    const data = new Uint8Array(this.analyser.frequencyBinCount)
    const tick = () => {
      if (!this._enabled) return
      this.analyser.getByteTimeDomainData(data)
      let sum = 0
      for (let i = 0; i < data.length; i++) {
        const v = (data[i] - 128) / 128
        sum += v * v
      }
      const rms = Math.sqrt(sum / data.length)
      this.onLevel(Math.min(1, rms * 3))

      const now = performance.now()
      // While the AI is talking, require a louder signal so its own playback
      // bleeding into the mic can't trip the detector — only a real, deliberate
      // human voice (a genuine barge-in) clears the higher bar.
      const threshold = this._aiSpeaking ? this.bargeInThreshold : this.speechThreshold

      if (rms >= threshold) {
        this._lastVoiceAt = now
        if (!this._active) {
          // Require the signal to stay above threshold for `startSpeechMs`
          // before we actually begin a turn. This debounces short transients
          // (speaker bleed, a click, a cough) that briefly cross the line.
          if (this._candidateAt === 0) this._candidateAt = now
          if (now - this._candidateAt >= this.startSpeechMs) {
            this._active = true
            this._candidateAt = 0
            this._speechStartAt = now
            this.onSpeechStart()
          }
        }
      } else if (this._active) {
        // We were in speech; check if we've been silent long enough to end.
        const silentFor = now - this._lastVoiceAt
        const spokeFor = this._lastVoiceAt - this._speechStartAt
        if (silentFor >= this.silenceMs) {
          this._active = false
          // Only count it as a real turn if they spoke long enough.
          if (spokeFor >= this.minSpeechMs) {
            this.onSpeechEnd()
          } else {
            // Too short — treat as noise, silently re-arm.
            this.onSpeechEnd(true /* discard */)
          }
        }
      } else {
        // Below threshold and not in a turn — the run of loud frames ended
        // before it lasted long enough, so reset the start-debounce.
        this._candidateAt = 0
      }
      this._rafId = requestAnimationFrame(tick)
    }
    this._rafId = requestAnimationFrame(tick)
  }

  /** Temporarily ignore input (e.g. while we process a turn). Keeps ctx alive. */
  pause() {
    this._enabled = false
    this._active = false
    if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null }
    this.onLevel(0)
  }

  /** Resume watching after a pause. */
  resume() {
    if (this._enabled) return
    this._enabled = true
    this._active = false
    this._candidateAt = 0
    this._lastVoiceAt = 0
    this._loop()
  }

  stop() {
    this.pause()
    try { this.source?.disconnect() } catch { /* ignore */ }
    if (this.ctx && this.ctx.state !== 'closed') {
      this.ctx.close().catch(() => {})
    }
    this.ctx = null
    this.analyser = null
    this.source = null
  }
}

// --------- Queued WAV Player ---------
// Receives WAV byte chunks (one per sentence), plays them back-to-back,
// and exposes a real-time amplitude value (0..1) for the avatar mouth.
export class QueuedAudioPlayer {
  constructor() {
    this.ctx = null
    this.analyser = null
    this.gainNode = null
    this.queue = []          // ArrayBuffer queue
    this.isPlaying = false
    this.currentSource = null
    this.onAmplitudeChange = null   // callback(level: 0..1)
    this.onPlaybackEnd = null       // callback()
    this._rafId = null
  }

  _ensureContext() {
    if (this.ctx) return
    this.ctx = new (window.AudioContext || window.webkitAudioContext)()
    this.analyser = this.ctx.createAnalyser()
    this.analyser.fftSize = 256
    this.gainNode = this.ctx.createGain()
    this.analyser.connect(this.gainNode)
    this.gainNode.connect(this.ctx.destination)
  }

  async enqueue(arrayBuffer) {
    this._ensureContext()
    this.queue.push(arrayBuffer)
    if (!this.isPlaying) {
      this._playNext()
    }
  }

  async _playNext() {
    if (this.queue.length === 0) {
      this.isPlaying = false
      this.currentSource = null
      this._stopAmplitudeLoop()
      if (this.onPlaybackEnd) this.onPlaybackEnd()
      return
    }
    this.isPlaying = true
    const buf = this.queue.shift()

    let decoded
    try {
      decoded = await this.ctx.decodeAudioData(buf.slice(0))
    } catch (err) {
      console.error('decodeAudioData failed', err)
      return this._playNext()
    }

    const source = this.ctx.createBufferSource()
    source.buffer = decoded
    source.connect(this.analyser)
    source.onended = () => {
      if (source === this.currentSource) this._playNext()
    }
    this.currentSource = source
    source.start()
    this._startAmplitudeLoop()
  }

  /** True while audio is queued or actively playing. */
  get active() {
    return this.isPlaying || this.queue.length > 0
  }

  _startAmplitudeLoop() {
    if (this._rafId) return
    const data = new Uint8Array(this.analyser.frequencyBinCount)
    const tick = () => {
      this.analyser.getByteTimeDomainData(data)
      // RMS calc for amplitude
      let sum = 0
      for (let i = 0; i < data.length; i++) {
        const v = (data[i] - 128) / 128
        sum += v * v
      }
      const rms = Math.sqrt(sum / data.length)
      // Boost a bit so the mouth feels responsive
      const level = Math.min(1, rms * 3)
      if (this.onAmplitudeChange) this.onAmplitudeChange(level)
      this._rafId = requestAnimationFrame(tick)
    }
    this._rafId = requestAnimationFrame(tick)
  }

  _stopAmplitudeLoop() {
    if (this._rafId) {
      cancelAnimationFrame(this._rafId)
      this._rafId = null
    }
    if (this.onAmplitudeChange) this.onAmplitudeChange(0)
  }

  /** Stop playback immediately and drop anything queued (barge-in). */
  clear() {
    this.queue = []
    if (this.currentSource) {
      try {
        this.currentSource.onended = null
        this.currentSource.stop()
      } catch { /* ignore */ }
      this.currentSource = null
    }
    this.isPlaying = false
    this._stopAmplitudeLoop()
  }
}
