/**
 * Hook that manages:
 *  - WebSocket lifecycle
 *  - Hands-free voice turn-taking via VAD (no buttons to hold)
 *  - Recording state (idle | listening | thinking | speaking)
 *  - Transcript log
 *  - Audio playback queue
 *  - Live mouth amplitude for the avatar
 *
 * Flow (fully automatic, with barge-in):
 *  1. User turns the mic on once.
 *  2. VAD watches the mic. When you start talking -> capture begins.
 *     If the AI is mid-sentence, your speech interrupts it (barge-in).
 *  3. When you go quiet for ~1s -> capture stops, audio is sent.
 *  4. AI thinks, then speaks. When it finishes, the mic auto-arms again.
 */
import { useEffect, useRef, useState, useCallback } from 'react'
import { AudioRecorder, VoiceActivityDetector, QueuedAudioPlayer } from '../utils/audio.js'

const WS_URL = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`

export function useInterview() {
  // states: 'connecting' | 'idle' | 'listening' | 'thinking' | 'speaking' | 'error'
  const [status, setStatus] = useState('connecting')
  const [transcript, setTranscript] = useState([])  // [{role, text, meta?}]
  const [error, setError] = useState(null)
  const [amplitude, setAmplitude] = useState(0)       // avatar mouth (AI speaking)
  const [micLevel, setMicLevel] = useState(0)         // your live mic level
  const [partialAI, setPartialAI] = useState('')      // currently speaking sentence
  const [micOn, setMicOn] = useState(false)

  const wsRef = useRef(null)
  const recorderRef = useRef(null)
  const vadRef = useRef(null)
  const playerRef = useRef(null)
  const pendingSentenceTextRef = useRef('')
  const pendingFullTextRef = useRef('')
  const capturingRef = useRef(false)   // are we recording the user right now?
  const statusRef = useRef('connecting')
  const micOnRef = useRef(false)
  const closeTimerRef = useRef(null)   // deferred WS close (StrictMode guard)

  // Keep refs of reactive values so stable callbacks (VAD/WS) read the latest.
  useEffect(() => { statusRef.current = status }, [status])
  useEffect(() => { micOnRef.current = micOn }, [micOn])

  // ---- Setup audio player once ----
  useEffect(() => {
    const player = new QueuedAudioPlayer()
    player.onAmplitudeChange = setAmplitude
    player.onPlaybackEnd = () => {
      vadRef.current?.setAiSpeaking(false)
      // Commit the full AI response to the transcript now that audio is done,
      // so it never overlaps with the partialAI sentence display.
      if (pendingFullTextRef.current) {
        const text = pendingFullTextRef.current
        pendingFullTextRef.current = ''
        setTranscript((t) => [...t, { role: 'assistant', text }])
      }
      setPartialAI('')
      setStatus('idle')
    }
    playerRef.current = player
  }, [])

  // ---- WebSocket connection ----
  useEffect(() => {
    // React 18 StrictMode mounts effects twice in dev (mount → cleanup →
    // mount), which would open two sockets — a connect/disconnect storm with
    // duplicate greetings. Reuse a still-good socket and cancel its pending
    // close instead of opening another.
    const existing = wsRef.current
    if (existing && (existing.readyState === WebSocket.OPEN ||
                     existing.readyState === WebSocket.CONNECTING)) {
      if (closeTimerRef.current) {
        clearTimeout(closeTimerRef.current)
        closeTimerRef.current = null
      }
      return
    }

    const ws = new WebSocket(WS_URL)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onopen = () => setStatus('idle')
    ws.onerror = () => {
      setError('Could not connect to backend at ' + WS_URL)
      setStatus('error')
    }
    ws.onclose = () => {
      setStatus((s) => (s === 'error' ? s : 'error'))
      if (!error) setError('Connection closed.')
    }

    ws.onmessage = async (evt) => {
      // Binary => audio chunk
      if (evt.data instanceof ArrayBuffer) {
        // Don't play the AI over the user if they've barged in.
        if (capturingRef.current) return
        // The AI is about to make sound — raise the VAD bar so the playback
        // leaking into the mic can't self-trigger the listening state.
        vadRef.current?.setAiSpeaking(true)
        setStatus('speaking')
        if (pendingSentenceTextRef.current) {
          setPartialAI(pendingSentenceTextRef.current)
        }
        await playerRef.current.enqueue(evt.data)
        return
      }

      let msg
      try { msg = JSON.parse(evt.data) } catch { return }

      switch (msg.type) {
        case 'ai_text':
          pendingFullTextRef.current = msg.text
          break
        case 'user_text':
          setTranscript((t) => [
            ...t,
            { role: 'user', text: msg.text, meta: `STT ${msg.stt_ms}ms` },
          ])
          break
        case 'tts_chunk_start':
          pendingSentenceTextRef.current = msg.text
          break
        case 'ai_text_final':
          pendingFullTextRef.current = msg.text
          break
        case 'turn_end':
          break
        case 'error':
          // e.g. "I didn't catch that" — just re-arm and keep going.
          if (msg.message && !/didn't catch/i.test(msg.message)) {
            setError(msg.message)
          }
          if (!micOnRef.current) setStatus('idle')
          break
        default:
          break
      }
    }

    // Defer the close. On a StrictMode remount the effect re-runs right after
    // this cleanup and cancels the timer (above). Only a real unmount lets it
    // fire, actually closing the socket.
    return () => {
      closeTimerRef.current = setTimeout(() => {
        if (wsRef.current === ws) {
          ws.close()
          wsRef.current = null
        }
        closeTimerRef.current = null
      }, 150)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---- Send a completed utterance to the backend ----
  const sendUtterance = useCallback(async () => {
    const blob = await recorderRef.current.stopCapture()
    capturingRef.current = false
    if (!blob || blob.size < 1200) {
      // Too little audio — ignore and stay armed.
      setStatus('idle')
      return
    }
    setStatus('thinking')
    const arrayBuf = await blob.arrayBuffer()
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(arrayBuf)
    }
  }, [])

  // ---- VAD callbacks ----
  const handleSpeechStart = useCallback(() => {
    if (statusRef.current === 'thinking') return  // don't capture while waiting for AI
    // Barge-in: if the AI is speaking, cut it off and listen.
    if (playerRef.current?.active) {
      playerRef.current.clear()
      // We cleared playback ourselves, so onPlaybackEnd won't fire — drop the
      // VAD back to normal sensitivity for the rest of this utterance.
      vadRef.current?.setAiSpeaking(false)
      setPartialAI('')
    }
    if (!recorderRef.current.isCapturing) {
      recorderRef.current.startCapture()
      capturingRef.current = true
      setStatus('listening')
    }
  }, [])

  const handleSpeechEnd = useCallback((discard = false) => {
    if (!capturingRef.current) return
    if (discard) {
      // Was just noise — throw it away, stay armed.
      recorderRef.current.stopCapture().then(() => {
        capturingRef.current = false
        if (statusRef.current === 'listening') setStatus('idle')
      })
      return
    }
    sendUtterance()
  }, [sendUtterance])

  // ---- Turn the hands-free mic on/off ----
  const enableMic = useCallback(async () => {
    try {
      recorderRef.current = recorderRef.current || new AudioRecorder()
      const stream = await recorderRef.current.acquire()
      vadRef.current = new VoiceActivityDetector(stream, {
        onSpeechStart: handleSpeechStart,
        onSpeechEnd: handleSpeechEnd,
        onLevel: setMicLevel,
        speechThreshold: 0.10,    // base sensitivity (mic quiet, AI not talking)
        bargeInThreshold: 0.20,   // must be louder to interrupt the AI mid-sentence
        silenceMs: 1100,
        minSpeechMs: 250,
        startSpeechMs: 180,       // sustain above threshold this long before a turn starts
      })
      vadRef.current.start()
      setMicOn(true)
      setError(null)
      setStatus('idle')
      // Ask backend to send the greeting now that the user has started.
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'start' }))
      }
    } catch (e) {
      setError('Microphone permission denied or not available.')
      setMicOn(false)
    }
  }, [handleSpeechStart, handleSpeechEnd])

  const disableMic = useCallback(() => {
    vadRef.current?.stop()
    vadRef.current = null
    recorderRef.current?.release()
    recorderRef.current = null
    capturingRef.current = false
    setMicOn(false)
    setMicLevel(0)
    setStatus('idle')
  }, [])

  const toggleMic = useCallback(() => {
    if (micOnRef.current) disableMic()
    else enableMic()
  }, [enableMic, disableMic])

  const reset = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
    playerRef.current?.clear()
    capturingRef.current = false
    if (recorderRef.current?.isCapturing) recorderRef.current.stopCapture()
    setTranscript([])
    setPartialAI('')
    setStatus('idle')
    // Backend will re-send the greeting in response to reset.
    wsRef.current.send(JSON.stringify({ type: 'reset' }))
  }, [])

  // Cleanup on unmount
  useEffect(() => () => {
    vadRef.current?.stop()
    recorderRef.current?.release()
  }, [])

  return {
    status,
    transcript,
    error,
    amplitude,
    micLevel,
    partialAI,
    micOn,
    toggleMic,
    reset,
  }
}
