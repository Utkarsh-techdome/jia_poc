/**
 * Hands-free controls.
 * One toggle starts/stops the conversation. Once on, the app listens for you
 * automatically (VAD): start talking and it captures; stop and the AI replies.
 * A live level ring shows your mic picking you up.
 */
export default function Controls({ status, micOn, micLevel, onToggleMic, onReset }) {
  const connecting = status === 'connecting'
  const disconnected = status === 'error'

  let hint = 'Tap to start the conversation'
  if (disconnected) hint = 'Disconnected'
  else if (connecting) hint = 'Connecting…'
  else if (micOn) {
    if (status === 'listening') hint = 'Listening… speak naturally'
    else if (status === 'thinking') hint = 'Thinking…'
    else if (status === 'speaking') hint = 'Speaking — just talk to interrupt'
    else hint = 'Listening… go ahead, start talking'
  }

  // Ring scales with your live mic level when armed.
  const ringScale = micOn ? 1 + Math.min(micLevel, 1) * 0.35 : 1

  return (
    <div className="controls">
      <button
        className={`talk-btn talk-btn-${micOn ? status : 'off'} ${micOn ? 'is-on' : ''}`}
        onClick={onToggleMic}
        disabled={connecting || disconnected}
        aria-pressed={micOn}
      >
        <span
          className="talk-btn-ring"
          style={{ transform: `scale(${ringScale})` }}
        />
        <span className="talk-btn-icon">
          {micOn ? <MicActiveIcon /> : <MicIcon />}
        </span>
      </button>

      <div className="controls-text">
        <span className="talk-btn-label">{hint}</span>
        <button
          className="reset-btn"
          onClick={onReset}
          disabled={connecting || disconnected}
        >
          Restart interview
        </button>
      </div>
    </div>
  )
}

function MicIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  )
}

function MicActiveIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="currentColor">
      <rect x="9" y="2" width="6" height="13" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" />
      <line x1="12" y1="19" x2="12" y2="23" stroke="currentColor" strokeWidth="2" />
      <line x1="8" y1="23" x2="16" y2="23" stroke="currentColor" strokeWidth="2" />
    </svg>
  )
}
