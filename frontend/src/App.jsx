import Avatar from './components/Avatar.jsx'
import Transcript from './components/Transcript.jsx'
import Controls from './components/Controls.jsx'
import { useInterview } from './hooks/useInterview.js'

export default function App() {
  const {
    status, transcript, error, amplitude, micLevel, partialAI,
    micOn, toggleMic, reset,
  } = useInterview()

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" />
          <span className="brand-name">Interview<span className="brand-dot">.</span>POC</span>
        </div>
        <div className="header-meta">
          Self-hosted · STT + LLM + TTS pipeline
        </div>
      </header>

      <main className="main">
        <section className="avatar-col">
          <Avatar status={status} amplitude={amplitude} />
        </section>

        <section className="transcript-col">
          <Transcript entries={transcript} partialAI={partialAI} />
          {error && <div className="error-banner">{error}</div>}
          <Controls
            status={status}
            micOn={micOn}
            micLevel={micLevel}
            onToggleMic={toggleMic}
            onReset={reset}
          />
        </section>
      </main>

      <footer className="app-footer">
        <span>Hands-free: start the mic, then just talk. The AI listens, replies, and you can interrupt anytime.</span>
      </footer>
    </div>
  )
}
