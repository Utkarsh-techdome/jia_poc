import { useEffect, useRef } from 'react'

export default function Transcript({ entries, partialAI }) {
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [entries, partialAI])

  return (
    <div className="transcript" ref={scrollRef}>
      {entries.length === 0 && !partialAI && (
        <div className="transcript-empty">
          Press <kbd>Hold to speak</kbd> below to begin.
        </div>
      )}
      {entries.map((e, i) => (
        <div key={i} className={`turn turn-${e.role}`}>
          <div className="turn-label">{e.role === 'user' ? 'You' : 'Interviewer'}</div>
          <div className="turn-text">{e.text}</div>
          {e.meta && <div className="turn-meta">{e.meta}</div>}
        </div>
      ))}
      {partialAI && (
        <div className="turn turn-assistant turn-partial">
          <div className="turn-label">Interviewer</div>
          <div className="turn-text">{partialAI}<span className="cursor">▌</span></div>
        </div>
      )}
    </div>
  )
}
