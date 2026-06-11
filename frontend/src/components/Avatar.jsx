/**
 * SVG-based stylized avatar.
 * - Mouth opens/closes from `amplitude` (0..1) while speaking.
 * - Eyes blink occasionally; pupils dilate when listening.
 * - Soft glow + breathing scale based on status.
 *
 * Designed to be replaced later with a Ready Player Me + TalkingHead.js
 * integration without changing the API surface.
 */
import { useEffect, useState } from 'react'

export default function Avatar({ status, amplitude }) {
  const [blink, setBlink] = useState(false)
  const [breathe, setBreathe] = useState(0)

  // Random blinks
  useEffect(() => {
    let timeout
    const scheduleBlink = () => {
      timeout = setTimeout(() => {
        setBlink(true)
        setTimeout(() => setBlink(false), 120)
        scheduleBlink()
      }, 2500 + Math.random() * 3500)
    }
    scheduleBlink()
    return () => clearTimeout(timeout)
  }, [])

  // Soft breathing animation
  useEffect(() => {
    let raf
    const start = performance.now()
    const tick = (t) => {
      const elapsed = (t - start) / 1000
      setBreathe(Math.sin(elapsed * 1.2) * 0.5 + 0.5) // 0..1
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  // Mouth opening: 0..1 → pixels
  const mouthOpenPx = Math.max(2, amplitude * 28)
  const mouthY = 130
  const mouthW = 38 + amplitude * 8

  // Pupil size hint based on status
  const pupilR = status === 'listening' ? 5.5 : 4

  // Halo color per status
  const haloColor = {
    connecting: '#7a7367',
    idle: '#a89f8e',
    listening: '#d18b3a',
    thinking: '#9c7bd1',
    speaking: '#4a8c6f',
    error: '#c0584e',
  }[status] || '#a89f8e'

  // Scale a touch with breathing
  const scale = 1 + breathe * 0.012

  return (
    <div className="avatar-wrap">
      {/* Status halo */}
      <div
        className={`avatar-halo halo-${status}`}
        style={{ '--halo': haloColor }}
      />
      <svg
        viewBox="0 0 220 220"
        width="280"
        height="280"
        style={{ transform: `scale(${scale})`, transition: 'transform 0.15s linear' }}
      >
        <defs>
          <radialGradient id="faceGrad" cx="50%" cy="40%" r="60%">
            <stop offset="0%" stopColor="#f5ebd9" />
            <stop offset="100%" stopColor="#d6c5a8" />
          </radialGradient>
          <radialGradient id="cheekGrad" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#e8a585" stopOpacity="0.55" />
            <stop offset="100%" stopColor="#e8a585" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* Face */}
        <circle cx="110" cy="110" r="92" fill="url(#faceGrad)" />

        {/* Cheeks */}
        <ellipse cx="68" cy="135" rx="18" ry="11" fill="url(#cheekGrad)" />
        <ellipse cx="152" cy="135" rx="18" ry="11" fill="url(#cheekGrad)" />

        {/* Eyes */}
        <g>
          {/* Eye whites */}
          <ellipse cx="80" cy="100" rx="14" ry={blink ? 1 : 11} fill="#fdfaf3" />
          <ellipse cx="140" cy="100" rx="14" ry={blink ? 1 : 11} fill="#fdfaf3" />
          {/* Pupils (hidden during blink) */}
          {!blink && (
            <>
              <circle cx="80" cy="101" r={pupilR} fill="#1f1a14" />
              <circle cx="140" cy="101" r={pupilR} fill="#1f1a14" />
              <circle cx="82" cy="98" r="1.4" fill="#fdfaf3" />
              <circle cx="142" cy="98" r="1.4" fill="#fdfaf3" />
            </>
          )}
        </g>

        {/* Mouth - rectangle that grows with amplitude */}
        <rect
          x={110 - mouthW / 2}
          y={mouthY - mouthOpenPx / 2}
          width={mouthW}
          height={mouthOpenPx}
          rx={Math.min(mouthOpenPx / 2, 8)}
          fill="#3b2820"
          style={{ transition: 'all 60ms linear' }}
        />
        {/* Lip line when mostly closed */}
        {mouthOpenPx < 5 && (
          <line
            x1={110 - mouthW / 2}
            y1={mouthY}
            x2={110 + mouthW / 2}
            y2={mouthY}
            stroke="#8a5c4a"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        )}
      </svg>
      <div className="avatar-status">
        <span className={`status-dot status-dot-${status}`} />
        <span className="status-label">{statusLabel(status)}</span>
      </div>
    </div>
  )
}

function statusLabel(s) {
  switch (s) {
    case 'connecting': return 'Connecting'
    case 'idle': return 'Ready'
    case 'listening': return 'Listening'
    case 'thinking': return 'Thinking'
    case 'speaking': return 'Speaking'
    case 'error': return 'Disconnected'
    default: return s
  }
}
