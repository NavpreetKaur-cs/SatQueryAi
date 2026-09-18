export default function Header({ mode, onModeChange }) {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-graphite-800 bg-graphite-900 px-4">
      <div className="flex items-baseline gap-2">
        <span className="font-display text-lg font-600 tracking-tight text-paper">SatQuery</span>
        <span className="font-data text-[11px] text-signal">AI</span>
      </div>

      <div className="flex overflow-hidden rounded-sm border border-graphite-700">
        {[
          { key: 'single', label: 'Single image' },
          { key: 'compare', label: '2 images' },
        ].map((opt) => (
          <button
            key={opt.key}
            onClick={() => onModeChange(opt.key)}
            className={`px-3 py-1.5 font-body text-sm transition-colors ${
              mode === opt.key
                ? 'bg-signal/15 text-signal'
                : 'bg-transparent text-fog-300 hover:text-paper'
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      <a
        href="https://github.com"
        target="_blank"
        rel="noreferrer"
        className="font-data text-[11px] text-fog-300 hover:text-signal"
      >
        SIH26167 · demo build
      </a>
    </header>
  );
}
