export default function QueryLogEntry({ entry, onFocusRegion, focusedRegionId }) {
  if (entry.role === 'user') {
    return (
      <div className="border-b border-graphite-800 px-3 py-2.5">
        <div className="flex items-start gap-2">
          <span className="font-data text-signal">{'>'}</span>
          <p className="font-body text-[13px] leading-snug text-paper">{entry.text}</p>
        </div>
      </div>
    );
  }

  if (entry.role === 'error') {
    return (
      <div className="border-b border-graphite-800 px-3 py-2.5">
        <p className="font-body text-[13px] leading-snug text-alert">{entry.text}</p>
      </div>
    );
  }

  if (entry.role === 'pending') {
    return (
      <div className="border-b border-graphite-800 px-3 py-2.5">
        <p className="font-data text-[12px] text-fog-300">
          <span className="inline-block animate-pulse">analyzing scene…</span>
        </p>
      </div>
    );
  }

  // assistant
  return (
    <div className="border-b border-graphite-800 px-3 py-2.5">
      {entry.queryType && entry.queryType !== 'unknown' && (
        <span className="mb-1.5 inline-block rounded-sm border border-graphite-700 px-1.5 py-0.5 font-data text-[10px] uppercase tracking-wide text-fog-300">
          classified as: {entry.queryType}
        </span>
      )}
      <p className="font-body text-[13px] leading-snug text-fog-200">{entry.text}</p>
      {typeof entry.confidence === 'number' && (
        <div className="mt-1.5 flex items-center gap-1.5">
          <div className="h-1 w-16 overflow-hidden rounded-sm bg-graphite-700">
            <div
              className="h-full bg-signal-dim"
              style={{ width: `${Math.round(entry.confidence * 100)}%` }}
            />
          </div>
          <span className="font-data text-[10px] text-fog-300">
            {Math.round(entry.confidence * 100)}% confidence
          </span>
        </div>
      )}
      {entry.regions && entry.regions.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {entry.regions.map((r) => (
            <button
              key={r.id}
              onClick={() => onFocusRegion(r.id)}
              className={`rounded-sm border px-1.5 py-0.5 font-data text-[10px] transition-colors ${
                focusedRegionId === r.id
                  ? 'border-signal text-signal'
                  : 'border-graphite-700 text-fog-300 hover:border-fog-300 hover:text-paper'
              }`}
              title="Locate this region on the image"
            >
              ▸ {r.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
