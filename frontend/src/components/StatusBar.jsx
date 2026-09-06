export default function StatusBar({ scene, view, mode, syncViews, onToggleSync }) {
  return (
    <div className="flex h-8 shrink-0 items-center justify-between border-t border-graphite-800 bg-graphite-900 px-3 font-data text-[11px] text-fog-300">
      <div className="flex gap-4">
        <span>
          center {view ? `${view.center[0].toFixed(1)}, ${view.center[1].toFixed(1)}` : '—'}
        </span>
        <span>zoom {view ? view.zoom.toFixed(2) : '—'}</span>
        <span>{scene ? scene.sensor : 'no scene loaded'}</span>
        <span>{scene ? scene.capturedOn : ''}</span>
      </div>
      {mode === 'compare' && (
        <button
          onClick={onToggleSync}
          className={`rounded-sm border px-2 py-0.5 transition-colors ${
            syncViews
              ? 'border-signal/50 text-signal'
              : 'border-graphite-700 text-fog-300 hover:text-paper'
          }`}
        >
          sync views: {syncViews ? 'on' : 'off'}
        </button>
      )}
    </div>
  );
}
