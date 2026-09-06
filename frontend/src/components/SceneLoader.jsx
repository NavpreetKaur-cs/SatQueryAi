export default function SceneLoader({ mode, onUpload, onLoadDemo, hasScene }) {
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-graphite-800 bg-graphite-950 px-3 py-2">
      <label className="cursor-pointer rounded-sm border border-graphite-700 px-2.5 py-1 font-body text-[12px] text-fog-200 transition-colors hover:border-signal/50 hover:text-paper">
        Upload {mode === 'compare' ? 'before image' : 'image'}
        <input
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => e.target.files[0] && onUpload('before', e.target.files[0])}
        />
      </label>

      {mode === 'compare' && (
        <label className="cursor-pointer rounded-sm border border-graphite-700 px-2.5 py-1 font-body text-[12px] text-fog-200 transition-colors hover:border-signal/50 hover:text-paper">
          Upload after image
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => e.target.files[0] && onUpload('after', e.target.files[0])}
          />
        </label>
      )}

      <button
        onClick={onLoadDemo}
        className="rounded-sm border border-signal/40 px-2.5 py-1 font-body text-[12px] text-signal transition-colors hover:bg-signal/10"
      >
        Load demo scene
      </button>

      {hasScene && (
        <span className="ml-auto font-data text-[11px] text-fog-300">scene loaded</span>
      )}
    </div>
  );
}
