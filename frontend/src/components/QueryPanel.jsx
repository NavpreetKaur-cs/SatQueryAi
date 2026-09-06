import { useEffect, useRef, useState } from 'react';
import QueryLogEntry from './QueryLogEntry';
import { SAMPLE_QUERIES } from '../data/sampleQueries';

export default function QueryPanel({
  log,
  onSubmit,
  disabled,
  disabledReason,
  onFocusRegion,
  focusedRegionId,
  mode,
}) {
  const [text, setText] = useState('');
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [log]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setText('');
  };

  const applicableSamples = SAMPLE_QUERIES.filter(
    (s) => !s.requiresCompare || mode === 'compare'
  );

  return (
    <div className="flex h-full flex-col bg-graphite-900">
      <div className="flex h-9 shrink-0 items-center border-b border-graphite-800 px-3">
        <span className="font-data text-[11px] tracking-wide text-fog-300">QUERY LOG</span>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        {log.length === 0 && (
          <div className="px-3 py-4">
            <p className="font-body text-[13px] text-fog-300">
              Load a scene, then ask a question about it. Try one of these:
            </p>
            <div className="mt-3 flex flex-col gap-1.5">
              {applicableSamples.map((s) => (
                <button
                  key={s.text}
                  onClick={() => !disabled && onSubmit(s.text)}
                  disabled={disabled}
                  className="rounded-sm border border-graphite-700 px-2.5 py-2 text-left font-body text-[12px] text-fog-200 transition-colors hover:border-signal/50 hover:text-paper disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {s.text}
                </button>
              ))}
            </div>
          </div>
        )}
        {log.map((entry) => (
          <QueryLogEntry
            key={entry.id}
            entry={entry}
            onFocusRegion={onFocusRegion}
            focusedRegionId={focusedRegionId}
          />
        ))}
      </div>

      <form onSubmit={handleSubmit} className="shrink-0 border-t border-graphite-800 p-2.5">
        {disabled && disabledReason && (
          <p className="mb-1.5 font-data text-[11px] text-amber">{disabledReason}</p>
        )}
        <div className="flex gap-1.5">
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={disabled}
            placeholder="Ask about the scene…"
            className="flex-1 rounded-sm border border-graphite-700 bg-graphite-950 px-2.5 py-2 font-body text-[13px] text-paper placeholder:text-fog-300 focus:border-signal disabled:opacity-40"
          />
          <button
            type="submit"
            disabled={disabled || !text.trim()}
            className="rounded-sm bg-signal px-3 py-2 font-body text-[13px] font-medium text-graphite-950 transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-30"
          >
            Ask
          </button>
        </div>
      </form>
    </div>
  );
}
