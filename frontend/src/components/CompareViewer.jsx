import { useRef, useState } from 'react';
import ImageViewer from './ImageViewer';

export default function CompareViewer({ before, after, regions, focusedRegionId, syncViews, onStatus }) {
  const mapsRef = useRef({ before: null, after: null });
  const applyingSync = useRef(false);

  const [, forceReady] = useState(0);

  const handleReady = (which) => (map) => {
    mapsRef.current[which] = map;
    forceReady((n) => n + 1);
  };

  const handleViewChange = (which) => (view) => {
    onStatus?.(view);
    if (!syncViews || applyingSync.current) return;
    const other = which === 'before' ? mapsRef.current.after : mapsRef.current.before;
    if (!other) return;
    applyingSync.current = true;
    other.setView(view.center, view.zoom, { animate: false });
    applyingSync.current = false;
  };

  const beforeRegions = regions.filter((r) => !r.image || r.image === 'before');
  const afterRegions = regions.filter((r) => r.image === 'after');

  return (
    <div className="grid h-full grid-cols-2 gap-px bg-graphite-800">
      <ImageViewer
        image={before}
        regions={beforeRegions}
        focusedRegionId={focusedRegionId}
        onReady={handleReady('before')}
        onViewChange={handleViewChange('before')}
        label="BEFORE"
      />
      <ImageViewer
        image={after}
        regions={afterRegions}
        focusedRegionId={focusedRegionId}
        onReady={handleReady('after')}
        onViewChange={handleViewChange('after')}
        label="AFTER"
      />
    </div>
  );
}
