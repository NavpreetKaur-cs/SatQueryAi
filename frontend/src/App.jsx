import { useCallback, useState } from 'react';
import Header from './components/Header';
import SceneLoader from './components/SceneLoader';
import ImageViewer from './components/ImageViewer';
import CompareViewer from './components/CompareViewer';
import StatusBar from './components/StatusBar';
import QueryPanel from './components/QueryPanel';
import { useSceneImage } from './hooks/useSceneImage';
import { generateSyntheticScene } from './utils/syntheticScene';
import { sendQuery } from './api/client';

let idCounter = 0;
const nextId = () => `e${++idCounter}`;

export default function App() {
  const [mode, setMode] = useState('single');
  const [before, setBefore] = useState(null);
  const [after, setAfter] = useState(null);
  const [regions, setRegions] = useState([]);
  const [focusedRegionId, setFocusedRegionId] = useState(null);
  const [log, setLog] = useState([]);
  const [pending, setPending] = useState(false);
  const [view, setView] = useState(null);
  const [syncViews, setSyncViews] = useState(true);

  const { loadFromFile, fromSynthetic } = useSceneImage();

  const handleModeChange = (next) => {
    setMode(next);
    setRegions([]);
    setFocusedRegionId(null);
    if (next === 'single') setAfter(null);
  };

  const handleUpload = useCallback(
    async (which, file) => {
      const scene = await loadFromFile(file);
      if (which === 'before') setBefore(scene);
      else setAfter(scene);
      setRegions([]);
      setFocusedRegionId(null);
    },
    [loadFromFile]
  );

  const handleLoadDemo = useCallback(() => {
    const seed = Math.floor(Math.random() * 10000);
    const baseline = fromSynthetic(
      generateSyntheticScene({ seed, variant: 'baseline' }),
      'demo-baseline.png'
    );
    setBefore(baseline);
    if (mode === 'compare') {
      // No query-type tab to key off anymore — the model classifies the
      // question itself, so just pick a plausible "after" variant at random.
      const variant = Math.random() < 0.5 ? 'flood' : 'construction';
      const changed = fromSynthetic(
        generateSyntheticScene({ seed, variant }),
        `demo-${variant}.png`
      );
      setAfter(changed);
    }
    setRegions([]);
    setFocusedRegionId(null);
  }, [mode, fromSynthetic]);

  const readyToQuery = mode === 'single' ? !!before : !!before && !!after;
  const disabledReason =
    mode === 'compare' && (!before || !after)
      ? 'Load both a before and an after scene to compare.'
      : !before
      ? 'Load a scene to begin.'
      : null;

  const handleAsk = useCallback(
    async (question) => {
      setLog((l) => [...l, { id: nextId(), role: 'user', text: question }]);
      setPending(true);
      const pendingId = nextId();
      setLog((l) => [...l, { id: pendingId, role: 'pending' }]);

      try {
        // No queryType is sent — the model classifies count/locate/compare/
        // classify on its own from the question text.
        const result = await sendQuery({
          question,
          before,
          after: mode === 'compare' ? after : null,
        });
        setRegions(result.regions);
        setFocusedRegionId(result.regions[0]?.id || null);
        setLog((l) =>
          l.map((entry) =>
            entry.id === pendingId
              ? {
                  id: pendingId,
                  role: 'assistant',
                  text: result.answer,
                  confidence: result.confidence,
                  regions: result.regions,
                  queryType: result.queryType,
                }
              : entry
          )
        );
      } catch (err) {
        setLog((l) =>
          l.map((entry) =>
            entry.id === pendingId
              ? { id: pendingId, role: 'error', text: `Query failed: ${err.message}` }
              : entry
          )
        );
      } finally {
        setPending(false);
      }
    },
    [before, after, mode]
  );

  const currentScene = mode === 'compare' ? after || before : before;

  return (
    <div className="flex h-screen flex-col bg-graphite-950">
      <Header mode={mode} onModeChange={handleModeChange} />
      <div className="flex min-h-0 flex-1">
        <div className="flex min-w-0 flex-[1_1_65%] flex-col border-r border-graphite-800">
          <SceneLoader
            mode={mode}
            onUpload={handleUpload}
            onLoadDemo={handleLoadDemo}
            hasScene={!!before}
          />
          <div className="min-h-0 flex-1">
            {mode === 'single' ? (
              <ImageViewer
                image={before}
                regions={regions}
                focusedRegionId={focusedRegionId}
                onViewChange={setView}
              />
            ) : (
              <CompareViewer
                before={before}
                after={after}
                regions={regions}
                focusedRegionId={focusedRegionId}
                syncViews={syncViews}
                onStatus={setView}
              />
            )}
          </div>
          <StatusBar
            scene={currentScene}
            view={view}
            mode={mode}
            syncViews={syncViews}
            onToggleSync={() => setSyncViews((s) => !s)}
          />
        </div>

        <div className="w-[380px] shrink-0">
          <QueryPanel
            log={log}
            onSubmit={handleAsk}
            disabled={!readyToQuery || pending}
            disabledReason={pending ? null : disabledReason}
            onFocusRegion={setFocusedRegionId}
            focusedRegionId={focusedRegionId}
            mode={mode}
          />
        </div>
      </div>
    </div>
  );
}
