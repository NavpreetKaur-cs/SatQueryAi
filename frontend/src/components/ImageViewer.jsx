import { useEffect, useRef } from 'react';
import L from 'leaflet';

// Convert an image-pixel coordinate (origin top-left, y down) into the
// lat/lng space Leaflet's CRS.Simple uses (origin bottom-left, y up).
function pixelToLatLng(px, py, imgHeight) {
  return L.latLng(imgHeight - py, px);
}

function bboxToBounds([x, y, w, h], imgHeight) {
  return L.latLngBounds(
    pixelToLatLng(x, y + h, imgHeight),
    pixelToLatLng(x + w, y, imgHeight)
  );
}

const REGION_COLORS = {
  before: '#E8A23D',
  after: '#FF5C5C',
  default: '#FF5C5C',
};

/**
 * @param {Object} props
 * @param {{url:string,width:number,height:number}} props.image
 * @param {Array} props.regions - regions belonging to THIS image only
 * @param {string|null} props.focusedRegionId
 * @param {(map: L.Map) => void} [props.onReady]
 * @param {(view: {center:[number,number], zoom:number}) => void} [props.onViewChange]
 * @param {string} [props.label] - small corner label, e.g. "BEFORE" / "AFTER"
 */
export default function ImageViewer({ image, regions = [], focusedRegionId, onReady, onViewChange, label }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const overlayLayerRef = useRef(null);
  const regionLayerRef = useRef(null);
  const suppressViewEvent = useRef(false);

  // init map once
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current, {
      crs: L.CRS.Simple,
      minZoom: -5,
      zoomControl: false,
      attributionControl: false,
    });
    mapRef.current = map;
    L.control.zoom({ position: 'bottomright' }).addTo(map);

    map.on('moveend zoomend', () => {
      if (suppressViewEvent.current) return;
      const c = map.getCenter();
      onViewChange?.({ center: [c.lat, c.lng], zoom: map.getZoom() });
    });

    onReady?.(map);

    return () => {
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // swap image
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !image) return;

    if (overlayLayerRef.current) {
      map.removeLayer(overlayLayerRef.current);
    }
    const bounds = L.latLngBounds([0, 0], [image.height, image.width]);
    overlayLayerRef.current = L.imageOverlay(image.url, bounds).addTo(map);
    map.setMaxBounds(bounds.pad(0.25));
    map.fitBounds(bounds, { animate: false });
  }, [image]);

  // draw regions
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !image) return;

    if (regionLayerRef.current) {
      map.removeLayer(regionLayerRef.current);
    }
    const group = L.layerGroup();
    regions.forEach((region) => {
      const isFocused = region.id === focusedRegionId;
      const color = REGION_COLORS[region.image] || REGION_COLORS.default;
      const rect = L.rectangle(bboxToBounds(region.bbox, image.height), {
        color,
        weight: isFocused ? 3 : 1.5,
        fillOpacity: isFocused ? 0.18 : 0.06,
        className: 'sq-region-box',
      });
      rect.bindTooltip(`${region.label}${region.score ? ` · ${Math.round(region.score * 100)}%` : ''}`, {
        direction: 'top',
        className: 'sq-tooltip',
        sticky: true,
      });
      rect.addTo(group);
    });
    group.addTo(map);
    regionLayerRef.current = group;
  }, [regions, focusedRegionId, image]);

  // pan/zoom to a focused region
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !image || !focusedRegionId) return;
    const region = regions.find((r) => r.id === focusedRegionId);
    if (!region) return;
    suppressViewEvent.current = true;
    map.flyToBounds(bboxToBounds(region.bbox, image.height), { padding: [40, 40], duration: 0.6 });
    setTimeout(() => {
      suppressViewEvent.current = false;
    }, 700);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusedRegionId]);

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="h-full w-full bg-graphite-950" />
      {label && (
        <div className="pointer-events-none absolute left-3 top-3 z-[500] rounded-sm border border-graphite-700 bg-graphite-900/85 px-2 py-1 font-data text-[11px] tracking-wide text-fog-200">
          {label}
        </div>
      )}
      {!image && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-fog-300">
          No image loaded
        </div>
      )}
    </div>
  );
}
