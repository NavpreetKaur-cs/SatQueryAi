import { useCallback } from 'react';

/**
 * A "scene" is the shape every viewer/API function expects:
 *   { url, width, height, name, sensor, capturedOn }
 * Both an uploaded file and a synthetic demo scene are normalized to this
 * so the rest of the app never needs to know which one it's looking at.
 */
export function useSceneImage() {
  const loadFromFile = useCallback((file) => {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        resolve({
          url,
          width: img.naturalWidth,
          height: img.naturalHeight,
          name: file.name,
          sensor: 'Uploaded image (sensor unknown)',
          capturedOn: 'unknown',
        });
      };
      img.onerror = () => reject(new Error('Could not read that image file.'));
      img.src = url;
    });
  }, []);

  const fromSynthetic = useCallback((generated, name) => {
    return {
      url: generated.dataUrl,
      width: generated.width,
      height: generated.height,
      name,
      sensor: generated.sensor,
      capturedOn: generated.capturedOn,
    };
  }, []);

  return { loadFromFile, fromSynthetic };
}
