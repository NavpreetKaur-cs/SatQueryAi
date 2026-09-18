import { useCallback } from 'react';

/**
 * A "scene" is the shape every viewer/API function expects:
 *   { url, width, height, name, sensor, capturedOn }
 * `url` is ALWAYS a real base64 data URL (data:image/...;base64,...) here —
 * for both uploaded files and synthetic demo scenes — never a blob: URL.
 * That matters because client.js sends this straight to the backend as
 * `dataUrl`; a blob: URL is a local-browser-only reference and can't be
 * decoded by anything else, so it would silently fail server-side.
 */
export function useSceneImage() {
  const loadFromFile = useCallback((file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('Could not read that image file.'));
      reader.onload = () => {
        const dataUrl = reader.result;
        const img = new Image();
        img.onload = () => {
          resolve({
            url: dataUrl,
            width: img.naturalWidth,
            height: img.naturalHeight,
            name: file.name,
            sensor: 'Uploaded image (sensor unknown)',
            capturedOn: 'unknown',
          });
        };
        img.onerror = () => reject(new Error('Could not read that image file.'));
        img.src = dataUrl;
      };
      reader.readAsDataURL(file);
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
