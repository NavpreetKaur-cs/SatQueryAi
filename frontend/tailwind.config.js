/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // Graphite base — instrument-panel background, not pure black
        graphite: {
          950: '#0D1113',
          900: '#141A1D',
          800: '#1E262A',
          700: '#2B353A',
          600: '#3D4A50',
        },
        fog: {
          300: '#8B98A0',
          200: '#AEB9BE',
        },
        paper: '#EDF1F2',
        // Primary accent — active states, links, focus rings
        signal: {
          DEFAULT: '#3ED9B0',
          dim: '#2A9C80',
        },
        // Secondary accent — detection/highlight boxes on imagery, alerts
        alert: {
          DEFAULT: '#FF5C5C',
          dim: '#C94848',
        },
        // Tertiary — "after" / construction / new-feature overlays
        amber: {
          DEFAULT: '#E8A23D',
        },
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        body: ['"IBM Plex Sans"', 'sans-serif'],
        data: ['"IBM Plex Mono"', 'monospace'],
      },
      borderRadius: {
        none: '0px',
        sm: '2px',
        DEFAULT: '3px',
      },
    },
  },
  plugins: [],
};
