import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/static/',
  plugins: [react()],
  build: {
    // Write only assets into the committed folder; do NOT overwrite the server shell index.html.
    outDir: '../src/quantdsl_backtest/platform_ui/assets_dist',
    emptyOutDir: false,
    assetsDir: 'assets',
    sourcemap: true,
    rollupOptions: {
      // Do not emit index.html; backend serves its own committed shell.
      input: 'src/main.tsx',
      output: {
        entryFileNames: 'assets/main.js',
        chunkFileNames: 'assets/chunk-[name].js',
        assetFileNames: (assetInfo) => {
          const name = String(assetInfo.name || '');
          if (name.slice(-4) === '.css') return 'assets/main.css';
          return 'assets/[name][extname]';
        },
      },
    },
  },
});
