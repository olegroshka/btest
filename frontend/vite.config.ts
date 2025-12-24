import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

// Build directly into the committed, local-first assets_dist folder.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: path.resolve(__dirname, '../src/quantdsl_backtest/platform_ui/assets_dist'),
    emptyOutDir: false,
    assetsDir: 'assets',
    sourcemap: true,
  },
});

