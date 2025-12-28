import React from 'react';
import ReactDOM from 'react-dom/client';
import { PlatformApp } from './platform/PlatformApp';
import './ag-theme-quant.css';

// AG Grid v34+ requires explicit module registration.
import { ModuleRegistry, AllCommunityModule } from 'ag-grid-community';

// Register the full Community module bundle.
ModuleRegistry.registerModules([AllCommunityModule]);

function mount() {
  // Mount into the legacy-stable #app node in the server shell.
  const host = document.getElementById('app') ?? document.getElementById('root');
  if (!host) return;

  ReactDOM.createRoot(host).render(
    <React.StrictMode>
      <PlatformApp />
    </React.StrictMode>
  );
}

mount();
