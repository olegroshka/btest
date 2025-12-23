This folder holds local static assets for the Platform UI.

We keep it local-first: no CDN required.

- plotly.min.js is served by the platform API under /static/plotly.min.js
  Note: This file is no longer committed to the repo to save space. 
  The API automatically serves it from the installed 'plotly' Python package.

