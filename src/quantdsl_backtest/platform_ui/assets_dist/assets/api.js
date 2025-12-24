// Local-first API helpers (no framework) - shared across UI modules.

export async function fetchJson(url, opts) {
  const res = await fetch(url, {
    ...(opts || {}),
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      ...((opts && opts.headers) || {}),
    },
  });

  const data = await res.json().catch(() => ({ error: { message: 'non-json response' } }));
  if (!res.ok) {
    if (data && data.detail && !data.error) {
      throw { error: { code: `HTTP_${res.status}`, message: String(data.detail) } };
    }
    throw data;
  }
  return data;
}

export function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function renderJsonError(err) {
  try {
    return '<pre style="white-space:pre-wrap">' + escapeHtml(JSON.stringify(err, null, 2)) + '</pre>';
  } catch (e) {
    return '<pre style="white-space:pre-wrap">(error)</pre>';
  }
}
