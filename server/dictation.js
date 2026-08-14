// HTTP wrapper around the shared passage engine in ../passages.js.
//
// The engine itself now runs in the browser (app.js loads the same file), so
// this endpoint is no longer how "Fetch a passage" normally works — the page
// asks the wikis directly, which is what lets the GitHub Pages copy fetch
// passages with no server running at all. It stays because a browser that
// cannot reach the wikis (a school filter, say) may still reach this server,
// and because anything already pointed at /api/dictation keeps working.
//
// createRequire, not import: passages.js is a plain script the browser loads
// with a <script> tag, and this loads that exact file rather than a second copy
// of the logic.
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { fetchPassage, LENGTHS } = require('../passages.js');

function sendJson(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

/** Node http handler for GET /api/dictation?lang=<code>&length=short|medium|long */
export async function handleDictationRequest(req, res) {
  try {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const lang = (url.searchParams.get('lang') || 'en').toLowerCase();
    const requestedLength = url.searchParams.get('length');
    const length = Object.hasOwn(LENGTHS, requestedLength) ? requestedLength : 'short';

    const result = await fetchPassage(lang, length);
    if (!result) {
      sendJson(res, 502, { error: 'Could not find a suitable passage. Please try again.' });
      return;
    }

    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(result));
  } catch (error) {
    if (error.code === 'UNSUPPORTED_LANGUAGE') {
      sendJson(res, 400, { error: error.message });
      return;
    }
    console.error('Dictation fetch error:', error);
    if (error.code === 'UPSTREAM_UNAVAILABLE') {
      sendJson(res, 503, { error: error.message });
      return;
    }
    sendJson(res, 500, { error: 'Failed to fetch passage', details: error.message });
  }
}
