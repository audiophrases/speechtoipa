const test = require('node:test');
const assert = require('node:assert');

const passages = require('./passages.js');
const {
  cleanExtract,
  buildPassage,
  isUnsuitableTitle,
  numberRatio,
  contentJunkRatio,
  normalizeLength,
  resolveSources,
  fetchPassage,
} = passages;

// passages.js decides once, at load time, whether it is running in a browser —
// that decision is what adds `origin=*` (and drops the headers a browser is not
// allowed to set), so testing it means loading a second copy with a window in
// place. The stand-in is deliberately minimal: the file only ever checks that
// `window` and `window.document` exist, and hangs its global off `window`.
function loadAsBrowser() {
  const path = require.resolve('./passages.js');
  const saved = global.window;
  global.window = { document: {} };
  delete require.cache[path];
  const browserCopy = require('./passages.js');
  delete require.cache[path];
  if (saved === undefined) delete global.window;
  else global.window = saved;
  return browserCopy;
}

// Captures the URLs a fetch-driven call would request, answering each with the
// same canned article.
function stubFetch(article, { fail = false } = {}) {
  const calls = [];
  const original = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    if (fail) throw new Error('network down');
    return {
      ok: true,
      json: async () => ({ query: { pages: { 42: article } } }),
    };
  };
  return {
    calls,
    restore() {
      globalThis.fetch = original;
    },
  };
}

const ARTICLE = {
  pageid: 42,
  title: 'Rain',
  fullurl: 'https://simple.wikipedia.org/wiki/Rain',
  revisions: [{ timestamp: '2026-01-02T03:04:05Z' }],
  extract:
    'Rain is water that falls from clouds. It is a kind of precipitation. ' +
    'Rain helps plants to grow and fills rivers and lakes.',
};

test('cleanExtract drops footnote markers and folds paragraphs into one line', () => {
  const raw = 'England[1] is a country.\n\nIt   is part of the UK .';

  assert.strictEqual(cleanExtract(raw), 'England is a country. It is part of the UK.');
});

test('buildPassage keeps a short passage inside its sentence and word caps', () => {
  const text = buildPassage(ARTICLE.extract, 'short');

  assert.ok(text.startsWith('Rain is water that falls from clouds.'));
  assert.ok(text.split(/(?<=[.!?])\s+/).length <= 3);
  assert.ok(text.split(/\s+/).length <= 40);
});

test('buildPassage rejects an extract with too few usable sentences', () => {
  assert.strictEqual(buildPassage('Rain is wet.', 'short'), '');
});

test('buildPassage rejects a number-dense stub', () => {
  const stub =
    'It was built in 1876 near 1880 and 1890. ' +
    'The town had 1200 people in 1901 and 1500 in 1911. ' +
    'It covers 45 km and sits 320 m up.';

  assert.strictEqual(buildPassage(stub, 'short'), '');
});

test('buildPassage rejects a proper-noun-dense stub, unless caps mean nothing', () => {
  const stub =
    'Kovacs played for Steaua Bucharest and Dinamo Kiev. ' +
    'He joined Marek Dupnitsa under Georgi Vasilev. ' +
    'Later he coached Lokomotiv Sofia with Petar Hubchev.';

  assert.strictEqual(buildPassage(stub, 'short'), '');
  assert.notStrictEqual(buildPassage(stub, 'short', { checkCaps: false }), '');
});

test('isUnsuitableTitle catches list and disambiguation pages in every source language', () => {
  assert.ok(isUnsuitableTitle('List of rivers in France'));
  assert.ok(isUnsuitableTitle('Mercury (disambiguation)'));
  assert.ok(isUnsuitableTitle('Llista de peixos'));
  assert.ok(!isUnsuitableTitle('Rain'));
});

test('numberRatio and contentJunkRatio ignore leading capitals at sentence starts', () => {
  assert.strictEqual(numberRatio('Rain fell in 1990.'), 0.25);
  assert.strictEqual(contentJunkRatio('Rain is wet.'), 0);
});

test('resolveSources accepts app codes and full locale codes, and rejects the rest', () => {
  assert.strictEqual(resolveSources('fr').sources[0].host, 'fr.vikidia.org');
  assert.strictEqual(resolveSources('fr-FR').sources[0].host, 'fr.vikidia.org');
  assert.strictEqual(resolveSources('en').sources[0].host, 'simple.wikipedia.org');
  // Arabic script has no letter case, so the proper-noun heuristic is off.
  assert.strictEqual(resolveSources('ma').checkCaps, false);
  assert.strictEqual(resolveSources('de'), null);
});

test('normalizeLength falls back to short for anything unexpected', () => {
  assert.strictEqual(normalizeLength('long'), 'long');
  assert.strictEqual(normalizeLength('enormous'), 'short');
  assert.strictEqual(normalizeLength(null), 'short');
  assert.strictEqual(normalizeLength('toString'), 'short');
});

test('fetchPassage returns the passage with the citation its source page needs', async () => {
  const fetchStub = stubFetch(ARTICLE);
  try {
    const result = await fetchPassage('en', 'short');

    assert.ok(result.text.startsWith('Rain is water'));
    assert.strictEqual(result.title, 'Rain');
    assert.strictEqual(result.url, 'https://simple.wikipedia.org/wiki/Rain');
    assert.strictEqual(result.date, '2026-01-02T03:04:05Z');
    assert.strictEqual(result.source, 'Simple English Wikipedia');
    assert.ok(fetchStub.calls[0].url.startsWith('https://simple.wikipedia.org/w/api.php?'));
  } finally {
    fetchStub.restore();
  }
});

test('fetchPassage rejects a language with no encyclopedia before touching the network', async () => {
  await assert.rejects(fetchPassage('de', 'short'), (err) => err.code === 'UNSUPPORTED_LANGUAGE');
});

test('fetchPassage reports an unreachable source rather than an empty result', async () => {
  const fetchStub = stubFetch(ARTICLE, { fail: true });
  try {
    await assert.rejects(
      fetchPassage('en', 'short', { attemptsPerSource: 1 }),
      (err) => err.code === 'UPSTREAM_UNAVAILABLE'
    );
  } finally {
    fetchStub.restore();
  }
});

test('in a browser the query asks MediaWiki for anonymous CORS and sets no headers', async () => {
  const browserCopy = loadAsBrowser();
  const fetchStub = stubFetch(ARTICLE);
  try {
    await browserCopy.fetchPassage('en', 'short');

    const { url, options } = fetchStub.calls[0];
    // Without origin=* the response carries no Access-Control-Allow-Origin and
    // the browser discards it — this parameter IS the GitHub Pages support.
    assert.ok(new URL(url).searchParams.get('origin') === '*');
    // A custom header would force a preflight; User-Agent is forbidden outright.
    assert.strictEqual(options.headers, undefined);
  } finally {
    fetchStub.restore();
  }
});

test('outside a browser the request identifies itself and stays same-origin-free', async () => {
  const fetchStub = stubFetch(ARTICLE);
  try {
    await fetchPassage('en', 'short');

    const { url, options } = fetchStub.calls[0];
    assert.strictEqual(new URL(url).searchParams.get('origin'), null);
    assert.match(options.headers['Api-User-Agent'], /^speechtoipa\//);
  } finally {
    fetchStub.restore();
  }
});
