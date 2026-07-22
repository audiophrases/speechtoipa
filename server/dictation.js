// "Fetch a passage" backend — ported from the DictationApp project.
// Self-contained (no external deps): given a target language, it pulls a random
// article from a plain-language encyclopedia and shapes it into a dictation /
// reading passage, with source + date so the learner can find the original.
//
// Every source is a MediaWiki site, so one code path (the /w/api.php random +
// extracts query) serves them all. For each language we try a simpler, curated
// encyclopedia FIRST — Simple English Wikipedia or Vikidia (a children's
// encyclopedia) — because their articles are general-knowledge topics in plain
// language, far better practice material than random Wikipedia stubs (which
// skew to obscure villages/species: proper-noun lists). Wikipedia is the
// always-available fallback, and the only source where there's no kids' wiki
// (Moroccan Darija).

// --- sentence helpers (inlined so this module has no local imports) ---

function splitIntoSentences(text) {
  const sentences = [];
  for (const line of (text || '').split(/\n+/)) {
    const parts = line.match(/[^.!?]*[.!?]+["'”’»)\]]*\s*|[^.!?]+$/g) || [];
    for (const part of parts) {
      const trimmed = part.trim();
      if (trimmed) sentences.push(trimmed);
    }
  }
  return sentences;
}

function countWords(text) {
  const trimmed = (text || '').trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

// --- sources, keyed by the app's target language codes (en/fr/ca/it/es/ma) ---

const vikidia = (lang) => ({ host: `${lang}.vikidia.org`, label: 'Vikidia', edition: lang });
const wikipedia = (edition) => ({ host: `${edition}.wikipedia.org`, label: 'Wikipedia', edition });
const simpleWiki = { host: 'simple.wikipedia.org', label: 'Simple English Wikipedia', edition: 'simple' };

export const SOURCES = {
  en: [simpleWiki, wikipedia('en')],
  fr: [vikidia('fr'), wikipedia('fr')],
  ca: [vikidia('ca'), wikipedia('ca')],
  it: [vikidia('it'), wikipedia('it')],
  es: [vikidia('es'), wikipedia('es')],
  // 'ma' (app code for Moroccan Darija) -> 'ary', the Moroccan Arabic Wikipedia
  // edition: a small but real wiki, distinct from Modern Standard Arabic 'ar'.
  ma: [wikipedia('ary')],
};

// Languages where the "capitalized mid-sentence word = proper noun" heuristic
// misfires: Arabic script has no letter case (German too, if ever added).
const NO_CAPS_CHECK = new Set(['ma', 'de']);

const LENGTHS = {
  short: { maxSentences: 3, maxWords: 40, minWordsPerSentence: 3, maxWordsPerSentence: 16, minSentences: 2 },
  medium: { maxSentences: 5, maxWords: 65, minWordsPerSentence: 3, maxWordsPerSentence: 25, minSentences: 3 },
  long: { maxSentences: 11, maxWords: 140, minWordsPerSentence: 0, maxWordsPerSentence: Infinity, minSentences: 4 },
};

// A passage is rejected if this fraction of its tokens are proper-noun-ish or
// bare numbers — i.e. it reads like a stub of names/stats rather than practice
// prose. Numbers alone get a tighter, script-independent cap.
const MAX_JUNK_RATIO = 0.4;
const MAX_NUMBER_RATIO = 0.22;

const SKIP_TITLE_MARKERS = [
  'disambiguation', 'desambiguació', 'desambiguación', 'homonymie', 'disambigua',
  'list of ', 'llista de ', 'liste des ', 'liste de ', 'lista de ', 'lista di ', 'anexo:',
  '(значения)', 'список ',
  '(dezambiguizare)', 'listă de ', 'lista ',
  'توضيح',
];

const USER_AGENT = 'speechtoipa/1.0 (educational reading practice)';
const API_PATH = '/w/api.php';

export function isUnsuitableTitle(title) {
  const t = (title || '').toLowerCase();
  return SKIP_TITLE_MARKERS.some((m) => t.includes(m));
}

export function cleanExtract(raw) {
  return (raw || '')
    .replace(/\[[^\]]*\]/g, '')      // footnote markers [1] and IPA/editorial [ˈɪŋɡlənd]
    .replace(/\s*\n+\s*/g, ' ')      // paragraph breaks -> spaces
    .replace(/\s{2,}/g, ' ')         // collapse runs of whitespace
    .replace(/\s+([.,;:!?])/g, '$1') // no space before punctuation
    .trim();
}

/**
 * Fraction of tokens that look like proper nouns (capitalized mid-sentence, when
 * `checkCaps`) or bare numbers/years. High values mark name/stat-dense stubs.
 */
export function contentJunkRatio(text, checkCaps = true) {
  const tokens = (text || '').split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return 1;
  let junk = 0;
  let sentenceStart = true;
  for (const tok of tokens) {
    const word = tok.replace(/^[^\p{L}\p{N}]+/u, '').replace(/[^\p{L}\p{N}]+$/u, '');
    if (word) {
      if (/^\p{Nd}/u.test(word)) {
        junk++;
      } else if (checkCaps && !sentenceStart) {
        const f = word[0];
        if (f.toLocaleLowerCase() !== f.toLocaleUpperCase() && f === f.toLocaleUpperCase()) junk++;
      }
    }
    sentenceStart = /[.!?]$/.test(tok);
  }
  return junk / tokens.length;
}

/** Fraction of tokens that are bare numbers/years. Script-independent. */
export function numberRatio(text) {
  const tokens = (text || '').split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return 0;
  let n = 0;
  for (const tok of tokens) {
    const word = tok.replace(/^[^\p{L}\p{N}]+/u, '').replace(/[^\p{L}\p{N}]+$/u, '');
    if (word && /^\p{Nd}/u.test(word)) n++;
  }
  return n / tokens.length;
}

/**
 * Turn a raw article extract into a passage sized for the requested length.
 * Returns '' when it can't yield enough usable sentences, or when the result is
 * too dense with proper nouns/numbers to be good practice material.
 */
export function buildPassage(extract, length = 'short', { checkCaps = true } = {}) {
  const cfg = LENGTHS[length] || LENGTHS.short;
  const sentences = splitIntoSentences(cleanExtract(extract));

  const suitable = sentences.filter((s) => {
    const w = countWords(s);
    return w >= cfg.minWordsPerSentence && w <= cfg.maxWordsPerSentence;
  });
  const pool = suitable.length >= cfg.minSentences ? suitable : sentences;

  const picked = [];
  let words = 0;
  for (const s of pool) {
    if (picked.length >= cfg.maxSentences) break;
    const w = countWords(s);
    if (picked.length > 0 && words + w > cfg.maxWords) break;
    picked.push(s);
    words += w;
  }

  if (picked.length < cfg.minSentences) return '';
  const text = picked.join(' ');
  if (numberRatio(text) > MAX_NUMBER_RATIO) return '';
  if (contentJunkRatio(text, checkCaps) > MAX_JUNK_RATIO) return '';
  return text;
}

async function fetchRandomArticle(host, signal) {
  const params = new URLSearchParams({
    action: 'query',
    format: 'json',
    generator: 'random',
    grnnamespace: '0',
    grnlimit: '1',
    prop: 'extracts|info|revisions',
    explaintext: '1',
    exintro: '1',
    inprop: 'url',
    rvprop: 'timestamp',
    redirects: '1',
  });
  const url = `https://${host}${API_PATH}?${params.toString()}`;
  const resp = await fetch(url, { headers: { 'User-Agent': USER_AGENT, 'Api-User-Agent': USER_AGENT }, signal });
  if (!resp.ok) throw new Error(`${host} responded ${resp.status}`);
  const data = await resp.json();
  const pages = data?.query?.pages;
  if (!pages) return null;
  const page = Object.values(pages)[0];
  if (!page || page.missing !== undefined) return null;
  return {
    title: page.title,
    url: page.fullurl || page.canonicalurl || null,
    date: page.revisions?.[0]?.timestamp || null,
    extract: page.extract || '',
  };
}

/**
 * Try each source in order (simpler encyclopedia first, Wikipedia fallback),
 * fetching random articles and shaping them into a passage. Retries per source
 * to skip unusable (too short, list/disambiguation, or name/number-dense)
 * articles. Returns null if nothing suitable was found.
 */
export async function fetchDictation(sources, length, { checkCaps = true, attemptsPerSource = 5, timeoutMs = 6000, budgetMs = 12000 } = {}) {
  const deadline = Date.now() + budgetMs;
  let networkErrors = 0;
  let fetchedAny = false;
  for (const source of sources) {
    for (let i = 0; i < attemptsPerSource && Date.now() < deadline; i++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const article = await fetchRandomArticle(source.host, controller.signal);
        fetchedAny = true;
        if (article && article.title && !isUnsuitableTitle(article.title)) {
          const text = buildPassage(article.extract, length, { checkCaps });
          if (text) {
            return {
              text,
              title: article.title,
              url: article.url,
              date: article.date,
              source: source.label,
              edition: source.edition,
            };
          }
        }
      } catch {
        networkErrors++;
      } finally {
        clearTimeout(timer);
      }
    }
  }
  if (!fetchedAny && networkErrors > 0) {
    const err = new Error('Could not reach the source. Please try again in a moment.');
    err.code = 'UPSTREAM_UNAVAILABLE';
    throw err;
  }
  return null;
}

function sendJson(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

/** Node http handler for GET /api/dictation?lang=<code>&length=short|medium|long */
export async function handleDictationRequest(req, res) {
  try {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const rawLang = (url.searchParams.get('lang') || 'en').toLowerCase();
    const requestedLength = url.searchParams.get('length');
    const length = Object.hasOwn(LENGTHS, requestedLength) ? requestedLength : 'short';
    // Accept the app's short codes directly, and tolerate full codes (en-US).
    const langKey = SOURCES[rawLang] ? rawLang : rawLang.split('-')[0];
    const sources = SOURCES[langKey];

    if (!sources) {
      sendJson(res, 400, { error: `Unsupported language: ${rawLang}` });
      return;
    }

    const result = await fetchDictation(sources, length, { checkCaps: !NO_CAPS_CHECK.has(langKey) });
    if (!result) {
      sendJson(res, 502, { error: 'Could not find a suitable passage. Please try again.' });
      return;
    }

    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(result));
  } catch (error) {
    console.error('Dictation fetch error:', error);
    if (error.code === 'UPSTREAM_UNAVAILABLE') {
      sendJson(res, 503, { error: error.message });
      return;
    }
    sendJson(res, 500, { error: 'Failed to fetch passage', details: error.message });
  }
}
