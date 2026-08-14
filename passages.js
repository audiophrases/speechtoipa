// "Fetch a passage" — given a target language, pull a random article from a
// plain-language encyclopedia and shape it into a reading passage, with source
// + date so the learner can find the original.
//
// This one file runs in two places. The browser app (app.js) calls it directly,
// which is what makes "Fetch a passage" work on GitHub Pages, where there is no
// server at all: every source is a MediaWiki site, and MediaWiki answers
// anonymous cross-origin requests when the query carries `origin=*`. The Node
// server (server/dictation.js) wraps the same functions as /api/dictation, kept
// as a fallback for a network that blocks the wikis but can reach the server.
//
// Every source being MediaWiki also means one code path (the /w/api.php random +
// extracts query) serves them all. For each language we try a simpler, curated
// encyclopedia FIRST — Simple English Wikipedia or Vikidia (a children's
// encyclopedia) — because their articles are general-knowledge topics in plain
// language, far better practice material than random Wikipedia stubs (which
// skew to obscure villages/species: proper-noun lists). Wikipedia is the
// always-available fallback, and the only source where there's no kids' wiki
// (Moroccan Darija).
//
// A plain script, not a module, because the rest of the app is: it attaches one
// global (window.Passages) and exports the same object to Node.
(function attachPassages(global) {
  'use strict';

  // The browser has no say over User-Agent (a forbidden header) and needs the
  // `origin=*` opt-in; Node sends a real UA, as Wikimedia's policy asks of
  // server-side callers, and has no origin to declare.
  const IS_BROWSER = typeof window !== 'undefined' && typeof window.document !== 'undefined';

  // --- sentence helpers (inlined so this file has no imports of its own) ---

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

  const SOURCES = {
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

  function isUnsuitableTitle(title) {
    const t = (title || '').toLowerCase();
    return SKIP_TITLE_MARKERS.some((m) => t.includes(m));
  }

  function cleanExtract(raw) {
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
  function contentJunkRatio(text, checkCaps = true) {
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
  function numberRatio(text) {
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
  function buildPassage(extract, length = 'short', { checkCaps = true } = {}) {
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

  /** The API URL for one random article, in the form this environment may ask for it. */
  function randomArticleUrl(host) {
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
    // MediaWiki only sends `Access-Control-Allow-Origin` when the query asks for
    // it. '*' means "answer anonymously to anyone", which is all a public
    // encyclopedia read needs — and it keeps the request "simple", so the
    // browser sends it straight out with no preflight round trip.
    if (IS_BROWSER) params.set('origin', '*');
    return `https://${host}${API_PATH}?${params.toString()}`;
  }

  async function fetchRandomArticle(host, signal) {
    const options = { signal };
    // Setting these in a browser would be pointless (User-Agent is forbidden to
    // scripts) and costly (a custom header turns this into a preflighted request).
    if (!IS_BROWSER) options.headers = { 'User-Agent': USER_AGENT, 'Api-User-Agent': USER_AGENT };
    const resp = await fetch(randomArticleUrl(host), options);
    if (!resp.ok) throw new Error(`${host} responded ${resp.status}`);
    const data = await resp.json();
    const pages = data && data.query && data.query.pages;
    if (!pages) return null;
    const page = Object.values(pages)[0];
    if (!page || page.missing !== undefined) return null;
    return {
      title: page.title,
      url: page.fullurl || page.canonicalurl || null,
      date: (page.revisions && page.revisions[0] && page.revisions[0].timestamp) || null,
      extract: page.extract || '',
    };
  }

  /**
   * Try each source in order (simpler encyclopedia first, Wikipedia fallback),
   * fetching random articles and shaping them into a passage. Retries per source
   * to skip unusable (too short, list/disambiguation, or name/number-dense)
   * articles. Returns null if nothing suitable was found.
   */
  async function fetchDictation(sources, length, { checkCaps = true, attemptsPerSource = 5, timeoutMs = 6000, budgetMs = 12000 } = {}) {
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

  /** One of 'short' | 'medium' | 'long'; anything else reads as 'short'. */
  function normalizeLength(length) {
    return Object.prototype.hasOwnProperty.call(LENGTHS, length) ? length : 'short';
  }

  /**
   * Sources for an app language code. Accepts the app's short codes ('fr') and
   * tolerates full ones ('fr-FR'). Null when the language has no encyclopedia.
   */
  function resolveSources(lang) {
    const raw = String(lang || 'en').toLowerCase();
    const langKey = SOURCES[raw] ? raw : raw.split('-')[0];
    const sources = SOURCES[langKey];
    if (!sources) return null;
    return { langKey, sources, checkCaps: !NO_CAPS_CHECK.has(langKey) };
  }

  /**
   * The whole feature in one call: language + length in, passage (or null when
   * nothing suitable turned up) out. Throws with `code` set for the two failures
   * a caller may want to word differently: UNSUPPORTED_LANGUAGE and
   * UPSTREAM_UNAVAILABLE (nothing answered — offline, or the wikis blocked).
   */
  async function fetchPassage(lang, length, options = {}) {
    const resolved = resolveSources(lang);
    if (!resolved) {
      const err = new Error(`Unsupported language: ${lang}`);
      err.code = 'UNSUPPORTED_LANGUAGE';
      throw err;
    }
    return fetchDictation(resolved.sources, normalizeLength(length), {
      checkCaps: resolved.checkCaps,
      ...options,
    });
  }

  const Passages = {
    SOURCES,
    LENGTHS,
    isUnsuitableTitle,
    cleanExtract,
    contentJunkRatio,
    numberRatio,
    buildPassage,
    fetchDictation,
    fetchPassage,
    normalizeLength,
    resolveSources,
  };

  global.Passages = Passages;
  if (typeof module !== 'undefined' && module.exports) module.exports = Passages;
})(typeof window !== 'undefined' ? window : globalThis);
