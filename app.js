const TARGET_LANGS = [
  { code: 'en', label: 'English' },
  { code: 'fr', label: 'French' },
  { code: 'ca', label: 'Catalan' },
  { code: 'ma', label: 'Moroccan Darija' },
  { code: 'it', label: 'Italian' }
];

const BASE_LANGS = [
  { code: 'ca', label: 'Catalan' },
  { code: 'en', label: 'English' },
  { code: 'fr', label: 'French' },
  { code: 'es', label: 'Spanish' },
  { code: 'ma', label: 'Moroccan Darija' },
  { code: 'it', label: 'Italian' }
];

const STORAGE_KEY = 'speechReadingProgress';
const DEFAULT_LESSON_SUFFIX = 'a1_introductions';
let availableLessons = [];
const CUSTOM_LESSON_ID = 'custom';
const DEFAULT_TTS_BASE_URL = 'https://translate.googleapis.com';
const DEFAULT_APPROX_THRESHOLD = 0.65;
const DEFAULT_MATCH_THRESHOLD = 0.7;
const PROPER_NOUN_THRESHOLD_FLOOR = 0.55;
const CEFR_LEVELS = [50, 60, 70, 80, 90, 100];
const DEFAULT_CEFR_INDEX = 0;

const MASTER_CSV_URLS = {
  ca: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=1216373156&single=true&output=csv',
  en: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=1053057720&single=true&output=csv',
  fr: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=484976070&single=true&output=csv',
  it: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=1338439854&single=true&output=csv',
  ma: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=710375040&single=true&output=csv',
};

const MASTER_ROWS_BY_LANG = {};
const TRANSLATION_LANG_CODES = ['ca', 'es', 'en', 'fr', 'it', 'ma'];
const DARJA_TRANSCRIPTION_HEADER = 'ma_latn';
const DARJA_TRANSCRIPTION_FALLBACK_HEADERS = [
  'ma_transcription',
  'ma_translit',
  'ma_latin',
  'transcription',
];

const NO_TTS_SUPPORT_MESSAGE =
  'No local text-to-speech voice found. Using fallback TTS service for playback.';

const TTS_CACHE_PREFIX = 'speechtoipa-tts:';
const TTS_LOCAL_CACHE_CHAR_LIMIT = 180;
const ttsAudioCache = new Map();
const SILENT_AUDIO_BASE64 = 'UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA=';
const CURATED_VOICE_PRIORITIES = {
  ar: ['ar-MA', 'ar-SA', 'ar-EG', 'ar'],
  ca: ['ca-ES', 'es-ES'],
  en: ['en-US', 'en-GB', 'en-AU'],
  fr: ['fr-CA', 'fr-FR', 'fr-BE'],
  it: ['it-IT'],
};

const SpeechRecognition =
  typeof window !== 'undefined'
    ? window.SpeechRecognition || window.webkitSpeechRecognition
    : null;

function getTtsBaseUrl() {
  if (typeof window === 'undefined') return DEFAULT_TTS_BASE_URL;
  if (window.TTS_BASE_URL) return window.TTS_BASE_URL.replace(/\/$/, '');
  if (window.__TTS_BASE_URL__) return window.__TTS_BASE_URL__.replace(/\/$/, '');

  if (typeof document !== 'undefined') {
    const meta = document.querySelector('meta[name="tts-base-url"]');
    if (meta && meta.content) {
      return meta.content.trim().replace(/\/$/, '');
    }
  }

  return DEFAULT_TTS_BASE_URL;
}

function isGoogleTranslateTts(baseUrl) {
  return /translate\.googleapis\.com/i.test(baseUrl || '');
}

// A dedicated TTS server (e.g. the neural server in server/) beats browser
// voices, which vary wildly across browsers and platforms. The Google
// Translate endpoint is only a last-resort fallback, so it never wins.
function shouldPreferTtsService() {
  const baseUrl = getTtsBaseUrl();
  return Boolean(baseUrl) && !isGoogleTranslateTts(baseUrl);
}

function buildTtsRequestUrl(baseUrl, text, langCode) {
  const sanitizedBase = (baseUrl || '').replace(/\/$/, '');
  if (isGoogleTranslateTts(sanitizedBase)) {
    const googleLang = (langCode || '').split('-')[0] || langCode || 'en';
    const params = new URLSearchParams({
      ie: 'UTF-8',
      q: text,
      tl: googleLang,
      client: 'gtx',
    });
    return `${sanitizedBase}/translate_tts?${params.toString()}`;
  }

  return `${sanitizedBase}/tts?text=${encodeURIComponent(text)}&lang=${encodeURIComponent(langCode)}`;
}

function getTtsCacheKey(text, lang) {
  // Include the source host so switching TTS services never replays audio
  // that was cached from a different (lower-quality) service.
  let host = 'default';
  try {
    host = new URL(getTtsBaseUrl()).host || 'default';
  } catch {
    /* keep default */
  }
  return `${TTS_CACHE_PREFIX}${host}:${lang}:${text}`;
}

function isMobileDevice() {
  if (typeof navigator === 'undefined') return false;
  return /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent || '');
}

function isSpeechSynthesisSupported() {
  if (typeof window === 'undefined') return false;
  return Boolean(window.speechSynthesis) && 'SpeechSynthesisUtterance' in window;
}

if (typeof window !== 'undefined' && window.speechSynthesis) {
  window.speechSynthesis.onvoiceschanged = () => {
    window.speechSynthesis.getVoices();
    buildVoiceMap();
    warnIfArabicVoiceMissing();
  };
}

let wordSpans = [];
let targetTokens = [];
let targetTokenVariants = [];
let lastTranscript = '';
let currentSentenceText = '';
let wordStatus = [];
// Interim results sometimes match a word further down the sentence before the
// learner has reached it, making it flash green and then revert. Out-of-order
// matches (ones with unpronounced words before them) must persist for this
// long before they turn green; sequential matches stay instant.
const OUT_OF_ORDER_CONFIRM_MS = 900;
const outOfOrderCorrectSince = new Map();
let wordTooltipEl;
let sentenceTooltipEl;
let sentenceTooltipTimer = null;
let currentTooltipTarget = null;
let hasWarnedAboutArabicVoice = false;
let lastLessonId = '';
let recognitionRestartTimer = null;
let nextSentenceTimer = null;
let sentenceNavToken = 0;
let lastCoachAt = 0;
let pendingCoachTimer = null;
let pendingCoachIndex = -1;
const coachedWordIndices = new Set();
// Wait for this much silence after a stumble before coaching, so we never
// talk over a learner who is still reading. Any new recognition activity
// (including interim results) cancels the pending coach.
const COACH_SILENCE_MS = 3300;
const COACH_COOLDOWN_MS = 8000;
const state = {
  targetLang: 'fr',
  baseLang: 'en',
  lessonId: '',
  sentences: [],
  currentIndex: 0,
  mode: 'lesson',
  customSentence: '',
  savedLessonState: null,
  recognition: null,
  recording: false,
  supportsSpeechSynthesis: isSpeechSynthesisSupported(),
  supportsTtsService: Boolean(getTtsBaseUrl()),
  supportsRecognition: Boolean(SpeechRecognition),
  manualStopRequested: false,
  sentenceComplete: false,
  pendingAutoAdvance: false,
  shouldAutoRestartRecognition: false,
  micPausedForTts: false,
  ttsLoading: false,
  approxLevelIndex: DEFAULT_CEFR_INDEX,
  audioUnlocked: false,
  ttsAudioElement: null,
};

const els = {};
const readyVoiceKeys = new Set();

function getVoiceKey(voice) {
  if (!voice) return '';
  return `${voice.lang}|${voice.name}`;
}

function markVoicesReady(voices) {
  if (!Array.isArray(voices) || !voices.length) return;
  voices.forEach((voice) => {
    const key = getVoiceKey(voice);
    if (key) readyVoiceKeys.add(key);
  });
}

function getVoiceNaturalness(voice) {
  if (!voice) return 0;
  const name = String(voice.name || '').toLowerCase();
  // Edge exposes its neural voices as e.g. "Microsoft Aria Online (Natural)".
  if (name.includes('natural') || name.includes('neural')) return 3;
  // iOS/macOS premium and enhanced voices.
  if (name.includes('premium') || name.includes('enhanced')) return 2;
  // Chrome's remote Google voices sound much better than local SAPI/eSpeak ones.
  if (name.includes('google') && !voice.localService) return 2;
  if (name.includes('siri')) return 1;
  return 0;
}

function rankVoicesForLang(voices, langCode) {
  if (!Array.isArray(voices)) return [];
  const normalizedLang = (langCode || '').toLowerCase();
  const baseLang = normalizedLang.split('-')[0];
  const curatedList = CURATED_VOICE_PRIORITIES[baseLang] || [];

  return voices.slice().sort((a, b) => {
    const aLang = (a.lang || '').toLowerCase();
    const bLang = (b.lang || '').toLowerCase();
    const aExact = Boolean(normalizedLang) && aLang === normalizedLang;
    const bExact = Boolean(normalizedLang) && bLang === normalizedLang;
    if (aExact !== bExact) return aExact ? -1 : 1;

    const aPrefix = Boolean(baseLang) && aLang.startsWith(baseLang);
    const bPrefix = Boolean(baseLang) && bLang.startsWith(baseLang);
    if (aPrefix !== bPrefix) return aPrefix ? -1 : 1;

    const aQuality = getVoiceNaturalness(a);
    const bQuality = getVoiceNaturalness(b);
    if (aQuality !== bQuality) return bQuality - aQuality;

    const aLocal = Boolean(a.localService);
    const bLocal = Boolean(b.localService);
    if (aLocal !== bLocal) return aLocal ? -1 : 1;

    const aReady = readyVoiceKeys.has(getVoiceKey(a));
    const bReady = readyVoiceKeys.has(getVoiceKey(b));
    if (aReady !== bReady) return aReady ? -1 : 1;

    const aCuratedIndex = curatedList.findIndex((lang) => aLang.startsWith(lang.toLowerCase()));
    const bCuratedIndex = curatedList.findIndex((lang) => bLang.startsWith(lang.toLowerCase()));
    const aCuratedScore = aCuratedIndex === -1 ? Number.MAX_SAFE_INTEGER : aCuratedIndex;
    const bCuratedScore = bCuratedIndex === -1 ? Number.MAX_SAFE_INTEGER : bCuratedIndex;
    if (aCuratedScore !== bCuratedScore) return aCuratedScore - bCuratedScore;

    if (aLang !== bLang) return aLang.localeCompare(bLang);
    return (a.name || '').localeCompare(b.name || '');
  });
}

function setTtsLoading(isLoading) {
  state.ttsLoading = isLoading;
  updateSpeechSynthesisState();
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const res = reader.result || '';
      const base64 = res.toString().split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

function base64ToBlob(data, mimeType) {
  let byteChars;
  if (typeof atob === 'function') {
    byteChars = atob(data);
  } else if (typeof Buffer !== 'undefined') {
    byteChars = Buffer.from(data, 'base64').toString('binary');
  } else {
    throw new Error('No base64 decoder available');
  }
  const byteNumbers = new Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i++) {
    byteNumbers[i] = byteChars.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: mimeType || 'audio/mpeg' });
}

function loadCachedTts(cacheKey) {
  if (ttsAudioCache.has(cacheKey)) {
    return ttsAudioCache.get(cacheKey);
  }

  try {
    if (typeof localStorage !== 'undefined') {
      const stored = localStorage.getItem(cacheKey);
      if (stored) {
        const [mimeType, data] = stored.split('|');
        const blob = base64ToBlob(data, mimeType);
        ttsAudioCache.set(cacheKey, blob);
        return blob;
      }
    }
  } catch (err) {
    console.warn('Unable to read cached TTS audio', err);
  }

  return null;
}

async function cacheTts(cacheKey, blob, originalText) {
  ttsAudioCache.set(cacheKey, blob);
  if (!originalText || originalText.length > TTS_LOCAL_CACHE_CHAR_LIMIT) return;

  try {
    if (typeof localStorage !== 'undefined') {
      const base64 = await blobToBase64(blob);
      const mimeType = blob.type || 'audio/mpeg';
      localStorage.setItem(cacheKey, `${mimeType}|${base64}`);
    }
  } catch (err) {
    console.warn('Unable to persist TTS audio to localStorage', err);
  }
}

async function fetchTtsAudio(text, langCode) {
  const baseUrl = getTtsBaseUrl();
  if (!baseUrl) {
    throw new Error('TTS service not configured');
  }

  const cacheKey = getTtsCacheKey(text, langCode);
  const cached = loadCachedTts(cacheKey);
  if (cached) return cached;

  setTtsLoading(true);
  setStatus('Fetching audio from TTS service…');
  try {
    const url = buildTtsRequestUrl(baseUrl, text, langCode);
    try {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`TTS service returned ${res.status}`);
      }
      const blob = await res.blob();
      await cacheTts(cacheKey, blob, text);
      return blob;
    } catch (err) {
      if (isGoogleTranslateTts(baseUrl)) {
        const audio = getPlaybackAudioElement();
        if (audio) {
          audio.src = url;
          return { directUrl: url };
        }
      }
      throw err;
    }
  } finally {
    setTtsLoading(false);
  }
}

async function playTtsBlob(source, rate = 1.0) {
  if (!source) return;
  if (!state.audioUnlocked) {
    setStatus('Tap Play again to enable audio.');
    await unlockAudioPlayback();
    if (!state.audioUnlocked) return;
  }
  const audio = getPlaybackAudioElement();
  if (!audio) return;
  let url = '';
  let shouldRevoke = false;
  if (source instanceof Blob) {
    url = URL.createObjectURL(source);
    shouldRevoke = true;
  } else if (typeof source === 'string') {
    url = source;
  } else if (source && source.directUrl) {
    url = source.directUrl;
  }
  if (!url) return;
  audio.src = url;
  audio.playbackRate = rate;

  const playbackFinished = new Promise((resolve) => {
    const finish = () => {
      if (shouldRevoke) {
        URL.revokeObjectURL(url);
      }
      resolve();
    };
    audio.onended = finish;
    audio.onerror = finish;
  });

  try {
    await audio.play();
    setStatus('Playing audio from TTS service.');
  } catch (err) {
    console.error('Failed to play fetched audio', err);
    const recommendation = buildBrowserRecommendation();
    setStatus(recommendation ? `Could not play fetched audio. ${recommendation}` : 'Could not play fetched audio.');
  }

  await playbackFinished;
}

function getPlaybackAudioElement() {
  if (state.ttsAudioElement) return state.ttsAudioElement;
  if (typeof Audio === 'undefined') return null;
  const audio = new Audio();
  audio.preload = 'auto';
  state.ttsAudioElement = audio;
  return audio;
}

async function unlockAudioPlayback() {
  if (state.audioUnlocked) return true;
  const audio = getPlaybackAudioElement();
  if (!audio) return false;

  let url = '';
  try {
    const blob = base64ToBlob(SILENT_AUDIO_BASE64, 'audio/wav');
    url = URL.createObjectURL(blob);
    audio.src = url;
    audio.muted = true;
    await audio.play();
    audio.pause();
    audio.currentTime = 0;
    audio.muted = false;
    state.audioUnlocked = true;
    updatePlaybackWarnings();
    return true;
  } catch (err) {
    console.warn('Unable to unlock audio playback', err);
    return false;
  } finally {
    if (url) URL.revokeObjectURL(url);
  }
}

function parseCourseCsv(text) {
  const rows = [];
  let current = '';
  let inQuotes = false;
  let row = [];

  const pushCell = () => {
    row.push(current);
    current = '';
  };
  const pushRow = () => {
    if (row.length) {
      rows.push(row.map((cell) => cell.replace(/^"|"$/g, '')));
      row = [];
    }
  };

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      pushCell();
    } else if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') {
        i++;
      }
      pushCell();
      pushRow();
    } else {
      current += char;
    }
  }

  if (current.length || row.length) {
    pushCell();
    pushRow();
  }

  if (!rows.length) return [];

  const headers = rows[0].map((h) => h.trim());
  return rows
    .slice(1)
    .filter((cells) => cells.some((c) => c && c.trim().length))
    .map((cells) => {
      const obj = {};
      headers.forEach((h, idx) => {
        obj[h] = (cells[idx] || '').trim();
      });
      if (obj.pronunciation_aliases) {
        obj.pronunciation_aliases = obj.pronunciation_aliases
          .split('|')
          .map((alias) => alias.trim())
          .filter(Boolean);
      }
      return obj;
    });
}

function getDarijaTranscription(row) {
  if (!row) return '';
  if (row[DARJA_TRANSCRIPTION_HEADER]) return row[DARJA_TRANSCRIPTION_HEADER];
  for (const header of DARJA_TRANSCRIPTION_FALLBACK_HEADERS) {
    if (row[header]) return row[header];
  }
  return '';
}

function parseBooleanLikeValue(value) {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  if (typeof value !== 'string') return false;
  const normalized = value.trim().toLowerCase();
  if (!normalized) return false;
  return ['1', 'true', 'yes', 'y'].includes(normalized);
}

function inferProperNounFromTokenRow(row) {
  if (!row || typeof row !== 'object') return false;
  const explicitFlag =
    parseBooleanLikeValue(row.is_proper_noun) ||
    parseBooleanLikeValue(row.proper_noun) ||
    parseBooleanLikeValue(row.isProperNoun);
  if (explicitFlag) return true;

  const pos = String(row.pos || row.part_of_speech || row.token_type || '')
    .trim()
    .toLowerCase();
  if (pos === 'propn' || pos === 'proper_noun' || pos === 'proper noun') {
    return true;
  }

  return false;
}

async function ensureMasterRowsForLang(lang) {
  if (MASTER_ROWS_BY_LANG[lang]) return MASTER_ROWS_BY_LANG[lang];

  const url = MASTER_CSV_URLS[lang];
  if (!url) return null;

  const res = await fetch(url);
  if (!res.ok) {
    console.error('Failed to fetch master CSV for', lang, res.status);
    return null;
  }
  const text = await res.text();
  const rows = parseCourseCsv(text);
  MASTER_ROWS_BY_LANG[lang] = rows;

  console.log('Loaded master rows for', lang, 'count =', rows.length);
  return rows;
}

const LOCAL_TTS_SERVER_URL = 'http://127.0.0.1:8787';

// If the neural TTS server (server/) is running locally, use it automatically.
// Only possible from http:/file: pages — https pages cannot call a local http
// server, so production deployments set the meta tts-base-url tag instead.
async function detectLocalTtsServer() {
  if (typeof window === 'undefined' || typeof fetch !== 'function') return;
  if (window.TTS_BASE_URL || window.__TTS_BASE_URL__) return;
  const proto = window.location?.protocol;
  if (proto !== 'http:' && proto !== 'file:') return;

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1500);
    const res = await fetch(`${LOCAL_TTS_SERVER_URL}/health`, { signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) return;
    const data = await res.json();
    if (!data || !data.ok) return;

    window.TTS_BASE_URL = LOCAL_TTS_SERVER_URL;
    state.supportsTtsService = true;
    updateSpeechSynthesisState();
    console.log('Neural TTS server detected at', LOCAL_TTS_SERVER_URL);
  } catch {
    /* no local server running — keep defaults */
  }
}

if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', async () => {
    cacheElements();
    createTooltips();
    detectLocalTtsServer();
    updateSpeechSynthesisState();
    hydrateSelections();
    attachEventListeners();
    hydrateFromStorage();
    buildVoiceMap();
    await loadLessonManifest();
    updateLessonId();
    await loadLesson();
    warnIfArabicVoiceMissing();
  });
}

function cacheElements() {
  els.targetSelect = document.getElementById('target-lang');
  els.baseSelect = document.getElementById('base-lang');
  els.lessonSelect = document.getElementById('lesson-select');
  els.approxSlider = document.getElementById('approx-slider');
  els.approxLabel = document.getElementById('approx-label');
  els.sentence = document.getElementById('sentence');
  els.play = document.getElementById('play-btn');
  els.slow = document.getElementById('slow-btn');
  els.record = document.getElementById('record-btn');
  els.stop = document.getElementById('stop-btn');
  els.next = document.getElementById('next-btn');
  els.playbackWarnings = document.getElementById('playback-warnings');
  els.status = document.getElementById('status');
  // removed: tts base url debug element (too noisy for UI)
  els.ttsBaseUrlDebug = null;
  els.transcript = document.getElementById('transcript');
  els.feedback = document.getElementById('feedback');
  els.customInput = document.getElementById('custom-sentence');
  els.customSubmit = document.getElementById('custom-submit');
  els.customReset = document.getElementById('custom-reset');
  els.customModal = document.getElementById('custom-modal');
  els.customDismissButtons = Array.from(document.querySelectorAll('[data-close-modal]'));

  // UI debug removed: don't display TTS base URL.
}

function createTooltips() {
  wordTooltipEl = document.createElement('div');
  wordTooltipEl.className = 'tooltip tooltip-word';
  wordTooltipEl.style.position = 'fixed';
  wordTooltipEl.style.pointerEvents = 'none';
  wordTooltipEl.style.visibility = 'hidden';

  sentenceTooltipEl = document.createElement('div');
  sentenceTooltipEl.className = 'tooltip tooltip-sentence';
  sentenceTooltipEl.style.position = 'fixed';
  sentenceTooltipEl.style.pointerEvents = 'none';
  sentenceTooltipEl.style.visibility = 'hidden';

  document.body.appendChild(wordTooltipEl);
  document.body.appendChild(sentenceTooltipEl);
}

function hydrateSelections() {
  populateSelect(els.targetSelect, TARGET_LANGS, state.targetLang);
  populateSelect(els.baseSelect, BASE_LANGS, state.baseLang);
}

function populateSelect(select, options, selected) {
  select.innerHTML = '';
  options.forEach((opt) => {
    const option = document.createElement('option');
    option.value = opt.code;
    option.textContent = `${opt.label} (${opt.code})`;
    if (opt.code === selected) option.selected = true;
    select.appendChild(option);
  });
}

function clampCefrIndex(index) {
  const max = CEFR_LEVELS.length - 1;
  if (!Number.isFinite(index)) return DEFAULT_CEFR_INDEX;
  return Math.min(max, Math.max(0, Math.round(index)));
}

function getApproxThresholdFromIndex(index) {
  const level = CEFR_LEVELS[clampCefrIndex(index)] || CEFR_LEVELS[DEFAULT_CEFR_INDEX];
  return level / 100;
}

function updateApproxLabel() {
  if (!els.approxLabel) return;
  const level = CEFR_LEVELS[clampCefrIndex(state.approxLevelIndex)] || CEFR_LEVELS[DEFAULT_CEFR_INDEX];
  els.approxLabel.textContent = `Accuracy ${level}%`;
}

function readStoredProgressData() {
  const raw = localStorage.getItem(STORAGE_KEY);
  let data = { progress: {} };
  try {
    data = raw ? JSON.parse(raw) : { progress: {} };
  } catch {
    data = { progress: {} };
  }
  if (!data.progress) data.progress = {};
  return data;
}

function writeStoredProgressData(data) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

function persistApproxSettings() {
  try {
    const data = readStoredProgressData();
    data.approxLevelIndex = state.approxLevelIndex;
    writeStoredProgressData(data);
  } catch (err) {
    console.warn('Failed to save approximation settings', err);
  }
}

function setApproxLevel(index, { persist = true } = {}) {
  state.approxLevelIndex = clampCefrIndex(index);
  if (els.approxSlider) {
    els.approxSlider.value = String(state.approxLevelIndex);
  }
  updateApproxLabel();
  if (persist) {
    persistApproxSettings();
  }
}

function getLessonsForLang(lang) {
  const lessons = availableLessons.filter((lesson) => lesson.lang === lang);
  if (lessons.length) return lessons;
  return [
    {
      id: `${lang}_${DEFAULT_LESSON_SUFFIX}`,
      lang,
      label: 'A1 introductions',
    },
  ];
}

function getCustomLessonOption(lang) {
  return {
    id: CUSTOM_LESSON_ID,
    lang,
    label: 'Custom text',
  };
}

function getLessonOptions() {
  const lessons = getLessonsForLang(state.targetLang);
  return [getCustomLessonOption(state.targetLang), ...lessons];
}

function getDefaultLessonId() {
  const lessons = getLessonsForLang(state.targetLang);
  return lessons[0]?.id || `${state.targetLang}_${DEFAULT_LESSON_SUFFIX}`;
}

function populateLessonSelect() {
  const options = getLessonOptions();
  els.lessonSelect.innerHTML = '';
  options.forEach((lesson) => {
    const option = document.createElement('option');
    option.value = lesson.id;
    option.textContent = `${lesson.label} (${lesson.lang})`;
    els.lessonSelect.appendChild(option);
  });

  const selectionIsCustomWithoutInput =
    state.lessonId === CUSTOM_LESSON_ID && !state.customSentence;
  const hasSelection =
    !selectionIsCustomWithoutInput && options.some((lesson) => lesson.id === state.lessonId);
  const fallbackLessonId = getDefaultLessonId();
  const nextLessonId = selectionIsCustomWithoutInput
    ? fallbackLessonId
    : hasSelection
      ? state.lessonId
      : fallbackLessonId;

  state.lessonId = nextLessonId;
  els.lessonSelect.value = state.lessonId;

  if (state.lessonId !== CUSTOM_LESSON_ID) {
    lastLessonId = state.lessonId;
  } else if (!lastLessonId) {
    lastLessonId = fallbackLessonId;
  }
}

async function loadLessonManifest() {
  const lang = state.targetLang;

  const rows = await ensureMasterRowsForLang(lang);
  if (!rows || !rows.length) {
    console.warn('No master sheet rows found for language', lang);
    availableLessons = [];
    populateLessonSelect();
    return;
  }

  const lessonMap = new Map();
  rows.forEach((row) => {
    if (!row.lesson_id) return;
    if (!lessonMap.has(row.lesson_id)) {
      const title = row.lesson_title || row.lesson_id;
      lessonMap.set(row.lesson_id, {
        id: row.lesson_id,
        lang,
        label: title,
        theme: title,
      });
    }
  });

  availableLessons = Array.from(lessonMap.values());
  populateLessonSelect();
}

function attachEventListeners() {
  els.targetSelect.addEventListener('change', async () => {
    state.targetLang = els.targetSelect.value;
    await loadLessonManifest();
    updateLessonId();
    saveProgress();
    loadLesson();
    warnIfArabicVoiceMissing();
    updatePlaybackWarnings();
  });

  els.baseSelect.addEventListener('change', () => {
    state.baseLang = els.baseSelect.value;
    saveProgress();
    renderCurrentSentence();
  });

  els.lessonSelect.addEventListener('change', () => {
    handleLessonSelection(els.lessonSelect.value);
  });

  els.approxSlider?.addEventListener('input', (event) => {
    const nextIndex = Number(event.target.value);
    setApproxLevel(nextIndex);
  });

  els.play.addEventListener('click', () => handlePlaybackClick(1));
  els.slow.addEventListener('click', () => handlePlaybackClick(0.7));
  els.next.addEventListener('click', () => goToNext());
  els.record.addEventListener('click', startRecording);
  els.stop.addEventListener('click', stopRecording);

  els.sentence.addEventListener('click', (e) => {
    if (e.target.classList.contains('word')) {
      speakWord(e.target.dataset.word);
      e.target.classList.toggle('active');
    }
  });

  els.sentence.addEventListener('mousemove', onSentenceMouseMove);
  els.sentence.addEventListener('mouseleave', hideTooltips);

  els.customSubmit.addEventListener('click', (e) => {
    e.preventDefault();
    const text = (els.customInput.value || '').trim();
    if (!text) {
      setStatus('Please enter a sentence to practice.');
      return;
    }
    state.customSentence = text;
    enterCustomMode(text);
    closeCustomModal();
  });

  els.customReset.addEventListener('click', (e) => {
    e.preventDefault();
    closeCustomModal(true);
    if (!state.customSentence) {
      els.customInput.value = '';
    }
  });

  (els.customDismissButtons || []).forEach((btn) => {
    btn.addEventListener('click', () => closeCustomModal(true));
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && els.customModal && !els.customModal.classList.contains('hidden')) {
      closeCustomModal(true);
      return;
    }

    if (shouldIgnoreHotkey(event)) return;

    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      goToPrevious();
      return;
    }

    if (event.key === 'ArrowRight') {
      event.preventDefault();
      goToNext();
      return;
    }

    if (event.key === ' ' || event.code === 'Space') {
      event.preventDefault();
      togglePlayback();
      return;
    }

    if (event.key === 'Enter' || event.code === 'NumpadEnter') {
      event.preventDefault();
      toggleRecording();
      return;
    }

    if (event.key === 'p' || event.key === 'P') {
      event.preventDefault();
      handlePlaybackClick(1);
    }
  });

}

function handleLessonSelection(selection) {
  if (selection === CUSTOM_LESSON_ID) {
    openCustomModal();
    els.lessonSelect.value = state.lessonId || lastLessonId || getDefaultLessonId();
    return;
  }

  if (state.mode === 'custom') {
    state.mode = 'lesson';
    state.savedLessonState = null;
  }

  state.lessonId = selection;
  lastLessonId = selection;
  state.currentIndex = 0;
  saveProgress();
  loadLesson();
}

function openCustomModal() {
  if (!els.customModal) return;
  els.customModal.classList.remove('hidden');
  els.customModal.setAttribute('aria-hidden', 'false');
  if (els.customInput) {
    els.customInput.value = state.customSentence || '';
    els.customInput.focus();
  }
}

function closeCustomModal(resetSelection = false) {
  if (!els.customModal) return;
  els.customModal.classList.add('hidden');
  els.customModal.setAttribute('aria-hidden', 'true');

  if (resetSelection) {
    const fallbackLessonId = lastLessonId || getDefaultLessonId();
    els.lessonSelect.value = fallbackLessonId;
    if (state.mode !== 'custom') {
      state.lessonId = fallbackLessonId;
    }
  }
}

function hydrateFromStorage() {
  const raw = localStorage.getItem(STORAGE_KEY);
  let saved = null;
  try {
    if (raw) {
      saved = normalizeLegacyCodes(JSON.parse(raw));
    }
  } catch (err) {
    console.error('Failed to parse saved progress', err);
  }

  if (saved) {
    if (saved.targetLang) state.targetLang = saved.targetLang;
    if (saved.baseLang) state.baseLang = saved.baseLang;
    if (saved.lessonId && saved.lessonId !== CUSTOM_LESSON_ID) {
      state.lessonId = saved.lessonId;
    }
    if (saved.progress && saved.progress[state.lessonId]) {
      state.currentIndex = saved.progress[state.lessonId].currentIndex || 0;
    }
    populateSelect(els.targetSelect, TARGET_LANGS, state.targetLang);
    populateSelect(els.baseSelect, BASE_LANGS, state.baseLang);
    populateLessonSelect();
  }

  if (saved && Number.isFinite(saved.approxLevelIndex)) {
    state.approxLevelIndex = clampCefrIndex(saved.approxLevelIndex);
  } else {
    state.approxLevelIndex = DEFAULT_CEFR_INDEX;
  }
  setApproxLevel(state.approxLevelIndex, { persist: false });
}

function normalizeLegacyCodes(saved) {
  if (saved.targetLang === 'ary') saved.targetLang = 'ma';
  if (saved.baseLang === 'ary') saved.baseLang = 'ma';
  if (saved.lessonId?.startsWith('ary_')) {
    saved.lessonId = saved.lessonId.replace(/^ary_/, 'ma_');
  }

  if (saved.progress) {
    const legacyLessonKey = `ary_${DEFAULT_LESSON_SUFFIX}`;
    const updatedLessonKey = `ma_${DEFAULT_LESSON_SUFFIX}`;
    if (saved.progress[legacyLessonKey] && !saved.progress[updatedLessonKey]) {
      saved.progress[updatedLessonKey] = saved.progress[legacyLessonKey];
    }
  }

  return saved;
}

function saveProgress(bestScore) {
  if (state.mode === 'custom') return;
  const data = readStoredProgressData();
  data.targetLang = state.targetLang;
  data.baseLang = state.baseLang;
  data.lessonId = state.lessonId;
  data.approxLevelIndex = state.approxLevelIndex;
  data.progress = data.progress || {};
  data.progress[state.lessonId] = data.progress[state.lessonId] || { currentIndex: 0, scores: {} };
  data.progress[state.lessonId].currentIndex = state.currentIndex;
  if (bestScore !== undefined) {
    const previous = data.progress[state.lessonId].scores?.[currentSentence().id] || 0;
    if (!data.progress[state.lessonId].scores) data.progress[state.lessonId].scores = {};
    data.progress[state.lessonId].scores[currentSentence().id] = Math.max(previous, bestScore);
  }
  writeStoredProgressData(data);
}

function updateLessonId() {
  state.lessonId = els.lessonSelect.value;
  initRecognition();
}

async function loadLesson() {
  state.mode = 'lesson';
  state.savedLessonState = null;
  const lang = state.targetLang;
  const lessonId = state.lessonId;
  if (!lessonId) return;

  if (lessonId === CUSTOM_LESSON_ID) {
    if (state.mode === 'custom') return;

    const fallbackLessonId = lastLessonId || getDefaultLessonId();
    state.lessonId = fallbackLessonId;
    els.lessonSelect.value = fallbackLessonId;
    return;
  }

  lastLessonId = lessonId;

  const rows = await ensureMasterRowsForLang(lang);
  if (!rows || !rows.length) {
    setStatus('No data available for this language.');
    state.sentences = [];
    els.sentence.textContent = '';
    return;
  }

  const lessonRows = rows.filter((r) => r.lesson_id === lessonId);
  if (!lessonRows.length) {
    setStatus('No sentences for this lesson.');
    state.sentences = [];
    els.sentence.textContent = '';
    return;
  }

  // Group by sentence_id and preserve sentence order by first occurrence
  const bySentence = {};
  const sentenceOrder = [];

  lessonRows.forEach((row) => {
    const sid = row.sentence_id;
    if (!sid) return;
    if (!bySentence[sid]) {
      bySentence[sid] = [];
      sentenceOrder.push(sid);
    }
    bySentence[sid].push(row);
  });

  const l2Col = lang;

  const sentences = sentenceOrder.map((sid, index) => {
    const group = bySentence[sid];
    const sentenceRow = group.find((r) => !r.token_id); // token_id empty = sentence row
    const tokenRows = group.filter((r) => r.token_id);

    // 1) Sentence text (L2)
    const text = (sentenceRow && sentenceRow[l2Col]) || '';

    // 2) Sentence-level translations
    const sentenceTranslations = {};
    ['ca', 'es', 'en', 'fr', 'it', 'ma'].forEach((code) => {
      const val = sentenceRow && sentenceRow[code];
      if (val) sentenceTranslations[code] = val;
    });

    const sentenceTranscription = getDarijaTranscription(sentenceRow);
    const sentenceTranscriptions = sentenceTranscription
      ? { [DARJA_TRANSCRIPTION_HEADER]: sentenceTranscription }
      : {};

    // 3) Tokens (if any token rows exist)
    const tokens = tokenRows
      .slice()
      .sort((a, b) => a.token_id.localeCompare(b.token_id))
      .map((r) => {
        const tokenTranslations = {};
        ['ca', 'es', 'en', 'fr', 'it', 'ma'].forEach((code) => {
          const val = r[code];
          if (val) tokenTranslations[code] = val;
        });
        const surface = r[l2Col] || '';
        const tokenTranscription = getDarijaTranscription(r);
        const pronunciationAliases = Array.isArray(r.pronunciation_aliases)
          ? r.pronunciation_aliases
          : String(r.pronunciation_aliases || '')
              .split('|')
              .map((alias) => alias.trim())
              .filter(Boolean);
        return {
          surface,
          translations: tokenTranslations,
          transcription: tokenTranscription,
          pronunciation_aliases: pronunciationAliases,
          isProperNoun: inferProperNounFromTokenRow(r),
        };
      });

    return {
      id: sid,
      unit: null, // we can keep null for now; order is controlled by sentenceOrder
      theme: sentenceRow?.lesson_title || lessonId,
      title: sentenceRow?.lesson_title || lessonId,
      sentenceNumber: index + 1,
      text,
      translations: sentenceTranslations,
      transcriptions: sentenceTranscriptions,
      tokens,
    };
  });

  state.sentences = sentences;
  const saved = loadProgressForLesson();
  state.currentIndex = saved?.currentIndex || 0;

  renderCurrentSentence();
  const lessonMeta = availableLessons.find((l) => l.id === lessonId) || {};
  updateSpeechSynthesisState();
  const ttsNotice = state.supportsSpeechSynthesis ? '' : ` • ${NO_TTS_SUPPORT_MESSAGE}`;
  setStatus(
    `Loaded ${lessonMeta.lang?.toUpperCase() || ''} • ${lessonMeta.label || lessonId}${ttsNotice}`
  );
  playbackQueue.warmVoicesForLang(getLangCode(state.targetLang));
}

function loadProgressForLesson() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    const data = JSON.parse(raw);
    return data.progress?.[state.lessonId] || null;
  } catch {
    return null;
  }
}

function buildVoiceMap() {
  if (!window.speechSynthesis) return;

  const voices = window.speechSynthesis.getVoices();
  markVoicesReady(voices);
  updatePlaybackWarnings();
}

function onSentenceMouseMove(e) {
  const target = e.target;
  if (!target.classList.contains('word')) {
    hideTooltips();
    return;
  }

  if (currentTooltipTarget !== target) {
    currentTooltipTarget = target;
    showWordTooltip(target, e.clientX, e.clientY);
    scheduleSentenceTooltip(target, e.clientX, e.clientY);
  } else {
    positionTooltips(e.clientX, e.clientY);
  }
}

function hideTooltips() {
  currentTooltipTarget = null;
  if (wordTooltipEl) {
    wordTooltipEl.style.visibility = 'hidden';
  }
  if (sentenceTooltipEl) {
    sentenceTooltipEl.style.visibility = 'hidden';
  }
  if (sentenceTooltipTimer) {
    clearTimeout(sentenceTooltipTimer);
    sentenceTooltipTimer = null;
  }
}

function showWordTooltip(span, x, y) {
  const wordTrans = span.dataset.wordTranslation || span.dataset.sentenceTranslation;
  const wordTranscription = span.dataset.wordTranscription || span.dataset.sentenceTranscription;
  const tooltipContent = buildTooltipContent(wordTrans, wordTranscription);
  if (!tooltipContent) {
    wordTooltipEl.style.visibility = 'hidden';
    return;
  }
  wordTooltipEl.textContent = tooltipContent;
  wordTooltipEl.style.visibility = 'visible';
  positionTooltips(x, y);
}

function buildTooltipContent(translation, transcription) {
  if (state.targetLang === 'ma' && transcription) {
    if (translation && translation !== transcription) {
      return `${translation}\n${transcription}`;
    }
    return translation || transcription;
  }
  return translation || '';
}

function scheduleSentenceTooltip(span, x, y) {
  if (sentenceTooltipEl) {
    sentenceTooltipEl.style.visibility = 'hidden';
  }
  if (sentenceTooltipTimer) {
    clearTimeout(sentenceTooltipTimer);
  }

  const sentenceTrans = span.dataset.sentenceTranslation;
  const wordTrans = span.dataset.wordTranslation;
  const sentenceTranscription = span.dataset.sentenceTranscription;

  if (!sentenceTrans || !wordTrans || sentenceTrans === wordTrans) {
    return;
  }

  sentenceTooltipTimer = setTimeout(() => {
    if (currentTooltipTarget !== span) return;

    sentenceTooltipEl.textContent = buildTooltipContent(sentenceTrans, sentenceTranscription);
    sentenceTooltipEl.style.visibility = 'visible';
    positionTooltips(x, y);
  }, 800);
}

function positionTooltips(x, y) {
  if (wordTooltipEl) {
    wordTooltipEl.style.left = x + 12 + 'px';
    wordTooltipEl.style.top = y + 8 + 'px';
  }

  if (sentenceTooltipEl && wordTooltipEl) {
    const wordRect = wordTooltipEl.getBoundingClientRect();
    sentenceTooltipEl.style.left = wordRect.left + 'px';
    sentenceTooltipEl.style.top = wordRect.bottom + 4 + 'px';
  }
}

function renderCurrentSentence() {
  if (!state.sentences.length) return;
  const sentence = currentSentence();
  const sentenceEl = els.sentence;

  sentenceNavToken += 1;
  state.sentenceComplete = false;
  coachedWordIndices.clear();
  clearPendingCoach();
  hideTooltips();
  sentenceEl.classList.remove('sentence-complete');
  sentenceEl.innerHTML = '';
  wordSpans = [];

  // RTL handling for Moroccan Darija
  const isRTL = state.targetLang === 'ma';
  if (isRTL) {
    sentenceEl.dir = 'rtl';
  } else {
    sentenceEl.dir = 'ltr';
  }
  sentenceEl.classList.toggle('rtl-sentence', isRTL);

  const fullText = sentence.text || '';
  currentSentenceText = fullText;

  const hasTokens = Array.isArray(sentence.tokens) && sentence.tokens.length > 0;

  if (hasTokens) {
    const tokensForMatching = sentence.tokens.map((tokenObj) => ({
      surface: tokenObj.surface || '',
      translations: tokenObj.translations || {},
      transcription: tokenObj.transcription || '',
      pronunciation_aliases: tokenObj.pronunciation_aliases || [],
      isProperNoun: Boolean(tokenObj.isProperNoun),
    }));

    // Scoring tokens
    // For Darija (MA), use ma_latn per-token transcriptions as the matching backbone.
    if (state.targetLang === 'ma') {
      const latnTokens = tokensForMatching
        .map((t) => t.transcription)
        .map((t) => normalizeToken(t, 'ma'))
        .filter(Boolean);
      targetTokens = latnTokens.length ? latnTokens : tokenizeText(fullText, state.targetLang);
    } else {
      targetTokens = tokenizeText(fullText, state.targetLang);
    }
    targetTokenVariants = targetTokens.map((token, index) => ({
      text: token,
      aliases: tokensForMatching[index]?.pronunciation_aliases || [],
      isProperNoun: Boolean(tokensForMatching[index]?.isProperNoun),
    }));

    let pos = 0;

    tokensForMatching.forEach((tokenObj) => {
      const word = tokenObj.surface;
      if (!word) return;

      const index = fullText.indexOf(word, pos);
      if (index === -1) return;

      const gap = fullText.slice(pos, index);
      if (gap) {
        sentenceEl.appendChild(document.createTextNode(gap));
      }

      const span = document.createElement('span');
      span.textContent = word;
      span.classList.add('word', 'word-pending');
      span.dataset.word = word;

      const wordTrans = tokenObj.translations?.[state.baseLang] || null;
      const sentenceTrans = sentence.translations?.[state.baseLang] || null;
      const wordTranscription = tokenObj.transcription || null;
      const sentenceTranscription = sentence.transcriptions?.[DARJA_TRANSCRIPTION_HEADER] || null;

      if (wordTrans) {
        span.dataset.wordTranslation = wordTrans;
      }
      if (sentenceTrans) {
        span.dataset.sentenceTranslation = sentenceTrans;
      }
      if (isRTL && wordTranscription) {
        span.dataset.wordTranscription = wordTranscription;
      }
      if (isRTL && sentenceTranscription) {
        span.dataset.sentenceTranscription = sentenceTranscription;
      }

      if (wordTrans) {
        span.setAttribute('aria-label', wordTrans);
      } else if (sentenceTrans) {
        span.setAttribute('aria-label', sentenceTrans);
      }

      wordSpans.push(span);
      sentenceEl.appendChild(span);

      if (isRTL) {
        sentenceEl.dir = 'rtl';
        span.dir = 'rtl';
      }

      pos = index + word.length;
    });

    const tail = fullText.slice(pos);
    if (tail) {
      sentenceEl.appendChild(document.createTextNode(tail));
    }
  } else {
    const rawTokens = fullText.split(/\s+/).filter(Boolean);
    targetTokens = rawTokens.map((w) => normalizeToken(w, state.targetLang));
    targetTokenVariants = targetTokens.map((token) => ({ text: token, aliases: [] }));

    rawTokens.forEach((word) => {
      const span = document.createElement('span');
      span.textContent = word + ' ';
      span.classList.add('word', 'word-pending');
      span.dataset.word = word;

      const sentenceTrans = sentence.translations?.[state.baseLang] || null;
      const sentenceTranscription = sentence.transcriptions?.[DARJA_TRANSCRIPTION_HEADER] || null;
      if (sentenceTrans) {
        span.dataset.sentenceTranslation = sentenceTrans;
        span.setAttribute('aria-label', sentenceTrans);
      }
      if (isRTL && sentenceTranscription) {
        span.dataset.wordTranscription = sentenceTranscription;
        span.dataset.sentenceTranscription = sentenceTranscription;
      }

      wordSpans.push(span);
      sentenceEl.appendChild(span);

      if (isRTL) {
        span.dir = 'rtl';
      }
    });
  }

  if (!targetTokenVariants.length) {
    targetTokenVariants = targetTokens.map((token) => ({ text: token, aliases: [] }));
  }

  resetSentenceState();
  els.feedback.textContent = '';
  els.transcript.textContent = '';

  const total = state.sentences.length;
  els.status.textContent =
    state.mode === 'custom' && total <= 1
      ? 'Custom sentence practice'
      : `Sentence ${state.currentIndex + 1} / ${total}`;
}

function currentSentence() {
  return state.sentences[state.currentIndex];
}

function goToPrevious() {
  if (!state.sentences.length) return;
  const total = state.sentences.length;
  state.currentIndex = (state.currentIndex - 1 + total) % total;
  renderCurrentSentence();
  saveProgress();
}

function goToNext() {
  if (!state.sentences.length) return;
  state.currentIndex = (state.currentIndex + 1) % state.sentences.length;
  renderCurrentSentence();
  saveProgress();
}

// Sentence-ending punctuation, including Arabic '؟' and ellipsis '…', for
// Darija custom text. Splits each line into one or more sentences so pasted
// paragraphs (or one-sentence-per-line lists) both work.
const SENTENCE_SPLIT_RE = /[^.!?؟…]+(?:[.!?؟…]+)?/g;

function splitIntoSentences(text) {
  const lines = String(text || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const sentences = [];
  lines.forEach((line) => {
    const normalized = line.replace(/\s+/g, ' ').trim();
    if (!normalized) return;
    const pieces = normalized.match(SENTENCE_SPLIT_RE) || [normalized];
    pieces.forEach((piece) => {
      const trimmed = piece.trim();
      if (trimmed) sentences.push(trimmed);
    });
  });

  if (sentences.length) return sentences;
  const fallback = String(text || '').trim();
  return fallback ? [fallback] : [];
}

function enterCustomMode(text) {
  if (!text) return;

  if (state.mode !== 'custom') {
    state.savedLessonState = {
      sentences: state.sentences.slice(),
      currentIndex: state.currentIndex,
      lessonId: state.lessonId,
    };
  }

  const sentenceTexts = splitIntoSentences(text);

  state.customSentence = text;
  state.mode = 'custom';
  state.lessonId = CUSTOM_LESSON_ID;
  state.currentIndex = 0;
  state.sentences = sentenceTexts.map((sentenceText, index) => ({
    id: `custom_${index + 1}`,
    unit: null,
    theme: 'Custom practice',
    title: 'Custom practice',
    sentenceNumber: index + 1,
    text: sentenceText,
    translations: { [state.baseLang]: sentenceText },
    tokens: [],
  }));

  els.lessonSelect.value = CUSTOM_LESSON_ID;
  closeCustomModal();
  renderCurrentSentence();
}

function exitCustomMode() {
  if (state.mode !== 'custom') return;

  state.mode = 'lesson';
  const saved = state.savedLessonState;
  state.savedLessonState = null;

  if (saved && saved.sentences?.length) {
    state.lessonId = saved.lessonId || state.lessonId;
    state.sentences = saved.sentences;
    state.currentIndex = saved.currentIndex || 0;
    lastLessonId = state.lessonId;
    populateLessonSelect();
    els.lessonSelect.value = state.lessonId;
    renderCurrentSentence();
    setStatus('Back to lesson mode.');
    return;
  }

  loadLesson();
}

function getLangCode(l2) {
  switch (l2) {
    case 'fr':
      return 'fr-FR';
    case 'en':
      return 'en-US';
    case 'ca':
      return 'ca-ES';
    case 'es':
      return 'es-ES';
    case 'it':
      return 'it-IT';
    case 'ma':
      return 'ar-SA'; // Fallback to standard Arabic so Chrome can speak it
    case 'ary':
      return 'ar-MA';
    default:
      return 'en-US';
  }
}

// Speech recognition and speech synthesis are independent browser capabilities:
// there's usually no Darija (ar-MA) TTS voice, so speech uses the ar-SA fallback
// above, but Chrome's recognizer supports ar-MA directly and it matches Darija
// pronunciation/vocabulary far better than the Modern Standard Arabic model.
function getRecognitionLangCode(l2) {
  if (l2 === 'ma') return 'ar-MA';
  return getLangCode(l2);
}

function ensureArabicVoiceAvailable() {
  if (!window.speechSynthesis) return false;
  const voices = speechSynthesis.getVoices() || [];
  return voices.some((v) => v.lang && v.lang.toLowerCase().startsWith('ar'));
}

function warnIfArabicVoiceMissing() {
  // Disabled: too noisy, and browser support varies widely on mobile.
  return;
}

function getVoiceForLang(langCode) {
  if (!window.speechSynthesis) return null;
  const voices = window.speechSynthesis.getVoices();
  if (!voices || !voices.length) return null;
  markVoicesReady(voices);

  const ranked = rankVoicesForLang(voices, langCode);
  return ranked[0] || null;
}

const playbackQueue = createPlaybackQueue();

function createPlaybackQueue() {
  const MAX_RETRIES = 2;
  const warmups = new Map();
  const isMobile = isMobileDevice();
  let isWarming = false;
  let processing = false;
  let queue = [];

  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function waitForVoicesReady() {
    const synth = window.speechSynthesis;
    if (!synth) return;

    const voices = synth.getVoices();
    if (voices && voices.length) return;

    await new Promise((resolve) => {
      const timer = setTimeout(() => {
        synth.removeEventListener('voiceschanged', handleVoicesChanged);
        resolve();
      }, 1200);

      function handleVoicesChanged() {
        clearTimeout(timer);
        synth.removeEventListener('voiceschanged', handleVoicesChanged);
        resolve();
      }

      synth.addEventListener('voiceschanged', handleVoicesChanged);
      synth.getVoices();
    });
  }

  async function warmVoicesForLang(langCode) {
    if (!isSpeechSynthesisSupported()) return null;
    if (warmups.has(langCode)) return warmups.get(langCode);

    const promise = (async () => {
      isWarming = true;
      try {
        await waitForVoicesReady();
        return getVoiceForLang(langCode);
      } finally {
        isWarming = false;
      }
    })();

    warmups.set(langCode, promise);
    return promise;
  }

  function resetWarmup(langCode) {
    if (!langCode) {
      warmups.clear();
      return;
    }
    warmups.delete(langCode);
  }

  async function waitForSynthAvailability() {
    const synth = window.speechSynthesis;
    if (!synth) return;

    if (!isMobile) {
      synth.cancel();
      return;
    }

    while (synth.speaking || synth.pending) {
      await wait(150);
    }
  }

  async function speakWithVoice(item) {
    const synth = window.speechSynthesis;
    if (!synth) return { success: false, retry: false };

    const voice = await warmVoicesForLang(item.langCode);
    if (!voice) {
      return { success: false, retry: item.attempt < MAX_RETRIES };
    }

    await waitForSynthAvailability();

    return new Promise((resolve) => {
      const utterance = new SpeechSynthesisUtterance(item.text);
      utterance.lang = item.langCode;
      utterance.rate = item.rate;
      utterance.voice = voice;

      utterance.onend = () => resolve({ success: true, retry: false });
      utterance.onerror = (event) => {
        const retryable = event.error === 'interrupted' || event.error === 'canceled';
        resolve({ success: false, retry: retryable });
      };

      if (synth.paused) synth.resume();
      synth.speak(utterance);
    });
  }

  async function speakWithService(item) {
    try {
      const blob = await fetchTtsAudio(item.text, item.langCode);
      await playTtsBlob(blob, item.rate);
      return true;
    } catch (err) {
      console.error('TTS service request failed', err);
      setStatus('TTS service unavailable. Check configuration and try again.');
      return false;
    }
  }

  async function handleItem(item) {
    const synthSupported = isSpeechSynthesisSupported();
    const ttsAvailable = Boolean(getTtsBaseUrl());
    const preferService = ttsAvailable && shouldPreferTtsService();
    let success = false;

    state.supportsTtsService = ttsAvailable;

    if (preferService) {
      success = await speakWithService(item);
    }

    if (!success && synthSupported) {
      const voiceResult = await speakWithVoice(item);
      success = voiceResult.success;

      if (!success && voiceResult.retry && item.attempt < MAX_RETRIES) {
        queue.unshift({ ...item, attempt: item.attempt + 1 });
        await wait(200);
        return;
      }
    }

    if (!success && ttsAvailable && !preferService) {
      success = await speakWithService(item);
    }

    if (!success && synthSupported && !ttsAvailable) {
      state.supportsSpeechSynthesis = false;
      updateSpeechSynthesisState({ announce: true });
    }

    if (typeof item.onDone === 'function') {
      try {
        item.onDone(success);
      } catch (err) {
        console.warn('TTS onDone callback failed', err);
      }
    }
  }

  async function processQueue() {
    if (processing || !queue.length) return;
    processing = true;

    while (queue.length) {
      const item = queue.shift();
      await handleItem(item);
    }

    processing = false;
  }

  return {
    enqueue(item) {
      queue.push({ ...item, attempt: item.attempt || 0 });
      processQueue();
    },
    warmVoicesForLang,
    resetWarmup,
    isWarming() {
      return isWarming;
    },
  };
}

async function speakSentence(text, langCode, rate = 1.0, { onDone } = {}) {
  if (!text) {
    setStatus('Nothing to play for this sentence.');
    if (typeof onDone === 'function') onDone(false);
    return;
  }

  if (isMobileDevice() && !state.audioUnlocked) {
    await unlockAudioPlayback();
  }

  await playbackQueue.warmVoicesForLang(langCode);
  state.supportsTtsService = Boolean(getTtsBaseUrl());
  playbackQueue.enqueue({ text, langCode, rate, onDone });
  updatePlaybackWarnings();
}

function speakCurrent(rate = 1, onDone) {
  if (!state.sentences.length) {
    if (typeof onDone === 'function') onDone(false);
    return;
  }
  const text = currentSentence().text;
  speakSentence(text, getLangCode(state.targetLang), rate, { onDone });
}

function speakWord(text, rate = 1) {
  if (!text) return;
  const wasPaused = pauseRecognitionForTts();
  speakSentence(text, getLangCode(state.targetLang), rate, {
    onDone: wasPaused ? () => resumeRecognitionAfterTts() : undefined,
  });
}

const FEEDBACK_PHRASES = {
  en: { tryWord: 'Try saying', praise: ['Well done!', 'Great reading!', 'Nice job!', 'Well read!'] },
  fr: { tryWord: 'Essaie de dire', praise: ['Bravo !', 'Très bien lu !', 'Super !'] },
  ca: { tryWord: 'Prova de dir', praise: ['Molt bé!', 'Ben llegit!', 'Fantàstic!'] },
  es: { tryWord: 'Intenta decir', praise: ['¡Muy bien!', '¡Bien leído!', '¡Genial!'] },
  it: { tryWord: 'Prova a dire', praise: ['Bravissimo!', 'Ben letto!', 'Ottimo!'] },
  ma: { tryWord: 'حاول تقول', praise: ['مزيان بزاف', 'برافو عليك'] },
};

function getFeedbackPhrases() {
  return FEEDBACK_PHRASES[state.baseLang] || FEEDBACK_PHRASES.en;
}

// Stop recognition while our own TTS speaks so the mic doesn't transcribe it.
function pauseRecognitionForTts() {
  if (!state.recognition || state.sentenceComplete) return false;
  if (!state.recording && !state.shouldAutoRestartRecognition) return false;
  clearPendingCoach();
  state.micPausedForTts = true;
  state.shouldAutoRestartRecognition = false;
  clearRecognitionRestartTimer();
  try {
    state.recognition.stop();
  } catch (err) {
    /* already stopped */
  }
  updateRecordState();
  return true;
}

function resumeRecognitionAfterTts() {
  if (!state.micPausedForTts) return;
  state.micPausedForTts = false;
  if (state.manualStopRequested || state.sentenceComplete) return;
  state.shouldAutoRestartRecognition = true;
  restartRecognitionSession();
}

// Speak a sequence of feedback items (each { text, langCode, rate }) with the
// mic paused, then resume listening and invoke onAllDone.
function speakFeedbackItems(items, onAllDone) {
  const list = (items || []).filter((item) => item && item.text);
  if (!list.length) {
    if (typeof onAllDone === 'function') onAllDone();
    return;
  }

  const wasPaused = pauseRecognitionForTts();
  list.forEach((item, idx) => {
    const isLast = idx === list.length - 1;
    playbackQueue.enqueue({
      text: item.text,
      langCode: item.langCode,
      rate: item.rate || 1,
      onDone: isLast
        ? () => {
            if (wasPaused) resumeRecognitionAfterTts();
            if (typeof onAllDone === 'function') onAllDone();
          }
        : undefined,
    });
  });
}

function clearPendingCoach() {
  if (pendingCoachTimer) {
    clearTimeout(pendingCoachTimer);
    pendingCoachTimer = null;
  }
  pendingCoachIndex = -1;
}

// Read Along-style coaching, but only when the learner is actually stuck:
// arm a timer on a stumble and let any further speech cancel it, so coaching
// only fires after COACH_SILENCE_MS of quiet — never mid-sentence.
function scheduleCoachAfterSilence(index) {
  clearPendingCoach();
  if (index < 0) return;
  if (coachedWordIndices.has(index)) return;

  pendingCoachIndex = index;
  pendingCoachTimer = setTimeout(() => {
    const idx = pendingCoachIndex;
    clearPendingCoach();

    const sessionActive =
      state.recording || state.shouldAutoRestartRecognition || state.micPausedForTts;
    if (!sessionActive || state.sentenceComplete || state.manualStopRequested) return;

    // Only coach if the learner is still stuck on this exact word.
    const firstNotCorrect = wordStatus.findIndex((s) => s !== 'correct');
    if (idx !== firstNotCorrect) return;

    maybeCoachWrongWord(idx);
  }, COACH_SILENCE_MS);
}

// Say “Try saying …” in the base language, then model the word slowly in the
// target voice.
function maybeCoachWrongWord(index) {
  if (index < 0) return;
  if (state.sentenceComplete || state.manualStopRequested) return;
  if (coachedWordIndices.has(index)) return;
  const now = Date.now();
  if (now - lastCoachAt < COACH_COOLDOWN_MS) return;

  const surface = (wordSpans[index]?.dataset?.word || '').trim();
  if (!surface) return;

  const phrases = getFeedbackPhrases();
  coachedWordIndices.add(index);
  lastCoachAt = now;

  speakFeedbackItems([
    { text: phrases.tryWord, langCode: getLangCode(state.baseLang), rate: 1 },
    { text: surface, langCode: getLangCode(state.targetLang), rate: 0.8 },
  ]);
}

function pickPraisePhrase() {
  const { praise } = getFeedbackPhrases();
  if (!Array.isArray(praise) || !praise.length) return '';
  return praise[Math.floor(Math.random() * praise.length)];
}

// After a correct sentence: flash it green, speak praise, then move on and
// reopen the mic so the learner can keep reading without touching anything.
function celebrateSentenceComplete() {
  const token = sentenceNavToken;

  if (els.sentence) {
    els.sentence.classList.remove('sentence-complete');
    void els.sentence.offsetWidth;
    els.sentence.classList.add('sentence-complete');
  }

  const praise = pickPraisePhrase();
  speakFeedbackItems(
    praise ? [{ text: praise, langCode: getLangCode(state.baseLang), rate: 1 }] : [],
    () => {
      setTimeout(() => {
        if (token !== sentenceNavToken || !state.sentenceComplete) return;
        state.sentenceComplete = false;
        goToNext();
        if (state.supportsRecognition) startRecording();
      }, 400);
    }
  );
}

function getBestVoiceSync(langCode) {
  if (!isSpeechSynthesisSupported()) return null;
  const synth = window.speechSynthesis;
  if (!synth) return null;
  const voices = synth.getVoices();
  if (!voices || !voices.length) return null;

  const ranked = rankVoicesForLang(voices, langCode);
  return ranked[0] || null;
}

function speakCurrentImmediate(rate = 1, onDone) {
  if (!state.sentences.length) return false;
  const text = currentSentence().text;
  const langCode = getLangCode(state.targetLang);
  if (!text || !isSpeechSynthesisSupported()) return false;

  const synth = window.speechSynthesis;
  if (!synth) return false;

  // Mobile browsers are picky about gesture timing; keep this sync.
  try {
    synth.cancel();
    if (synth.paused) synth.resume();
  } catch (_) {}

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = langCode;
  utterance.rate = rate;
  const voice = getBestVoiceSync(langCode);
  if (voice) utterance.voice = voice;

  utterance.onend = () => {
    if (typeof onDone === 'function') onDone(true);
  };
  utterance.onerror = () => {
    // Fall back to the existing async pipeline (service TTS if needed).
    speakCurrent(rate, onDone);
  };

  synth.speak(utterance);
  return true;
}

async function handlePlaybackClick(rate) {
  if (isMobileDevice() && !state.audioUnlocked) {
    await unlockAudioPlayback();
    if (!state.audioUnlocked) {
      setStatus('Tap Play again to enable audio.');
      return;
    }
  }

  const wasPaused = pauseRecognitionForTts();
  const onDone = wasPaused ? () => resumeRecognitionAfterTts() : undefined;

  // On mobile, try a synchronous speechSynthesis play first — unless a neural
  // TTS server is configured, which beats any local browser voice.
  if (isMobileDevice() && !shouldPreferTtsService()) {
    // Try to force voices to load.
    try {
      if (window.speechSynthesis) window.speechSynthesis.getVoices();
    } catch (_) {}

    const started = speakCurrentImmediate(rate, onDone);
    if (started) return;
  }

  speakCurrent(rate, onDone);
}

async function togglePlayback() {
  const audio = state.ttsAudioElement;
  const synth = typeof window !== 'undefined' ? window.speechSynthesis : null;

  if (audio && !audio.paused && !audio.ended) {
    audio.pause();
    setStatus('Playback paused.');
    return;
  }

  if (synth && synth.speaking && !synth.paused) {
    synth.pause();
    setStatus('Playback paused.');
    return;
  }

  if (audio && audio.paused && audio.src && audio.currentTime > 0 && !audio.ended) {
    try {
      await audio.play();
      setStatus('Playback resumed.');
      return;
    } catch (err) {
      console.warn('Could not resume audio playback', err);
    }
  }

  if (synth && synth.paused) {
    synth.resume();

    // Some engines can report paused/speaking inconsistently after resume.
    // If resume does not actually continue playback, restart from the beginning
    // instead of getting stuck in a paused state.
    await new Promise((resolve) => requestAnimationFrame(resolve));
    if (synth.speaking && !synth.paused) {
      setStatus('Playback resumed.');
      return;
    }
  }

  await handlePlaybackClick(1);
}

function toggleRecording() {
  if (state.recording) {
    stopRecording();
    return;
  }
  startRecording();
}

function initRecognition() {
  if (!state.supportsRecognition) return;
  if (state.recognition) {
    state.recognition.abort();
  }

  state.recognition = new SpeechRecognition();
  state.recognition.lang = getRecognitionLangCode(state.targetLang);
  state.recognition.continuous = true;
  state.recognition.interimResults = true;
  state.recognition.maxAlternatives = 1;

  // Keep a stable transcript: accumulate finalized chunks + latest interim chunk.
  // This prevents duplicate loops like "سلامسلامسلام..." from overlapping interim results.
  let finalTranscript = '';

  state.recognition.onresult = (event) => {
    let interim = '';

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const res = event.results[i];
      const text = (res && res[0] && res[0].transcript ? res[0].transcript : '').trim();
      if (!text) continue;

      if (res.isFinal) {
        finalTranscript = (finalTranscript + ' ' + text).trim();
      } else {
        // Keep only the most recent interim (avoids repetition loops)
        interim = text;
      }
    }

    const transcript = (finalTranscript + (interim ? ' ' + interim : '')).trim();
    const lastResult = event.results[event.results.length - 1];
    const isFinalResult = Boolean(lastResult && lastResult.isFinal);

    updateLiveFeedback(transcript, { isFinalResult });
  };

  state.recognition.onstart = () => {
    // Reset per-recording session so we don't carry a previous sentence into the next attempt.
    finalTranscript = '';

    state.recording = true;
    state.micPausedForTts = false;
    setStatus('Listening...');
    updateRecordState();
    updateWordSpanClasses();
  };

  state.recognition.onerror = (event) => {
    console.error('Recognition error', event.error);
    state.recording = false;
    updateRecordState();

    if (
      state.shouldAutoRestartRecognition &&
      !state.manualStopRequested &&
      !state.sentenceComplete &&
      (event.error === 'no-speech' || event.error === 'aborted')
    ) {
      setStatus('No speech detected, still listening...');
      restartRecognitionSession();
      return;
    }

    setStatus(`Recognition error: ${event.error}`);
  };

  state.recognition.onend = () => {
    state.recording = false;
    updateRecordState();
    updateWordSpanClasses();

    const shouldFinalize = state.manualStopRequested || state.sentenceComplete;
    if (shouldFinalize && lastTranscript !== null) {
      finalizeScore(lastTranscript);
      if (state.sentenceComplete && state.pendingAutoAdvance) {
        scheduleNextSentenceAdvance();
      }
    } else if (state.shouldAutoRestartRecognition && !state.manualStopRequested) {
      restartRecognitionSession();
    }
  };

  updateRecordState();
}

function startRecording() {
  if (!state.supportsRecognition) {
    setStatus('Speech recognition is not supported in this browser.');
    return;
  }
  if (!state.recognition) initRecognition();
  try {
    clearNextSentenceTimer();
    lastTranscript = '';
    els.transcript.textContent = '';
    els.feedback.textContent = '';
    resetSentenceState();
    state.manualStopRequested = false;
    state.sentenceComplete = false;
    state.pendingAutoAdvance = false;
    state.shouldAutoRestartRecognition = true;
    state.micPausedForTts = false;
    coachedWordIndices.clear();
    lastCoachAt = 0;
    clearPendingCoach();
    clearRecognitionRestartTimer();
    state.recognition.lang = getRecognitionLangCode(state.targetLang);
    state.recognition.start();
  } catch (err) {
    console.error('Failed to start recognition', err);
    setStatus('Could not start recording.');
  }
}

function stopRecording() {
  if (!state.recognition) return;
  const sessionActive =
    state.recording || state.micPausedForTts || state.shouldAutoRestartRecognition;
  if (!sessionActive) return;

  state.manualStopRequested = true;
  state.shouldAutoRestartRecognition = false;
  state.micPausedForTts = false;
  clearPendingCoach();
  clearRecognitionRestartTimer();
  setStatus('Stopping...');
  try {
    state.recognition.stop();
  } catch (err) {
    /* already stopped */
  }

  // If recognition was already stopped (paused for TTS or between auto-restarts),
  // onend won't fire again, so finalize here.
  if (!state.recording && lastTranscript !== null) {
    finalizeScore(lastTranscript);
    setStatus('Stopped.');
    updateRecordState();
    updateWordSpanClasses();
  }
}

function updateRecordState() {
  const listening = state.recording || state.micPausedForTts;
  els.record.disabled = !state.supportsRecognition || listening;
  els.stop.disabled = !listening;
  els.record.classList.toggle('recording', listening);
  els.record.textContent = listening ? '🎙️ Listening…' : '🎙️ Record';
  if (!state.supportsRecognition) {
    els.status.textContent = 'Speech recognition not available in this browser.';
  }
}

function clearRecognitionRestartTimer() {
  if (recognitionRestartTimer) {
    clearTimeout(recognitionRestartTimer);
    recognitionRestartTimer = null;
  }
}

function clearNextSentenceTimer() {
  if (nextSentenceTimer) {
    clearTimeout(nextSentenceTimer);
    nextSentenceTimer = null;
  }
}

function scheduleNextSentenceAdvance() {
  clearNextSentenceTimer();
  nextSentenceTimer = setTimeout(() => {
    nextSentenceTimer = null;
    if (state.sentenceComplete && state.pendingAutoAdvance) {
      state.pendingAutoAdvance = false;
      goToNext();
    }
  }, 800);
}

function restartRecognitionSession() {
  if (!state.recognition || !state.shouldAutoRestartRecognition || state.manualStopRequested) {
    return;
  }

  clearRecognitionRestartTimer();
  recognitionRestartTimer = setTimeout(() => {
    recognitionRestartTimer = null;
    try {
      state.recognition.start();
      state.recording = true;
      setStatus('Listening...');
      updateRecordState();
    } catch (err) {
      console.warn('Failed to restart recognition', err);
    }
  }, 150);
}

function buildVoiceWarning(_targetLangCode) {
  // Intentionally quiet: we only show a simple browser fallback message when needed.
  return '';
}

function buildBrowserRecommendation() {
  if (typeof navigator === 'undefined') return '';
  const ua = navigator.userAgent.toLowerCase();
  const isOpera = ua.includes('opr/') || ua.includes('opera');
  const isChrome = ua.includes('chrome') && !ua.includes('edg') && !isOpera;
  const isEdge = ua.includes('edg');
  const isSafari = ua.includes('safari') && !ua.includes('chrome') && !ua.includes('crios') && !ua.includes('fxios');
  const isFirefox = ua.includes('firefox') || ua.includes('fxios');

  // If you're already on a mainstream browser, don't nag.
  if (isChrome || isEdge || isSafari || isFirefox) return '';

  return 'If audio fails, try another browser.';
}

function updatePlaybackWarnings() {
  if (!els.playbackWarnings) return;
  const parts = [];
  const voiceWarning = buildVoiceWarning(state.targetLang);
  if (voiceWarning) parts.push(voiceWarning);
  const playbackUnavailable = !state.supportsSpeechSynthesis && !state.supportsTtsService;
  if (voiceWarning || playbackUnavailable) {
    const browserWarning = buildBrowserRecommendation();
    if (browserWarning) parts.push(browserWarning);
  }
  if (isMobileDevice()) {
    // Keep warnings minimal.
  }
  els.playbackWarnings.textContent = parts.join(' ');
}

function updateSpeechSynthesisState({ announce = false } = {}) {
  state.supportsSpeechSynthesis = isSpeechSynthesisSupported();
  state.supportsTtsService = Boolean(getTtsBaseUrl());
  const supported = state.supportsSpeechSynthesis || state.supportsTtsService;

  if (els.play) {
    els.play.disabled = !supported || state.ttsLoading;
    els.play.title = supported ? '' : NO_TTS_SUPPORT_MESSAGE;
  }

  if (els.slow) {
    els.slow.disabled = !supported || state.ttsLoading;
    els.slow.title = supported ? '' : NO_TTS_SUPPORT_MESSAGE;
  }

  if (!supported && announce) {
    setStatus(NO_TTS_SUPPORT_MESSAGE);
  }

  updatePlaybackWarnings();
}

function normalizeArabicToken(s) {
  if (!s) return '';
  s = String(s).normalize('NFC');
  // Remove Arabic diacritics + tatweel
  s = s.replace(/[\u064B-\u065F\u0670\u0640]/g, '');
  // Unify common letter variants
  s = s
    .replace(/[\u0622\u0623\u0625\u0671]/g, 'ا') // آأإٱ -> ا
    .replace(/\u0624/g, 'و') // ؤ -> و
    .replace(/\u0626/g, 'ي') // ئ -> ي
    .replace(/\u0629/g, 'ه') // ة -> ه
    .replace(/\u0649/g, 'ي'); // ى -> ي
  // Remove punctuation/whitespace-like chars (incl Arabic punctuation)
  s = s.replace(/[\s\u200f\u200e\u060C\u061B\u061F.,!?;:;()"«»¿¡]/g, '');
  return s;
}

function normalizeWord(w) {
  if (!w) return '';
  let s = String(w).trim();

  // If this is Darija mode, normalize Arabic-script tokens more aggressively
  // so variants like "أمينة" vs "امينه" match.
  if (state && state.targetLang === 'ma') {
    const arabic = normalizeArabicToken(s);
    if (arabic) return arabic;
  }

  return s
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[.,!?;:;()"«»¿¡]/g, '')
    .trim();
}

const NUMBER_WORDS_BY_LANG = {
  en: {
    small: {
      0: 'zero',
      1: 'one',
      2: 'two',
      3: 'three',
      4: 'four',
      5: 'five',
      6: 'six',
      7: 'seven',
      8: 'eight',
      9: 'nine',
      10: 'ten',
      11: 'eleven',
      12: 'twelve',
      13: 'thirteen',
      14: 'fourteen',
      15: 'fifteen',
      16: 'sixteen',
      17: 'seventeen',
      18: 'eighteen',
      19: 'nineteen',
      20: 'twenty',
    },
    tens: {
      20: 'twenty',
      30: 'thirty',
      40: 'forty',
      50: 'fifty',
      60: 'sixty',
      70: 'seventy',
      80: 'eighty',
      90: 'ninety',
    },
  },
  fr: {
    small: {
      0: 'zéro',
      1: 'un',
      2: 'deux',
      3: 'trois',
      4: 'quatre',
      5: 'cinq',
      6: 'six',
      7: 'sept',
      8: 'huit',
      9: 'neuf',
      10: 'dix',
      11: 'onze',
      12: 'douze',
      13: 'treize',
      14: 'quatorze',
      15: 'quinze',
      16: 'seize',
      17: 'dix-sept',
      18: 'dix-huit',
      19: 'dix-neuf',
      20: 'vingt',
    },
    tens: {
      20: 'vingt',
      30: 'trente',
      40: 'quarante',
      50: 'cinquante',
      60: 'soixante',
      70: 'soixante-dix',
      80: 'quatre-vingt',
      90: 'quatre-vingt-dix',
    },
  },
  ca: {
    small: {
      0: 'zero',
      1: 'u',
      2: 'dos',
      3: 'tres',
      4: 'quatre',
      5: 'cinc',
      6: 'sis',
      7: 'set',
      8: 'vuit',
      9: 'nou',
      10: 'deu',
      11: 'onze',
      12: 'dotze',
      13: 'tretze',
      14: 'catorze',
      15: 'quinze',
      16: 'setze',
      17: 'disset',
      18: 'divuit',
      19: 'dinou',
      20: 'vint',
    },
    tens: {
      20: 'vint',
      30: 'trenta',
      40: 'quaranta',
      50: 'cinquanta',
      60: 'seixanta',
      70: 'setanta',
      80: 'vuitanta',
      90: 'noranta',
    },
  },
  it: {
    small: {
      0: 'zero',
      1: 'uno',
      2: 'due',
      3: 'tre',
      4: 'quattro',
      5: 'cinque',
      6: 'sei',
      7: 'sette',
      8: 'otto',
      9: 'nove',
      10: 'dieci',
      11: 'undici',
      12: 'dodici',
      13: 'tredici',
      14: 'quattordici',
      15: 'quindici',
      16: 'sedici',
      17: 'diciassette',
      18: 'diciotto',
      19: 'diciannove',
      20: 'venti',
    },
    tens: {
      20: 'venti',
      30: 'trenta',
      40: 'quaranta',
      50: 'cinquanta',
      60: 'sessanta',
      70: 'settanta',
      80: 'ottanta',
      90: 'novanta',
    },
  },
  ma: {
    small: {
      0: 'sefr',
      1: 'wahed',
      2: 'jouj',
      3: 'tlata',
      4: 'rb3a',
      5: 'khamsa',
      6: 'stta',
      7: 'sb3a',
      8: 'tmnya',
      9: 'ts3ud',
      10: '3shra',
      11: 'hda3sh',
      12: 'tna3sh',
      13: 'tlata3sh',
      14: 'rb3ta3sh',
      15: 'khamsa3sh',
      16: 'stta3sh',
      17: 'sb3a3sh',
      18: 'tmnya3sh',
      19: 'ts3ud3sh',
      20: '3shrin',
    },
    tens: {
      20: '3shrin',
      30: 'tlata3shrin',
      40: 'rb3in',
      50: 'khamsin',
      60: 'sttin',
      70: 'sb3in',
      80: 'tmanin',
      90: 'ts3in',
    },
  },
};

const DIGIT_TOKEN_PATTERN = /^\d+$/;
const TIME_TOKEN_PATTERN = /\b\d{1,2}:\d{2}\b/g;
const TIME_TOKEN_EXACT_PATTERN = /^\d{1,2}:\d{2}$/;

function getHourWord(hourToken, langCode) {
  const normalizedLang = (langCode || state.targetLang || 'en').split('-')[0];
  const hourNum = Number.parseInt(hourToken, 10);
  if (!Number.isFinite(hourNum)) return '';
  if (normalizedLang === 'fr' && hourNum === 1) return 'une';
  return digitToNumberWord(hourToken, normalizedLang);
}

function timeTokenToWords(token, langCode) {
  if (!TIME_TOKEN_EXACT_PATTERN.test(token)) return '';
  const [hours, minutes] = token.split(':');
  if (minutes !== '00') return '';
  const normalizedLang = (langCode || state.targetLang || 'en').split('-')[0];
  const hourWord = getHourWord(hours, normalizedLang);
  if (!hourWord) return '';

  switch (normalizedLang) {
    case 'en':
      return `${hourWord} o'clock`;
    case 'fr': {
      const hourNum = Number.parseInt(hours, 10);
      const noun = hourNum === 1 ? 'heure' : 'heures';
      return `${hourWord} ${noun}`;
    }
    default:
      return '';
  }
}

function normalizeTimeTokens(text, langCode) {
  return (text || '').replace(TIME_TOKEN_PATTERN, (match) => {
    const replacement = timeTokenToWords(match, langCode);
    return replacement || match;
  });
}

function digitToNumberWord(token, langCode) {
  const normalizedLang = (langCode || state.targetLang || 'en').split('-')[0];
  const num = Number.parseInt(token, 10);
  if (!Number.isFinite(num)) return '';
  const mapForLang = NUMBER_WORDS_BY_LANG[normalizedLang] || NUMBER_WORDS_BY_LANG.en;
  if (mapForLang?.small && Object.prototype.hasOwnProperty.call(mapForLang.small, num)) {
    return mapForLang.small[num];
  }
  if (mapForLang?.tens && Object.prototype.hasOwnProperty.call(mapForLang.tens, num)) {
    return mapForLang.tens[num];
  }
  return token;
}

function normalizeToken(rawToken, langCode) {
  let token = normalizeWord(rawToken);
  if (!token) return '';

  // Darija (MA) latin: accept common vowel-omission variants.
  // e.g. "slam" (compact) ≈ "salam" (learner-friendly)
  if ((langCode || state.targetLang) === 'ma' && /[a-z]/i.test(token)) {
    if (token === 'slam') token = 'salam';
  }

  if (DIGIT_TOKEN_PATTERN.test(token)) {
    const numberWord = digitToNumberWord(token, langCode);
    return normalizeWord(numberWord);
  }
  return token;
}

function tokenizeText(t, langCode) {
  const timeNormalized = normalizeTimeTokens(t, langCode);

  // For Arabic-script (Darija) we can't rely on whitespace tokenization.
  // Also strip Arabic punctuation like "،" and normalize before splitting.
  if ((langCode || state.targetLang) === 'ma') {
    const cleaned = normalizeWord(timeNormalized)
      .replace(/[\u060C\u061B\u061F]/g, ' ') // Arabic comma/semicolon/question
      .replace(/[.,!?;:;()"«»¿¡]/g, ' ');

    // If there are spaces, use them. Otherwise keep as one token.
    const parts = cleaned.includes(' ') ? cleaned.split(/\s+/) : [cleaned];
    return parts.map((token) => normalizeToken(token, langCode)).filter(Boolean);
  }

  return normalizeWord(timeNormalized)
    .split(/\s+/)
    .map((token) => normalizeToken(token, langCode))
    .filter(Boolean);
}

function buildMaxConsecutiveRuns(tokens) {
  const maxRuns = new Map();
  let prev = null;
  let runLength = 0;

  tokens.forEach((token) => {
    if (token === prev) {
      runLength += 1;
    } else {
      prev = token;
      runLength = 1;
    }

    const currentMax = maxRuns.get(token) || 0;
    if (runLength > currentMax) {
      maxRuns.set(token, runLength);
    }
  });

  return maxRuns;
}

function collapseConsecutiveDuplicates(tokens, maxRunsByToken, defaultMaxRun = 1) {
  const out = [];
  let prev = null;
  let run = 0;
  for (const t of tokens) {
    if (t === prev) {
      run += 1;
    } else {
      prev = t;
      run = 1;
    }
    const maxRun = Math.max(maxRunsByToken?.get(t) || 0, defaultMaxRun);
    if (run <= maxRun) out.push(t);
  }
  return out;
}

function filterUnexpectedRepeats(transcript, targetTokensForSentence, langCode) {
  const spokenTokens = tokenizeText(transcript, langCode);
  const targetTokensNormalized = targetTokensForSentence || [];
  const allowedRuns = buildMaxConsecutiveRuns(targetTokensNormalized);
  const allowedCounts = new Map();

  targetTokensNormalized.forEach((token) => {
    allowedCounts.set(token, (allowedCounts.get(token) || 0) + 1);
  });

  const filteredTokens = [];
  const seenCounts = new Map();
  let prev = null;
  let runLength = 0;

  // Special case: Darija Arabic scripts often come back without spaces.
  // If the transcript is one long token, prefer showing it once in the UI.
  const uiSpokenTokens =
    (langCode || state.targetLang) === 'ma' && spokenTokens.length === 1
      ? [spokenTokens[0]]
      : collapseConsecutiveDuplicates(spokenTokens, allowedRuns, 1);

  uiSpokenTokens.forEach((token) => {
    if (token === prev) {
      runLength += 1;
    } else {
      prev = token;
      runLength = 1;
    }

    const allowedRun = allowedRuns.get(token) || 1;
    if (runLength > allowedRun) {
      return;
    }

    const allowedTotal = allowedCounts.has(token) ? allowedCounts.get(token) : 1;
    const seen = seenCounts.get(token) || 0;
    if (seen >= allowedTotal) {
      return;
    }

    seenCounts.set(token, seen + 1);
    filteredTokens.push(token);
  });

  return {
    filteredTokens,
    filteredTranscript: filteredTokens.join(' '),
    rawTranscript: transcript,
  };
}

function levenshtein(a, b) {
  const n = a.length;
  const m = b.length;
  if (n === 0) return m;
  if (m === 0) return n;

  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 0; i <= n; i++) dp[i][0] = i;
  for (let j = 0; j <= m; j++) dp[0][j] = j;

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }
  return dp[n][m];
}

const EQUIV_GROUPS = {
  em: ['em', 'amb', 'en'],
  amb: ['em', 'amb', 'en'],
  u: ['u', 'un'],
  un: ['u', 'un'],
};

const APPROX_RULES_BY_LANG = {
  default: {
    coverageThreshold: DEFAULT_APPROX_THRESHOLD,
    properNounThreshold: 0.6,
  },
};

function getApproxRules(langCode) {
  const baseRules =
    langCode && APPROX_RULES_BY_LANG[langCode]
      ? APPROX_RULES_BY_LANG[langCode]
      : APPROX_RULES_BY_LANG.default || {};
  const coverageThreshold = getApproxThresholdFromIndex(state.approxLevelIndex);
  const properNounThreshold = Math.min(
    DEFAULT_MATCH_THRESHOLD,
    Math.max(PROPER_NOUN_THRESHOLD_FLOOR, coverageThreshold)
  );
  return { ...baseRules, coverageThreshold, properNounThreshold };
}

function getTokenMatchThreshold(targetToken, langCode) {
  const isProperNoun = Boolean(targetToken && typeof targetToken === 'object' && targetToken.isProperNoun);
  if (!isProperNoun) return DEFAULT_MATCH_THRESHOLD;
  const { properNounThreshold } = getApproxRules(langCode);
  if (!Number.isFinite(properNounThreshold)) return DEFAULT_MATCH_THRESHOLD;
  return properNounThreshold;
}

function orderedCharacterCoverage(target, candidate) {
  if (!target) return 0;
  if (!candidate) return 0;
  let tIndex = 0;
  let cIndex = 0;
  let matched = 0;

  while (tIndex < target.length && cIndex < candidate.length) {
    if (target[tIndex] === candidate[cIndex]) {
      matched += 1;
      tIndex += 1;
      cIndex += 1;
    } else {
      cIndex += 1;
    }
  }

  return matched / target.length;
}

function passesApproximationRule(target, candidate, langCode) {
  const { coverageThreshold } = getApproxRules(langCode);
  if (!coverageThreshold) return false;
  // Coverage only counts target letters found inside the candidate, so tiny
  // words get free passes: "a" is fully "covered" by any word containing an
  // a, "et" by half the words in French. Short words must match exactly (or
  // via EQUIV_GROUPS/Levenshtein upstream) — leniency is for long words.
  if (target.length <= 2) return false;
  // Likewise, a candidate much longer than the target means the target is
  // just buried inside a different word — except the recognizer-merge case
  // ("bondia" for "bon dia"), where the candidate starts with the target.
  if (candidate.length > target.length + 2 && !candidate.startsWith(target)) return false;
  return orderedCharacterCoverage(target, candidate) >= coverageThreshold;
}

function normalizeDarijaLatnVariant(token) {
  if (!token) return '';
  let s = String(token).toLowerCase();
  // formatting variants
  s = s.replace(/[\-’'`]/g, '');
  // common digraph variation seen in user data
  s = s.replace(/\bch\b/g, 'sh');
  // collapse repeated letters (e.g. aa -> a)
  s = s.replace(/([a-z])\1{1,}/g, '$1');
  return s;
}

function darijaConsonantSkeleton(token) {
  // A conservative "skeleton" matcher for MA latin:
  // remove short vowels so "slam" ~ "salam" without hardcoding every word.
  const s = normalizeDarijaLatnVariant(token);
  return s.replace(/[aeiouə]/g, '');
}

function similarityScore(rawTarget, rawCandidate, langCode) {
  const target = normalizeToken(rawTarget, langCode);
  const cand = normalizeToken(rawCandidate, langCode);
  if (!target || !cand) return 0;

  if (target === cand) return 1;

  // Elision apostrophes (l'a, d'accord, qu'il...) and English contractions
  // (don't, it's...) are silent/fused in speech — recognizers commonly drop
  // or fuse across them. Score an apostrophe-insensitive match as exact,
  // without touching normalizeWord/tokenizeText (which other code — repeat
  // filtering, digit counting — depends on keeping apostrophes intact).
  const targetNoApos = target.replace(/['’`]/g, '');
  const candNoApos = cand.replace(/['’`]/g, '');
  if (targetNoApos && targetNoApos === candNoApos) {
    return 1;
  }

  // Darija latin: allow vowel-omission variants by comparing consonant skeletons.
  if ((langCode || state.targetLang) === 'ma' && /[a-z]/i.test(target) && /[a-z]/i.test(cand)) {
    const skT = darijaConsonantSkeleton(target);
    const skC = darijaConsonantSkeleton(cand);
    if (skT && skT === skC) {
      return 0.92;
    }
  }

  const equiv = EQUIV_GROUPS[target];
  if (equiv && equiv.includes(cand)) {
    return 0.95;
  }

  if (passesApproximationRule(target, cand, langCode)) {
    return 1;
  }

  const dist = levenshtein(target, cand);
  const maxLen = Math.max(target.length, cand.length);
  const normDist = maxLen === 0 ? 0 : dist / maxLen;
  return 1 - normDist;
}

function scoreTokenWithAliases(targetToken, candidate, langCode) {
  if (!targetToken || !candidate) return 0;

  const primary =
    typeof targetToken === 'string' ? targetToken : targetToken.text || targetToken.surface || '';
  const aliases =
    typeof targetToken === 'string'
      ? []
      : Array.isArray(targetToken.aliases)
        ? targetToken.aliases
        : Array.isArray(targetToken.pronunciation_aliases)
          ? targetToken.pronunciation_aliases
          : [];

  let bestScore = similarityScore(primary, candidate, langCode);
  aliases.forEach((alias) => {
    const aliasScore = similarityScore(alias, candidate, langCode);
    if (aliasScore > bestScore) {
      bestScore = aliasScore;
    }
  });

  return bestScore;
}

function findMatchesForTargetTokens(targetTokens, spokenTokens, { langCode } = {}) {
  const matches = new Array(targetTokens.length).fill(null);
  const usedUntil = { value: 0 };
  const targetLang = langCode || state.targetLang;

  const S = spokenTokens.length;
  const MAX_WINDOW = 3;

  for (let i = 0; i < targetTokens.length; i++) {
    const target = targetTokens[i];
    const tokenThreshold = getTokenMatchThreshold(target, targetLang);
    let best = null;

    for (let start = usedUntil.value; start < S; start++) {
      for (let end = start; end < Math.min(S, start + MAX_WINDOW); end++) {
        const windowTokens = spokenTokens.slice(start, end + 1);
        const concatenated = normalizeWord(windowTokens.join(''));
        const score = scoreTokenWithAliases(target, concatenated, targetLang);
        if (!best || score > best.score) {
          best = { start, end, score };
        }
      }
    }

    if (best && best.score >= tokenThreshold) {
      matches[i] = best;
      usedUntil.value = best.end + 1;
    } else {
      matches[i] = null;
    }
  }

  return matches;
}

// A single word that doesn't clear the match threshold (e.g. an elision like
// "l'a" scoring just under it) shouldn't block every later word from
// lighting up — those were still spoken in order right after it. So the gap
// is measured from the nearest earlier word actually confirmed correct, not
// from the locked prefix: one skipped/pending word in between still counts
// as "sequential" and turns green instantly. Only a jump of two or more
// unpronounced words — the recognizer speculatively completing a whole
// phrase ahead of the speaker — gets held back in `wordStatus` until it
// survives OUT_OF_ORDER_CONFIRM_MS of interim updates (tracked per word
// index in `sinceMap`).
//
// Exception: a word whose identical text already appeared EARLIER in the
// sentence ("a", "the", "le"...) never lights out of order at all, not even
// after the delay. Once the learner has said the word once it exists in the
// transcript permanently, so any leftover/echoed copy of it is stable
// "evidence" that would always survive the delay — the delay only filters
// transient noise, and this noise isn't transient. Repeated words light only
// when the reading actually reaches them (gap 0), which is also the only
// position where the evidence is unambiguous.
//
// Final results are stable, so they bypass the buffer entirely.
// Mutates `wordStatus` and `sinceMap` in place.
function applyOutOfOrderConfirmation(wordStatus, targetTokens, lockedPrefix, isFinalResult, sinceMap, now = Date.now()) {
  const OUT_OF_ORDER_GAP = 2;
  const n = wordStatus.length;

  // Apostrophe-insensitive, matching the elision equivalence in similarityScore.
  const wordKey = (t) => String(t || '').replace(/['’`]/g, '');
  const seenWords = new Set();
  const hasEarlierTwin = new Array(n).fill(false);
  for (let i = 0; i < n; i++) {
    const key = wordKey(targetTokens[i]);
    hasEarlierTwin[i] = seenWords.has(key);
    seenWords.add(key);
  }

  // Snapshot before mutating: the cleanup pass below must judge staleness
  // against what matches[] actually found this round, not against statuses
  // this same function demotes back to 'pending' while buffering them.
  const originalStatus = wordStatus.slice();
  let lastGoodIndex = lockedPrefix - 1;

  for (let i = lockedPrefix; i < n; i++) {
    if (wordStatus[i] !== 'correct') continue;
    const gap = i - lastGoodIndex - 1;
    if (isFinalResult || gap === 0) {
      lastGoodIndex = i;
      sinceMap.delete(i);
      continue;
    }
    if (hasEarlierTwin[i]) {
      sinceMap.delete(i);
      wordStatus[i] = 'pending';
      continue;
    }
    if (gap < OUT_OF_ORDER_GAP) {
      lastGoodIndex = i;
      sinceMap.delete(i);
      continue;
    }
    const since = sinceMap.get(i);
    if (since !== undefined && now - since >= OUT_OF_ORDER_CONFIRM_MS) {
      lastGoodIndex = i;
      sinceMap.delete(i);
    } else {
      if (since === undefined) sinceMap.set(i, now);
      wordStatus[i] = 'pending';
    }
  }

  for (const idx of sinceMap.keys()) {
    if (idx < lockedPrefix || originalStatus[idx] !== 'correct' || isFinalResult) {
      sinceMap.delete(idx);
    }
  }
}

function resetSentenceState() {
  lastTranscript = '';
  wordStatus = targetTokens.map(() => 'pending');
  outOfOrderCorrectSince.clear();
  updateWordSpanClasses();
}

function updateWordSpanClasses() {
  const listening = state.recording || state.micPausedForTts;
  let nextIndex = -1;
  if (listening && !state.sentenceComplete) {
    nextIndex = wordStatus.findIndex((s) => s !== 'correct');
  }

  wordSpans.forEach((span, i) => {
    span.classList.remove('word-correct', 'word-wrong', 'word-pending', 'word-next');
    const status = wordStatus[i] || 'pending';
    if (status === 'correct') span.classList.add('word-correct');
    else if (status === 'wrong') span.classList.add('word-wrong');
    else span.classList.add('word-pending');
    if (i === nextIndex) span.classList.add('word-next');
  });
}

function checkIfSentenceCompleteAndStop({ allowAutoStop = false } = {}) {
  if (!allowAutoStop) return;
  if (state.sentenceComplete) return;
  if (!wordStatus.length) return;
  const allCorrect = wordStatus.every((s) => s === 'correct');
  if (allCorrect && state.recognition) {
    state.sentenceComplete = true;
    state.pendingAutoAdvance = false;
    state.shouldAutoRestartRecognition = false;
    state.manualStopRequested = false;
    state.micPausedForTts = false;
    clearPendingCoach();
    clearRecognitionRestartTimer();
    try {
      state.recognition.stop();
    } catch (e) {
      // ignore
    }
    setStatus('Sentence complete ✅');
    celebrateSentenceComplete();
  }
}

function shouldIgnoreHotkey(event) {
  if (event.metaKey || event.ctrlKey || event.altKey) return true;
  const target = event.target;
  if (!target || !(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (target.isContentEditable) return true;
  if (els.customModal && !els.customModal.classList.contains('hidden')) return true;
  return false;
}

function buildDarijaSpokenTokens(transcript, sentence) {
  const raw = String(transcript || '').trim();
  if (!raw) return { spokenTokens: [], spokenTranscript: '' };

  // If recognizer gives Latin/space-delimited text, just tokenize normally.
  if (!/[\u0600-\u06FF]/.test(raw) && /[a-zA-Z]/.test(raw)) {
    const spokenTokens = tokenizeText(raw, 'ma');
    return { spokenTokens, spokenTranscript: spokenTokens.join(' ') };
  }

  const tokens = sentence?.tokens || [];
  if (!tokens.length) {
    const spokenTokens = tokenizeText(raw, 'ma');
    return { spokenTokens, spokenTranscript: spokenTokens.join(' ') };
  }

  // Arabic-script path: segment the recognized blob by fuzzily locating each
  // expected token's Arabic surface (a single misheard/mistranscribed letter
  // shouldn't silently drop the whole word), then emit the aligned ma_latn
  // token (transcription) for scoring. Uses the same coverage/Levenshtein
  // scoring and threshold as every other language's word matching, so the
  // Accuracy slider actually applies to Darija written in Arabic script too.
  const recognized = normalizeArabicToken(raw);
  let pos = 0;
  const spokenTokens = [];

  tokens.forEach((t) => {
    const surface = normalizeArabicToken(t.surface || '');
    const latn = normalizeToken(t.transcription || '', 'ma');
    if (!surface || !latn) return;

    const match = findBestArabicSurfaceWindow(surface, recognized, pos);
    if (match && match.score >= getTokenMatchThreshold(t, 'ma')) {
      spokenTokens.push(latn);
      pos = match.end;
    }
  });

  return { spokenTokens, spokenTranscript: spokenTokens.join(' ') };
}

// Scores a candidate Arabic-script window the same way similarityScore()
// scores latin tokens elsewhere: exact match, then the slider-driven coverage
// shortcut, then a Levenshtein-distance fallback.
function arabicSurfaceSimilarity(target, candidate) {
  if (!target || !candidate) return 0;
  if (target === candidate) return 1;
  if (passesApproximationRule(target, candidate, 'ma')) return 1;
  const dist = levenshtein(target, candidate);
  const maxLen = Math.max(target.length, candidate.length);
  return maxLen === 0 ? 0 : 1 - dist / maxLen;
}

// Slides a window (within a couple characters of the expected surface's
// length) across the recognized blob starting at searchFrom, and returns the
// best-scoring window — the closest thing to "where in the blob was this
// word said", even if the recognizer got a letter or two wrong.
function findBestArabicSurfaceWindow(surface, recognized, searchFrom) {
  const minLen = Math.max(1, surface.length - 2);
  const maxLen = surface.length + 2;
  let best = null;

  for (let start = searchFrom; start < recognized.length; start++) {
    for (let len = minLen; len <= maxLen; len++) {
      const end = start + len;
      if (end > recognized.length) break;
      const candidate = recognized.slice(start, end);
      const score = arabicSurfaceSimilarity(surface, candidate);
      if (!best || score > best.score) {
        best = { start, end, score };
      }
    }
  }

  return best;
}

function updateLiveFeedback(transcript, { isFinalResult = false } = {}) {
  // Fresh speech activity: the learner is still reading, so cancel any
  // coaching that was waiting for silence.
  clearPendingCoach();

  const sentence = currentSentence();

  // For Darija, score against ma_latn token backbone.
  const darijaResult = state.targetLang === 'ma' ? buildDarijaSpokenTokens(transcript, sentence) : null;
  const baseResult =
    state.targetLang === 'ma'
      ? { filteredTokens: darijaResult.spokenTokens, filteredTranscript: darijaResult.spokenTranscript, rawTranscript: transcript }
      : filterUnexpectedRepeats(transcript, targetTokens, state.targetLang);

  const { filteredTokens, filteredTranscript, rawTranscript } = baseResult;

  lastTranscript = filteredTranscript;

  const prevStatus = [...wordStatus];
  const n = targetTokens.length;
  let lockedPrefix = 0;

  while (lockedPrefix < prevStatus.length && prevStatus[lockedPrefix] === 'correct') {
    lockedPrefix += 1;
  }

  wordStatus = targetTokens.map(() => 'pending');

  for (let i = 0; i < lockedPrefix; i++) {
    wordStatus[i] = 'correct';
  }

  // IMPORTANT: don't slice spoken tokens by lockedPrefix.
  // SpeechRecognition can merge/split words (e.g., "bon dia" -> "bondia"),
  // so assuming 1 spoken token per target token can drop remaining words.
  const matches = findMatchesForTargetTokens(
    targetTokenVariants.slice(lockedPrefix),
    filteredTokens
  );

  for (let i = 0; i < matches.length; i++) {
    if (matches[i]) {
      wordStatus[i + lockedPrefix] = 'correct';
    }
  }

  applyOutOfOrderConfirmation(wordStatus, targetTokens, lockedPrefix, isFinalResult, outOfOrderCorrectSince);

  let stumbledIndex = -1;
  if (isFinalResult) {
    let firstNotCorrect = -1;
    for (let i = lockedPrefix; i < n; i++) {
      if (wordStatus[i] !== 'correct') {
        firstNotCorrect = i;
        break;
      }
    }
    if (firstNotCorrect !== -1) {
      wordStatus[firstNotCorrect] = 'wrong';
      // Only coach when the learner actually attempted something.
      if (filteredTokens.length) {
        stumbledIndex = firstNotCorrect;
      }
    }
  }

  for (let i = 0; i < n; i++) {
    const wasWrong = prevStatus[i] === 'wrong';
    const isNowCorrect = wordStatus[i] === 'correct';
    if (wasWrong && isNowCorrect) {
      for (let k = Math.max(i + 1, lockedPrefix); k < n; k++) {
        if (wordStatus[k] === 'correct') {
          wordStatus[k] = 'pending';
          outOfOrderCorrectSince.delete(k);
        }
      }
      break;
    }
  }

  updateWordSpanClasses();

  if (els.transcript) {
    let displayTranscript = filteredTranscript;

    // Darija STT on mobile can return a repeated, no-whitespace loop.
    // For UX, show a clean single sentence instead of the repeated blob.
    if (state.targetLang === 'ma') {
      const raw = (rawTranscript || '').trim();
      const hasWhitespace = /\s/.test(raw);
      if (!hasWhitespace && raw.length > 40) {
        // Prefer the expected sentence text (Arabic) when available.
        try {
          displayTranscript = currentSentence().text || filteredTranscript;
        } catch (_) {
          displayTranscript = filteredTranscript;
        }
      }
    }

    els.transcript.textContent = displayTranscript;
  }

  checkIfSentenceCompleteAndStop({ allowAutoStop: true });

  if (!state.sentenceComplete && stumbledIndex !== -1) {
    scheduleCoachAfterSilence(stumbledIndex);
  }
}

function finalizeScore(transcript) {
  const sentence = currentSentence();
  const darijaResult = state.targetLang === 'ma' ? buildDarijaSpokenTokens(transcript, sentence) : null;
  const baseResult =
    state.targetLang === 'ma'
      ? { filteredTokens: darijaResult.spokenTokens, filteredTranscript: darijaResult.spokenTranscript, rawTranscript: transcript }
      : filterUnexpectedRepeats(transcript, targetTokens, state.targetLang);

  const { filteredTokens, filteredTranscript, rawTranscript } = baseResult;
  lastTranscript = filteredTranscript;

  const matches = findMatchesForTargetTokens(targetTokenVariants, filteredTokens);
  const correct = matches.filter(Boolean).length;

  const score = Math.round((correct / (targetTokens.length || 1)) * 100);
  const feedbackEl = els.feedback;
  if (!feedbackEl) return;

  feedbackEl.textContent = '';

  saveProgress(score);
}

function setStatus(text) {
  els.status.textContent = text;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    tokenizeText,
    buildMaxConsecutiveRuns,
    filterUnexpectedRepeats,
    rankVoicesForLang,
    getVoiceNaturalness,
    splitIntoSentences,
    buildDarijaSpokenTokens,
    applyOutOfOrderConfirmation,
    findMatchesForTargetTokens,
  };
}
