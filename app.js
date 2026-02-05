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
const VOICE_STORAGE_KEY = 'speechtoipa-voices';
const DEFAULT_TTS_BASE_URL = 'https://translate.googleapis.com';
const DEFAULT_APPROX_THRESHOLD = 0.65;
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
const AUTO_VOICE_VALUE = 'auto';
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
  return `${TTS_CACHE_PREFIX}${lang}:${text}`;
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
let lastTranscript = '';
let currentSentenceText = '';
let wordStatus = [];
let wordTooltipEl;
let sentenceTooltipEl;
let sentenceTooltipTimer = null;
let currentTooltipTarget = null;
let hasWarnedAboutArabicVoice = false;
let lastLessonId = '';
let recognitionRestartTimer = null;
let nextSentenceTimer = null;
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
  ttsLoading: false,
  approxLevelIndex: DEFAULT_CEFR_INDEX,
  audioUnlocked: false,
  ttsAudioElement: null,
};

const els = {};
let normalizedVoices = [];
let voiceSelections = {};
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

function parseCsvToObjects(text) {
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
  const rows = parseCsvToObjects(text);
  MASTER_ROWS_BY_LANG[lang] = rows;

  console.log('Loaded master rows for', lang, 'count =', rows.length);
  return rows;
}

if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', async () => {
    cacheElements();
    createTooltips();
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
  els.voiceSelect = document.getElementById('voice-select');
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
  populateVoiceSelect();
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

function getStoredVoices() {
  try {
    const raw = localStorage.getItem(VOICE_STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function persistVoices() {
  try {
    localStorage.setItem(VOICE_STORAGE_KEY, JSON.stringify(voiceSelections));
  } catch {
    /* ignore */
  }
}

function normalizeVoiceList(voices, langCode) {
  const rankedVoices = rankVoicesForLang(voices, langCode);
  return rankedVoices.map((v) => ({
    name: v.name,
    lang: v.lang,
    deviceSupport: Boolean(v.localService),
    ready: readyVoiceKeys.has(getVoiceKey(v)),
  }));
}

function populateVoiceSelect() {
  if (!els.voiceSelect) return;

  if (window.speechSynthesis) {
    const voices = window.speechSynthesis.getVoices();
    if (voices && voices.length) {
      markVoicesReady(voices);
      normalizedVoices = normalizeVoiceList(voices, getLangCode(state.targetLang));
    }
  }

  const options = [{ value: AUTO_VOICE_VALUE, label: 'Auto (recommended)' }];
  normalizedVoices.forEach((v) => {
    options.push({
      value: `${v.lang}|${v.name}`,
      label: `${v.name} (${v.lang}${v.deviceSupport ? ', device' : ''}${v.ready ? ', ready' : ''})`,
    });
  });

  els.voiceSelect.innerHTML = '';
  options.forEach((opt) => {
    const option = document.createElement('option');
    option.value = opt.value;
    option.textContent = opt.label;
    els.voiceSelect.appendChild(option);
  });

  const stored = voiceSelections[state.targetLang] || AUTO_VOICE_VALUE;
  els.voiceSelect.value = stored;
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

  // If we have a CSV for this language, build lessons from it.
  const rows = await ensureMasterRowsForLang(lang);
  if (rows && rows.length) {
    const lessonMap = new Map();

    rows.forEach((row) => {
      if (!row.lesson_id) return;
      if (!lessonMap.has(row.lesson_id)) {
        // For now, use lesson_title as theme + label.
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
    return;
  }

  // Fallback: old JSON manifest (for languages without a CSV yet)
  try {
    const res = await fetch('data/lessons.json');
    if (!res.ok) throw new Error('No manifest');
    const data = await res.json();
    availableLessons = Array.isArray(data.lessons) ? data.lessons : [];
    populateLessonSelect();
  } catch (err) {
    console.warn('Falling back to empty lesson list', err);
    availableLessons = [];
    populateLessonSelect();
  }
}

function attachEventListeners() {
  els.targetSelect.addEventListener('change', async () => {
    state.targetLang = els.targetSelect.value;
    populateVoiceSelect();
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
      if (state.recording) {
        stopRecording();
      } else {
        startRecording();
      }
      return;
    }

    if (event.key === 'p' || event.key === 'P') {
      event.preventDefault();
      handlePlaybackClick(1);
    }
  });

  els.voiceSelect?.addEventListener('change', () => {
    voiceSelections[state.targetLang] = els.voiceSelect.value;
    playbackQueue.resetWarmup(getLangCode(state.targetLang));
    persistVoices();
    updatePlaybackWarnings();
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

  voiceSelections = getStoredVoices();
  populateVoiceSelect();
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
        return {
          surface,
          translations: tokenTranslations,
          transcription: tokenTranscription,
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
  if (!window.speechSynthesis) {
    normalizedVoices = [];
    return;
  }

  const voices = window.speechSynthesis.getVoices();
  markVoicesReady(voices);
  normalizedVoices = normalizeVoiceList(voices, getLangCode(state.targetLang));
  populateVoiceSelect();
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

  hideTooltips();
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

  resetSentenceState();
  els.feedback.textContent = '';
  els.transcript.textContent = '';

  const total = state.sentences.length;
  els.status.textContent =
    state.mode === 'custom'
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

function enterCustomMode(text) {
  if (!text) return;

  if (state.mode !== 'custom') {
    state.savedLessonState = {
      sentences: state.sentences.slice(),
      currentIndex: state.currentIndex,
      lessonId: state.lessonId,
    };
  }

  state.customSentence = text;
  state.mode = 'custom';
  state.lessonId = CUSTOM_LESSON_ID;
  state.currentIndex = 0;
  state.sentences = [
    {
      id: 'custom',
      unit: null,
      theme: 'Custom practice',
      title: 'Custom practice',
      sentenceNumber: 1,
      text,
      translations: { [state.baseLang]: text },
      tokens: [],
    },
  ];

  els.lessonSelect.value = CUSTOM_LESSON_ID;
  closeCustomModal();
  renderCurrentSentence();
  setStatus('Practicing your custom sentence.');
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

  const storedSelection = voiceSelections[state.targetLang];
  if (storedSelection && storedSelection !== AUTO_VOICE_VALUE) {
    const [storedLang, storedName] = storedSelection.split('|');
    const voice = voices.find((v) => v.lang === storedLang && v.name === storedName);
    if (voice) return voice;
  }

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
    let success = false;

    state.supportsTtsService = ttsAvailable;

    if (synthSupported) {
      const voiceResult = await speakWithVoice(item);
      success = voiceResult.success;

      if (!success && voiceResult.retry && item.attempt < MAX_RETRIES) {
        queue.unshift({ ...item, attempt: item.attempt + 1 });
        await wait(200);
        return;
      }
    }

    if (!success && ttsAvailable) {
      success = await speakWithService(item);
    }

    if (!success && synthSupported && !ttsAvailable) {
      state.supportsSpeechSynthesis = false;
      updateSpeechSynthesisState({ announce: true });
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

async function speakSentence(text, langCode, rate = 1.0) {
  if (!text) {
    setStatus('Nothing to play for this sentence.');
    return;
  }

  if (isMobileDevice() && !state.audioUnlocked) {
    await unlockAudioPlayback();
  }

  await playbackQueue.warmVoicesForLang(langCode);
  state.supportsTtsService = Boolean(getTtsBaseUrl());
  playbackQueue.enqueue({ text, langCode, rate });
  updatePlaybackWarnings();
}

function speakCurrent(rate = 1) {
  if (!state.sentences.length) return;
  const text = currentSentence().text;
  speakSentence(text, getLangCode(state.targetLang), rate);
}

function speakWord(text, rate = 1) {
  if (!text) return;
  speakSentence(text, getLangCode(state.targetLang), rate);
}

function getBestVoiceSync(langCode) {
  if (!isSpeechSynthesisSupported()) return null;
  const synth = window.speechSynthesis;
  if (!synth) return null;
  const voices = synth.getVoices();
  if (!voices || !voices.length) return null;

  const target = String(langCode || '').toLowerCase();
  const priorities = CURATED_VOICE_PRIORITIES[(target.split('-')[0] || target)] || [];

  // Prefer curated priorities first.
  for (const pref of priorities) {
    const match = voices.find((v) => String(v.lang || '').toLowerCase() === String(pref).toLowerCase());
    if (match) return match;
  }

  // Then any voice that matches language prefix.
  const prefix = target.split('-')[0];
  const byPrefix = voices.find((v) => String(v.lang || '').toLowerCase().startsWith(prefix));
  if (byPrefix) return byPrefix;

  return voices[0] || null;
}

function speakCurrentImmediate(rate = 1) {
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

  utterance.onerror = () => {
    // Fall back to the existing async pipeline (service TTS if needed).
    speakCurrent(rate);
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

  // On mobile, try a synchronous speechSynthesis play first.
  if (isMobileDevice()) {
    // Try to force voices to load.
    try {
      if (window.speechSynthesis) window.speechSynthesis.getVoices();
    } catch (_) {}

    const started = speakCurrentImmediate(rate);
    if (started) return;
  }

  speakCurrent(rate);
}

function initRecognition() {
  if (!state.supportsRecognition) return;
  if (state.recognition) {
    state.recognition.abort();
  }

  state.recognition = new SpeechRecognition();
  state.recognition.lang = getLangCode(state.targetLang);
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
    setStatus('Listening...');
    updateRecordState();
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
    clearRecognitionRestartTimer();
    state.recognition.lang = getLangCode(state.targetLang);
    state.recognition.start();
  } catch (err) {
    console.error('Failed to start recognition', err);
    setStatus('Could not start recording.');
  }
}

function stopRecording() {
  if (state.recording && state.recognition) {
    state.manualStopRequested = true;
    state.shouldAutoRestartRecognition = false;
    clearRecognitionRestartTimer();
    setStatus('Stopping...');
    state.recognition.stop();
  }
}

function updateRecordState() {
  els.record.disabled = !state.supportsRecognition || state.recording;
  els.stop.disabled = !state.recording;
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
  const token = normalizeWord(rawToken);
  if (!token) return '';
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

function collapseConsecutiveDuplicates(tokens, maxRun = 1) {
  const out = [];
  let prev = null;
  let run = 0;
  for (const t of tokens) {
    if (t === prev) {
      run += 1;
      if (run <= maxRun) out.push(t);
    } else {
      prev = t;
      run = 1;
      out.push(t);
    }
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
      : collapseConsecutiveDuplicates(spokenTokens, 1);

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
  },
};

function getApproxRules(langCode) {
  const baseRules =
    langCode && APPROX_RULES_BY_LANG[langCode]
      ? APPROX_RULES_BY_LANG[langCode]
      : APPROX_RULES_BY_LANG.default || {};
  const coverageThreshold = getApproxThresholdFromIndex(state.approxLevelIndex);
  return { ...baseRules, coverageThreshold };
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
  return orderedCharacterCoverage(target, candidate) >= coverageThreshold;
}

function similarityScore(rawTarget, rawCandidate, langCode) {
  const target = normalizeToken(rawTarget, langCode);
  const cand = normalizeToken(rawCandidate, langCode);
  if (!target || !cand) return 0;

  if (target === cand) return 1;

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

function findMatchesForTargetTokens(targetTokens, spokenTokens, { langCode } = {}) {
  const matches = new Array(targetTokens.length).fill(null);
  const usedUntil = { value: 0 };
  const targetLang = langCode || state.targetLang;

  const S = spokenTokens.length;
  const MAX_WINDOW = 3;

  for (let i = 0; i < targetTokens.length; i++) {
    const target = targetTokens[i];
    let best = null;

    for (let start = usedUntil.value; start < S; start++) {
      for (let end = start; end < Math.min(S, start + MAX_WINDOW); end++) {
        const windowTokens = spokenTokens.slice(start, end + 1);
        const concatenated = normalizeWord(windowTokens.join(''));
        const score = similarityScore(target, concatenated, targetLang);
        if (!best || score > best.score) {
          best = { start, end, score };
        }
      }
    }

    if (best && best.score >= 0.7) {
      matches[i] = best;
      usedUntil.value = best.end + 1;
    } else {
      matches[i] = null;
    }
  }

  return matches;
}

function resetSentenceState() {
  lastTranscript = '';
  wordStatus = targetTokens.map(() => 'pending');
  updateWordSpanClasses();
}

function updateWordSpanClasses() {
  wordSpans.forEach((span, i) => {
    span.classList.remove('word-correct', 'word-wrong', 'word-pending');
    const status = wordStatus[i] || 'pending';
    if (status === 'correct') span.classList.add('word-correct');
    else if (status === 'wrong') span.classList.add('word-wrong');
    else span.classList.add('word-pending');
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
    clearRecognitionRestartTimer();
    try {
      state.recognition.stop();
    } catch (e) {
      // ignore
    }
    const feedbackEl = document.getElementById('feedback');
    if (feedbackEl) {
      feedbackEl.textContent = 'Perfecte! 👏 Has pronunciat tota la frase correctament.';
    }
    setStatus('Recording stopped – sentence complete ✅');
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

  // Arabic-script path: segment the recognized blob by searching expected Arabic token surfaces,
  // then emit the aligned ma_latn token (transcription) for scoring.
  const recognized = normalizeArabicToken(raw);
  let pos = 0;
  const spokenTokens = [];

  tokens.forEach((t) => {
    const surface = normalizeArabicToken(t.surface || '');
    const latn = normalizeToken(t.transcription || '', 'ma');
    if (!surface || !latn) return;

    const idx = recognized.indexOf(surface, pos);
    if (idx !== -1) {
      spokenTokens.push(latn);
      pos = idx + surface.length;
    }
  });

  return { spokenTokens, spokenTranscript: spokenTokens.join(' ') };
}

function updateLiveFeedback(transcript, { isFinalResult = false } = {}) {
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
    targetTokens.slice(lockedPrefix),
    filteredTokens
  );

  for (let i = 0; i < matches.length; i++) {
    if (matches[i]) {
      wordStatus[i + lockedPrefix] = 'correct';
    }
  }

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
    }
  }

  for (let i = 0; i < n; i++) {
    const wasWrong = prevStatus[i] === 'wrong';
    const isNowCorrect = wordStatus[i] === 'correct';
    if (wasWrong && isNowCorrect) {
      for (let k = Math.max(i + 1, lockedPrefix); k < n; k++) {
        if (wordStatus[k] === 'correct') {
          wordStatus[k] = 'pending';
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

  const matches = findMatchesForTargetTokens(targetTokens, filteredTokens);
  const correct = matches.filter(Boolean).length;

  const score = Math.round((correct / (targetTokens.length || 1)) * 100);
  const feedbackEl = els.feedback;
  if (!feedbackEl) return;

  if (score >= 80) {
    feedbackEl.textContent = `Great! Score: ${score}%`;
  } else if (score >= 50) {
    feedbackEl.textContent = `Good effort! Score: ${score}%. Try to fix the red words.`;
  } else {
    feedbackEl.textContent = `Let's try again. Score: ${score}%. Focus on the first few words.`;
  }

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
  };
}

