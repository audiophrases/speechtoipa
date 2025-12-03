const TARGET_LANGS = [
  { code: 'fr', label: 'French' },
  { code: 'en', label: 'English' },
  { code: 'ca', label: 'Catalan' },
  { code: 'it', label: 'Italian' },
  { code: 'ma', label: 'Moroccan Darija' }
];

const BASE_LANGS = [
  { code: 'en', label: 'English' },
  { code: 'es', label: 'Spanish' },
  { code: 'ca', label: 'Catalan' },
  { code: 'fr', label: 'French' },
  { code: 'it', label: 'Italian' },
  { code: 'ma', label: 'Moroccan Darija' }
];

const STORAGE_KEY = 'speechReadingProgress';
const DEFAULT_LESSON_SUFFIX = 'a1_introductions';
let availableLessons = [];

const MASTER_CSV_URLS = {
  ca: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=1216373156&single=true&output=csv',
  en: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=1053057720&single=true&output=csv',
  fr: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=484976070&single=true&output=csv',
  it: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=1338439854&single=true&output=csv',
  ma: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=710375040&single=true&output=csv',
};

const MASTER_ROWS_BY_LANG = {};
const TRANSLATION_LANG_CODES = ['ca', 'es', 'en', 'fr', 'it', 'ma'];

const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition;

if (window.speechSynthesis) {
  window.speechSynthesis.onvoiceschanged = () => {
    window.speechSynthesis.getVoices();
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
const state = {
  targetLang: 'fr',
  baseLang: 'en',
  lessonId: '',
  sentences: [],
  currentIndex: 0,
  recognition: null,
  recording: false,
  supportsRecognition: Boolean(SpeechRecognition),
};

const els = {};

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

document.addEventListener('DOMContentLoaded', async () => {
  cacheElements();
  createTooltips();
  hydrateSelections();
  attachEventListeners();
  hydrateFromStorage();
  await loadLessonManifest();
  updateLessonId();
  loadLesson();
});

function cacheElements() {
  els.targetSelect = document.getElementById('target-lang');
  els.baseSelect = document.getElementById('base-lang');
  els.lessonSelect = document.getElementById('lesson-select');
  els.sentence = document.getElementById('sentence');
  els.play = document.getElementById('play-btn');
  els.slow = document.getElementById('slow-btn');
  els.record = document.getElementById('record-btn');
  els.stop = document.getElementById('stop-btn');
  els.next = document.getElementById('next-btn');
  els.status = document.getElementById('status');
  els.transcript = document.getElementById('transcript');
  els.feedback = document.getElementById('feedback');
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

function populateLessonSelect() {
  const lessons = getLessonsForLang(state.targetLang);
  els.lessonSelect.innerHTML = '';
  lessons.forEach((lesson) => {
    const option = document.createElement('option');
    option.value = lesson.id;
    option.textContent = `${lesson.label} (${lesson.lang})`;
    els.lessonSelect.appendChild(option);
  });

  const hasSelection = lessons.some((lesson) => lesson.id === state.lessonId);
  state.lessonId = hasSelection ? state.lessonId : lessons[0].id;
  els.lessonSelect.value = state.lessonId;
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
    await loadLessonManifest();
    updateLessonId();
    saveProgress();
    loadLesson();
  });

  els.baseSelect.addEventListener('change', () => {
    state.baseLang = els.baseSelect.value;
    saveProgress();
    renderCurrentSentence();
  });

  els.lessonSelect.addEventListener('change', () => {
    state.lessonId = els.lessonSelect.value;
    state.currentIndex = 0;
    saveProgress();
    loadLesson();
  });

  els.play.addEventListener('click', () => speakCurrent(1));
  els.slow.addEventListener('click', () => speakCurrent(0.7));
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
}

function hydrateFromStorage() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  try {
    const saved = normalizeLegacyCodes(JSON.parse(raw));
    if (saved.targetLang) state.targetLang = saved.targetLang;
    if (saved.baseLang) state.baseLang = saved.baseLang;
    if (saved.lessonId) state.lessonId = saved.lessonId;
    if (saved.progress && saved.progress[state.lessonId]) {
      state.currentIndex = saved.progress[state.lessonId].currentIndex || 0;
    }
    populateSelect(els.targetSelect, TARGET_LANGS, state.targetLang);
    populateSelect(els.baseSelect, BASE_LANGS, state.baseLang);
    populateLessonSelect();
  } catch (err) {
    console.error('Failed to parse saved progress', err);
  }
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
  const raw = localStorage.getItem(STORAGE_KEY);
  let data = { progress: {} };
  try {
    data = raw ? JSON.parse(raw) : { progress: {} };
  } catch {
    data = { progress: {} };
  }
  data.targetLang = state.targetLang;
  data.baseLang = state.baseLang;
  data.lessonId = state.lessonId;
  data.progress = data.progress || {};
  data.progress[state.lessonId] = data.progress[state.lessonId] || { currentIndex: 0, scores: {} };
  data.progress[state.lessonId].currentIndex = state.currentIndex;
  if (bestScore !== undefined) {
    const previous = data.progress[state.lessonId].scores?.[currentSentence().id] || 0;
    if (!data.progress[state.lessonId].scores) data.progress[state.lessonId].scores = {};
    data.progress[state.lessonId].scores[currentSentence().id] = Math.max(previous, bestScore);
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

function updateLessonId() {
  state.lessonId = els.lessonSelect.value;
  initRecognition();
}

async function loadLesson() {
  const lang = state.targetLang;
  const lessonId = state.lessonId;
  if (!lessonId) return;

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
        return {
          surface,
          translations: tokenTranslations,
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
      tokens,
    };
  });

  state.sentences = sentences;
  const saved = loadProgressForLesson();
  state.currentIndex = saved?.currentIndex || 0;

  renderCurrentSentence();
  const lessonMeta = availableLessons.find((l) => l.id === lessonId) || {};
  setStatus(
    `Loaded ${lessonMeta.lang?.toUpperCase() || ''} • ${lessonMeta.label || lessonId}`
  );
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
  if (!wordTrans) {
    wordTooltipEl.style.visibility = 'hidden';
    return;
  }
  wordTooltipEl.textContent = wordTrans;
  wordTooltipEl.style.visibility = 'visible';
  positionTooltips(x, y);
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

  if (!sentenceTrans || !wordTrans || sentenceTrans === wordTrans) {
    return;
  }

  sentenceTooltipTimer = setTimeout(() => {
    if (currentTooltipTarget !== span) return;

    sentenceTooltipEl.textContent = sentenceTrans;
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
  sentenceEl.dir = isRTL ? 'rtl' : 'ltr';
  sentenceEl.classList.toggle('rtl-sentence', isRTL);

  const fullText = sentence.text || '';
  currentSentenceText = fullText;

  const hasTokens = Array.isArray(sentence.tokens) && sentence.tokens.length > 0;

  if (hasTokens) {
    const tokensForMatching = sentence.tokens.map((tokenObj) => ({
      surface: tokenObj.surface || '',
      translations: tokenObj.translations || {},
    }));
    targetTokens = tokenizeText(fullText);

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

      if (wordTrans) {
        span.dataset.wordTranslation = wordTrans;
      }
      if (sentenceTrans) {
        span.dataset.sentenceTranslation = sentenceTrans;
      }

      if (wordTrans) {
        span.setAttribute('aria-label', wordTrans);
      } else if (sentenceTrans) {
        span.setAttribute('aria-label', sentenceTrans);
      }

      wordSpans.push(span);
      sentenceEl.appendChild(span);

      pos = index + word.length;
    });

    const tail = fullText.slice(pos);
    if (tail) {
      sentenceEl.appendChild(document.createTextNode(tail));
    }
  } else {
    const rawTokens = fullText.split(/\s+/).filter(Boolean);
    targetTokens = rawTokens.map((w) => normalizeWord(w));

    rawTokens.forEach((word) => {
      const span = document.createElement('span');
      span.textContent = word + ' ';
      span.classList.add('word', 'word-pending');
      span.dataset.word = word;

      const sentenceTrans = sentence.translations?.[state.baseLang] || null;
      if (sentenceTrans) {
        span.dataset.sentenceTranslation = sentenceTrans;
        span.setAttribute('aria-label', sentenceTrans);
      }

      wordSpans.push(span);
      sentenceEl.appendChild(span);
    });
  }

  resetSentenceState();
  els.feedback.textContent = '';
  els.transcript.textContent = '';
  els.status.textContent = `Sentence ${state.currentIndex + 1} / ${state.sentences.length}`;
}

function currentSentence() {
  return state.sentences[state.currentIndex];
}

function goToNext() {
  if (!state.sentences.length) return;
  state.currentIndex = (state.currentIndex + 1) % state.sentences.length;
  renderCurrentSentence();
  saveProgress();
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
      return 'ar-MA';
    case 'ary':
      return 'ar-MA';
    default:
      return 'en-US';
  }
}

function getVoiceForLang(langCode) {
  if (!window.speechSynthesis) return null;
  const voices = window.speechSynthesis.getVoices();
  if (!voices || !voices.length) return null;

  // 1) Exact match (e.g. 'ar-MA')
  let voice = voices.find((v) => v.lang === langCode);
  if (voice) return voice;

  // 2) Match by base language (e.g. 'ar' from 'ar-MA')
  const base = langCode.split('-')[0];
  voice = voices.find((v) => v.lang.toLowerCase().startsWith(base.toLowerCase()));
  if (voice) return voice;

  // 3) Fallback: any Arabic voice if we’re using Darija
  if (base === 'ar') {
    voice = voices.find((v) => v.lang.toLowerCase().startsWith('ar'));
    if (voice) return voice;
  }

  // 4) General fallback – first English or first voice
  voice = voices.find((v) => v.lang.toLowerCase().startsWith('en'));
  return voice || voices[0];
}

function speakSentence(text, langCode, rate = 1.0) {
  const u = new SpeechSynthesisUtterance(text);
  u.lang = langCode;

  const voice = getVoiceForLang(langCode);
  if (voice) {
    u.voice = voice;
  }

  u.rate = rate;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
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

  state.recognition.onresult = (event) => {
    let transcript = '';

    for (let i = 0; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript + ' ';
    }

    transcript = transcript.trim();
    lastTranscript = transcript;
    updateLiveFeedback(transcript);
  };

  state.recognition.onstart = () => {
    state.recording = true;
    setStatus('Listening...');
    updateRecordState();
  };

  state.recognition.onerror = (event) => {
    console.error('Recognition error', event.error);
    setStatus(`Recognition error: ${event.error}`);
    state.recording = false;
    updateRecordState();
  };

  state.recognition.onend = () => {
    state.recording = false;
    updateRecordState();
    if (lastTranscript !== null) {
      finalizeScore(lastTranscript);
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
    lastTranscript = '';
    els.transcript.textContent = '';
    els.feedback.textContent = '';
    resetSentenceState();
    state.recognition.lang = getLangCode(state.targetLang);
    state.recognition.start();
  } catch (err) {
    console.error('Failed to start recognition', err);
    setStatus('Could not start recording.');
  }
}

function stopRecording() {
  if (state.recording && state.recognition) {
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

function normalizeWord(w) {
  if (!w) return '';
  return w
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[.,!?;:;()"«»¿¡]/g, '')
    .trim();
}

function tokenizeText(t) {
  return normalizeWord(t)
    .split(/\s+/)
    .filter(Boolean);
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
};

function similarityScore(rawTarget, rawCandidate) {
  const target = normalizeWord(rawTarget);
  const cand = normalizeWord(rawCandidate);
  if (!target || !cand) return 0;

  if (target === cand) return 1;

  const equiv = EQUIV_GROUPS[target];
  if (equiv && equiv.includes(cand)) {
    return 0.95;
  }

  const dist = levenshtein(target, cand);
  const maxLen = Math.max(target.length, cand.length);
  const normDist = maxLen === 0 ? 0 : dist / maxLen;
  return 1 - normDist;
}

function findMatchesForTargetTokens(targetTokens, spokenTokens) {
  const matches = new Array(targetTokens.length).fill(null);
  const usedUntil = { value: 0 };

  const S = spokenTokens.length;
  const MAX_WINDOW = 3;

  for (let i = 0; i < targetTokens.length; i++) {
    const target = targetTokens[i];
    let best = null;

    for (let start = usedUntil.value; start < S; start++) {
      for (let end = start; end < Math.min(S, start + MAX_WINDOW); end++) {
        const windowTokens = spokenTokens.slice(start, end + 1);
        const concatenated = normalizeWord(windowTokens.join(''));
        const score = similarityScore(target, concatenated);
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

function checkIfSentenceCompleteAndStop() {
  if (!wordStatus.length) return;
  const allCorrect = wordStatus.every((s) => s === 'correct');
  if (allCorrect && state.recognition) {
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

function updateLiveFeedback(transcript) {
  const spokenTokens = tokenizeText(transcript);
  const matches = findMatchesForTargetTokens(targetTokens, spokenTokens);
  const prevStatus = [...wordStatus];
  const n = targetTokens.length;

  wordStatus = targetTokens.map(() => 'pending');

  for (let i = 0; i < n; i++) {
    if (matches[i]) {
      wordStatus[i] = 'correct';
    }
  }

  let firstNotCorrect = -1;
  for (let i = 0; i < n; i++) {
    if (wordStatus[i] !== 'correct') {
      firstNotCorrect = i;
      break;
    }
  }
  if (firstNotCorrect !== -1) {
    wordStatus[firstNotCorrect] = 'wrong';
  }

  for (let i = 0; i < n; i++) {
    const wasWrong = prevStatus[i] === 'wrong';
    const isNowCorrect = wordStatus[i] === 'correct';
    if (wasWrong && isNowCorrect) {
      for (let k = i + 1; k < n; k++) {
        if (wordStatus[k] === 'correct') {
          wordStatus[k] = 'pending';
        }
      }
      break;
    }
  }

  updateWordSpanClasses();

  if (els.transcript) {
    els.transcript.textContent = transcript;
  }

  checkIfSentenceCompleteAndStop();
}

function finalizeScore(transcript) {
  const spokenTokens = tokenizeText(transcript);
  const matches = findMatchesForTargetTokens(targetTokens, spokenTokens);
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
