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

const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition;

let wordSpans = [];
let targetTokens = [];
let lastTranscript = '';
let currentSentenceText = '';
let wordStatus = [];
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

document.addEventListener('DOMContentLoaded', async () => {
  cacheElements();
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

function hydrateSelections() {
  populateSelect(els.targetSelect, TARGET_LANGS, state.targetLang);
  populateSelect(els.baseSelect, BASE_LANGS, state.baseLang);
  populateLessonSelect();
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
  try {
    const res = await fetch('data/lessons.json');
    if (!res.ok) throw new Error('No manifest');
    const data = await res.json();
    availableLessons = Array.isArray(data.lessons) ? data.lessons : [];
  } catch (err) {
    console.warn('Falling back to default lesson list', err);
    availableLessons = [];
  }

  populateLessonSelect();
}

function attachEventListeners() {
  els.targetSelect.addEventListener('change', () => {
    state.targetLang = els.targetSelect.value;
    populateLessonSelect();
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
  const path = `data/${state.lessonId}.json`;
  setStatus(`Loading lesson ${path}...`);
  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`Unable to load ${path}`);
    const data = await res.json();
    state.sentences = data.sentences || [];
    if (!state.sentences.length) throw new Error('No sentences in lesson.');
    const saved = loadProgressForLesson();
    state.currentIndex = saved?.currentIndex || 0;
    renderCurrentSentence();
    setStatus(`Loaded ${data.lang.toUpperCase()} • ${data.level} • ${data.theme}`);
  } catch (err) {
    console.error(err);
    setStatus('Could not load lesson. Please ensure the JSON exists.');
    state.sentences = [];
    els.sentence.textContent = '';
  }
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

function renderCurrentSentence() {
  if (!state.sentences.length) return;
  const sentence = currentSentence();
  const sentenceEl = els.sentence;
  sentenceEl.innerHTML = '';
  wordSpans = [];
  currentSentenceText = sentence.text;
  targetTokens = tokenizeText(currentSentenceText);

  sentence.tokens.forEach((tokenObj) => {
    const span = document.createElement('span');
    const spokenWord = tokenObj.surface || tokenObj.text || '';
    span.textContent = `${spokenWord} `;
    span.classList.add('word', 'word-pending');
    span.dataset.word = spokenWord;

    const translation = tokenObj.translations?.[state.baseLang];
    if (translation) {
      span.setAttribute('aria-label', translation);
      span.dataset.translation = translation;
    }

    if (tokenObj.latin) {
      span.dataset.latin = tokenObj.latin;
    }

    wordSpans.push(span);
    sentenceEl.appendChild(span);
  });
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

function speakSentence(text, langCode, rate = 1.0) {
  const u = new SpeechSynthesisUtterance(text);
  u.lang = langCode;
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
