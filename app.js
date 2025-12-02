const TARGET_LANGS = [
  { code: 'fr', label: 'French' },
  { code: 'en', label: 'English' },
  { code: 'ca', label: 'Catalan' },
  { code: 'it', label: 'Italian' },
  { code: 'ar', label: 'Arabic (Darija friendly)' }
];

const BASE_LANGS = [
  { code: 'en', label: 'English' },
  { code: 'es', label: 'Spanish' },
  { code: 'ca', label: 'Catalan' },
  { code: 'fr', label: 'French' },
  { code: 'it', label: 'Italian' }
];

const STORAGE_KEY = 'speechReadingProgress';
const DEFAULT_LESSON_SUFFIX = 'a1_introductions';

const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition;

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

document.addEventListener('DOMContentLoaded', () => {
  cacheElements();
  hydrateSelections();
  attachEventListeners();
  hydrateFromStorage();
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

function populateLessonSelect() {
  const option = document.createElement('option');
  option.value = `${state.targetLang}_${DEFAULT_LESSON_SUFFIX}`;
  option.textContent = `A1 introductions (${state.targetLang})`;
  els.lessonSelect.innerHTML = '';
  els.lessonSelect.appendChild(option);
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
      e.target.classList.toggle('active');
    }
  });
}

function hydrateFromStorage() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  try {
    const saved = JSON.parse(raw);
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
  state.lessonId = `${state.targetLang}_${DEFAULT_LESSON_SUFFIX}`;
  els.lessonSelect.value = state.lessonId;
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
  els.sentence.innerHTML = '';
  sentence.tokens.forEach((token) => {
    const span = document.createElement('span');
    span.textContent = token.surface + ' ';
    span.className = 'word';
    const translation = token.translations?.[state.baseLang] || '—';
    span.dataset.translation = translation;
    els.sentence.appendChild(span);
  });
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
    case 'ar':
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

function initRecognition() {
  if (!state.supportsRecognition) return;
  state.recognition = new SpeechRecognition();
  state.recognition.lang = getLangCode(state.targetLang);
  state.recognition.interimResults = false;
  state.recognition.maxAlternatives = 1;
  state.recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    els.transcript.textContent = `You said: ${transcript}`;
    handleUserTranscript(transcript);
    state.recording = false;
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
    state.recognition.start();
    state.recording = true;
    setStatus('Listening...');
    updateRecordState();
  } catch (err) {
    console.error('Failed to start recognition', err);
    setStatus('Could not start recording.');
  }
}

function stopRecording() {
  if (state.recording && state.recognition) {
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

function normalizeText(t) {
  return t
    .toLowerCase()
    .normalize('NFD')
    .replace(/\p{Diacritic}+/gu, '')
    .replace(/[.,!?;:]/g, '')
    .trim();
}

function tokenize(t) {
  return normalizeText(t)
    .split(/\s+/)
    .filter(Boolean);
}

function levenshtein(a, b) {
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }
  return dp[m][n];
}

function compareTokens(targetTokens, userTokens) {
  const statuses = [];
  targetTokens.forEach((target, idx) => {
    const user = userTokens[idx];
    if (!user) {
      statuses.push('miss');
      return;
    }
    if (target === user) {
      statuses.push('ok');
      return;
    }
    const distance = levenshtein(target, user);
    statuses.push(distance <= 1 ? 'approx' : 'miss');
  });
  const correctCount = statuses.filter((s) => s === 'ok').length;
  const approxCount = statuses.filter((s) => s === 'approx').length;
  const score = Math.round(((correctCount + approxCount * 0.5) / targetTokens.length) * 100);
  return { statuses, score };
}

function handleUserTranscript(transcript) {
  const targetTokens = tokenize(currentSentence().text);
  const userTokens = tokenize(transcript);
  const { statuses, score } = compareTokens(targetTokens, userTokens);
  colorizeWords(statuses);
  const message = score >= 80
    ? 'Great!'
    : score >= 50
      ? 'Good, a few words need work.'
      : "Let\'s try again. Use the slow button if needed.";
  els.feedback.textContent = `Score: ${score}%. ${message}`;
  saveProgress(score);
}

function colorizeWords(statuses) {
  const spans = [...els.sentence.querySelectorAll('.word')];
  spans.forEach((span, idx) => {
    span.classList.remove('word-ok', 'word-approx', 'word-miss');
    const status = statuses[idx];
    if (status === 'ok') span.classList.add('word-ok');
    if (status === 'approx') span.classList.add('word-approx');
    if (status === 'miss') span.classList.add('word-miss');
  });
}

function setStatus(text) {
  els.status.textContent = text;
}
