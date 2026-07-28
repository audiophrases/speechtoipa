// The teacher's side of Speech to IPA: sign in, build a reading assignment
// from a lesson (or your own sentences), and read what the class did.
//
// A separate page rather than a mode of the app, for the same reason Dictation
// Time splits them: a Chromebook opening a student link downloads none of this,
// and nothing behind the password is in the student bundle at all.
//
// One screen is visible at a time. There is no router — the app is served from
// a static host with no rewrites — so "navigation" is showing one section and
// hiding the rest, and the browser's Back button is deliberately not involved.
(function teacherPage() {
  'use strict';

  var A = window.Assignments;

  var LANGS = [
    { code: 'en', label: 'English' },
    { code: 'fr', label: 'French' },
    { code: 'ca', label: 'Catalan' },
    { code: 'ma', label: 'Moroccan Darija' },
    { code: 'it', label: 'Italian' },
  ];

  var screens = {};
  var el = {};
  // What Back returns to, and what a delete/close should refresh.
  var currentCode = '';

  function $(id) {
    return document.getElementById(id);
  }

  function show(name) {
    Object.keys(screens).forEach(function each(key) {
      screens[key].classList.toggle('hidden', key !== name);
    });
    window.scrollTo(0, 0);
  }

  function setText(node, text) {
    if (node) node.textContent = text || '';
  }

  function formatDate(value) {
    if (!value) return '';
    try {
      return new Date(value).toLocaleString(undefined, {
        day: 'numeric',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch (err) {
      return '';
    }
  }

  function scoreClass(score) {
    if (typeof score !== 'number' || Number.isNaN(score)) return 'score-none';
    if (score >= 80) return 'score-high';
    if (score >= 50) return 'score-mid';
    return 'score-low';
  }

  function copyToClipboard(value, button) {
    var original = button.textContent;
    function done(ok) {
      button.textContent = ok ? 'Copied' : 'Press Ctrl+C';
      setTimeout(function restore() {
        button.textContent = original;
      }, 1600);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(value).then(
        function ok() {
          done(true);
        },
        function fail() {
          done(false);
        }
      );
      return;
    }
    done(false);
  }

  // ------------------------------------------------------------------ sign in

  async function signIn(event) {
    event.preventDefault();
    var password = el.password.value || '';
    if (!password) return;

    setText(el.loginError, '');
    el.loginSubmit.disabled = true;
    el.loginSubmit.textContent = 'Checking…';

    try {
      await A.verifyTeacherPassword(password);
    } catch (err) {
      setText(el.loginError, err.message || 'Could not sign in.');
      el.loginSubmit.disabled = false;
      el.loginSubmit.textContent = 'Sign in';
      el.password.select();
      return;
    }

    A.setTeacherPassword(password);
    el.password.value = '';
    el.loginSubmit.disabled = false;
    el.loginSubmit.textContent = 'Sign in';
    await openDashboard();
  }

  // ---------------------------------------------------------------- dashboard

  async function openDashboard() {
    show('dashboard');
    currentCode = '';
    setText(el.dashboardStatus, 'Loading…');
    el.list.innerHTML = '';

    var data;
    try {
      data = await A.listAssignments();
    } catch (err) {
      setText(el.dashboardStatus, err.message || 'Could not load your assignments.');
      return;
    }

    var assignments = data.assignments || [];
    setText(el.dashboardStatus, '');

    if (!assignments.length) {
      el.list.innerHTML = '<p class="empty">No assignments yet. Create one to get started.</p>';
      return;
    }

    assignments.forEach(function addRow(assignment) {
      var bits = [assignment.code];
      if (assignment.className) bits.push(assignment.className);
      bits.push(assignment.sentenceCount + ' sentences');
      if (assignment.status !== 'active') bits.push(assignment.status);
      if (assignment.dueAt) bits.push('due ' + formatDate(assignment.dueAt));

      var row = document.createElement('button');
      row.type = 'button';
      row.className = 'teacher-row';
      row.innerHTML =
        '<span class="teacher-row-main">' +
        '<p class="teacher-row-title"></p>' +
        '<p class="teacher-row-meta"></p>' +
        '</span>';
      row.querySelector('.teacher-row-title').textContent = assignment.title || 'Untitled';
      row.querySelector('.teacher-row-meta').textContent = bits.join(' • ');
      row.addEventListener('click', function open() {
        openAssignment(assignment.code);
      });
      el.list.appendChild(row);
    });
  }

  // ------------------------------------------------------------------- create

  function openCreate() {
    show('create');
    setText(el.createError, '');
    el.createForm.reset();
    el.createLang.value = 'en';
    updateSentenceCount();
    loadLessonsForLang('en');
  }

  async function loadLessonsForLang(lang) {
    el.createLesson.innerHTML = '<option value="">Loading lessons…</option>';
    var rows;
    try {
      rows = await ensureMasterRowsForLang(lang);
    } catch (err) {
      rows = null;
    }

    el.createLesson.innerHTML = '';
    var blank = document.createElement('option');
    blank.value = '';
    blank.textContent = rows && rows.length ? 'Type my own sentences' : 'Lessons unavailable';
    el.createLesson.appendChild(blank);

    lessonsFromRows(rows).forEach(function addOption(lesson) {
      var option = document.createElement('option');
      option.value = lesson.id;
      option.textContent = lesson.label;
      el.createLesson.appendChild(option);
    });
  }

  // Fills the textarea with the lesson's sentences, one per line — which is
  // also the format for typing your own, so the two paths converge here and the
  // teacher can edit either before creating.
  async function fillFromLesson(lang, lessonId) {
    if (!lessonId) return;
    var rows = await ensureMasterRowsForLang(lang);
    if (!rows) return;

    var seen = {};
    var sentences = [];
    var title = '';
    rows.forEach(function collect(row) {
      // token_id empty = the sentence row; the rest are its words.
      if (row.lesson_id !== lessonId || row.token_id) return;
      var text = (row[lang] || '').trim();
      if (!text || seen[row.sentence_id]) return;
      seen[row.sentence_id] = true;
      sentences.push(text);
      if (!title) title = row.lesson_title || '';
    });

    el.createText.value = sentences.join('\n');
    if (!el.createTitle.value && title) el.createTitle.value = title;
    updateSentenceCount();
  }

  function sentencesFromTextarea() {
    return (el.createText.value || '')
      .split(/\r?\n/)
      .map(function trim(line) {
        return line.replace(/\s+/g, ' ').trim();
      })
      .filter(Boolean);
  }

  function updateSentenceCount() {
    var count = sentencesFromTextarea().length;
    setText(
      el.createCount,
      count === 0
        ? 'Each line becomes one sentence to read aloud.'
        : count + (count === 1 ? ' sentence' : ' sentences') + ' to read aloud.'
    );
  }

  async function submitCreate(event) {
    event.preventDefault();
    var sentences = sentencesFromTextarea();
    setText(el.createError, '');

    if (!sentences.length) {
      setText(el.createError, 'Add at least one sentence.');
      return;
    }

    el.createSubmit.disabled = true;
    el.createSubmit.textContent = 'Creating…';

    var lessonId = el.createLesson.value;
    var due = el.createDue.value ? new Date(el.createDue.value).getTime() : 0;

    var data;
    try {
      data = await A.createAssignment({
        title: el.createTitle.value,
        className: el.createClass.value,
        sentences: sentences,
        lang: el.createLang.value,
        accuracyIndex: Number(el.createAccuracy.value),
        attemptsLimit: Number(el.createAttempts.value),
        feedbackMode: el.createFeedback.value,
        dueAt: due,
        // Lets the student app fetch this lesson's per-word data — the
        // translations behind the tooltips, and the transcriptions Darija is
        // scored against. Absent for hand-typed sentences, which behave like
        // custom text.
        source: lessonId ? { lessonId: lessonId } : null,
      });
    } catch (err) {
      setText(el.createError, err.message || 'Could not create the assignment.');
      el.createSubmit.disabled = false;
      el.createSubmit.textContent = 'Create assignment';
      return;
    }

    el.createSubmit.disabled = false;
    el.createSubmit.textContent = 'Create assignment';

    // No audio to record, so unlike a dictation this is ready immediately.
    setText(el.createdCode, data.code);
    el.createdLink.value = A.studentLink(data.code);
    show('created');
  }

  // ------------------------------------------------------------------- detail

  async function openAssignment(code) {
    currentCode = code;
    show('detail');
    setText(el.detailTitle, 'Loading…');
    setText(el.detailMeta, '');
    el.detailSentences.innerHTML = '';
    el.detailAttempts.innerHTML = '';

    // Close and Delete get their handlers below, once the assignment's current
    // status is known. Until then they are disabled rather than merely inert:
    // a button that looks ready and silently does nothing reads as a bug, and
    // on a slow connection the window is long enough to hit.
    el.detailArchive.disabled = true;
    el.detailDelete.disabled = true;

    var data;
    try {
      data = await A.getAssignment(code);
    } catch (err) {
      setText(el.detailTitle, 'Could not open this assignment');
      setText(el.detailMeta, err.message || '');
      return;
    }

    el.detailArchive.disabled = false;
    el.detailDelete.disabled = false;

    var record = data.record || {};
    setText(el.detailTitle, record.title || 'Assignment');

    var bits = [code];
    if (record.className) bits.push(record.className);
    bits.push(record.status);
    if (record.dueAt) bits.push('due ' + formatDate(record.dueAt));
    if (record.settings && typeof record.settings.accuracyIndex === 'number') {
      bits.push('accuracy ' + [50, 60, 70, 80, 90, 100][record.settings.accuracyIndex] + '%');
    }
    setText(el.detailMeta, bits.join(' • '));

    el.detailLink.value = A.studentLink(code);
    el.detailArchive.textContent =
      record.status === 'archived' ? 'Reopen assignment' : 'Close assignment';
    el.detailArchive.onclick = function toggle() {
      setStatusFor(code, record.status === 'archived' ? 'active' : 'archived');
    };
    el.detailDelete.onclick = function remove() {
      deleteAssignment(code, record.title);
    };

    (data.sentences || []).forEach(function addSentence(text) {
      var item = document.createElement('li');
      item.textContent = text;
      el.detailSentences.appendChild(item);
    });

    var attempts = data.attempts || [];
    if (!attempts.length) {
      el.detailAttempts.innerHTML = '<p class="empty">Nobody has started this yet.</p>';
      return;
    }

    attempts.forEach(function addAttempt(attempt) {
      var meta = [];
      meta.push(attempt.submitted ? 'handed in' : 'in progress');
      if (attempt.updatedAt) meta.push(formatDate(attempt.updatedAt));

      var row = document.createElement('button');
      row.type = 'button';
      row.className = 'teacher-row';
      row.innerHTML =
        '<span class="teacher-row-main">' +
        '<p class="teacher-row-title"></p>' +
        '<p class="teacher-row-meta"></p>' +
        '</span><span class="teacher-row-score"></span>';
      row.querySelector('.teacher-row-title').textContent = attempt.studentName || 'Unknown';
      row.querySelector('.teacher-row-meta').textContent = meta.join(' • ');

      var score = row.querySelector('.teacher-row-score');
      score.textContent =
        typeof attempt.scorePercent === 'number' ? attempt.scorePercent + '%' : '—';
      score.classList.add(scoreClass(attempt.scorePercent));

      row.addEventListener('click', function open() {
        openAttempt(code, attempt);
      });
      el.detailAttempts.appendChild(row);
    });
  }

  async function setStatusFor(code, status) {
    try {
      await A.setAssignmentStatus(code, status);
    } catch (err) {
      alert(err.message || 'Could not change the assignment.');
      return;
    }
    openAssignment(code);
  }

  async function deleteAssignment(code, title) {
    if (
      !confirm(
        'Delete "' + (title || code) + '"?\n\nThis also deletes every attempt. It cannot be undone.'
      )
    ) {
      return;
    }
    try {
      await A.deleteAssignment(code);
    } catch (err) {
      alert(err.message || 'Could not delete the assignment.');
      return;
    }
    openDashboard();
  }

  // ------------------------------------------------------------------ attempt

  async function openAttempt(code, summary) {
    show('attempt');
    setText(el.attemptTitle, summary.studentName || 'Attempt');
    setText(el.attemptMeta, 'Loading…');
    el.attemptSentences.innerHTML = '';

    var data;
    try {
      data = await A.getAttempt(code, summary.studentKey, summary.id);
    } catch (err) {
      setText(el.attemptMeta, err.message || 'Could not open this attempt.');
      return;
    }

    var attempt = data.attempt || {};
    var results = data.results || {};
    var bits = [];
    if (typeof results.scorePercent === 'number') bits.push(results.scorePercent + '% overall');
    bits.push(attempt.submitted ? 'handed in ' + formatDate(attempt.submittedAt) : 'in progress');
    if (attempt.warnings) bits.push(attempt.warnings + ' warnings');
    setText(el.attemptMeta, bits.join(' • '));

    var sentences = data.sentences || [];
    var perSentence = results.perSentence || [];

    sentences.forEach(function addSentence(text, index) {
      var entry = perSentence[index] || {};
      var block = document.createElement('div');
      block.className = 'attempt-sentence';
      block.innerHTML =
        '<div class="attempt-sentence-head">' +
        '<p class="attempt-target"></p>' +
        '<span class="teacher-row-score"></span>' +
        '</div><p class="attempt-heard">Heard: <span></span></p>';

      block.querySelector('.attempt-target').textContent = text;

      var score = block.querySelector('.teacher-row-score');
      score.textContent = typeof entry.score === 'number' ? entry.score + '%' : 'not read';
      score.classList.add(scoreClass(entry.score));

      // The recognizer's transcript is the evidence behind the mark — the only
      // way to tell a misread word from a mishearing.
      block.querySelector('.attempt-heard span').textContent = entry.transcript || '—';
      if (!entry.transcript) block.querySelector('.attempt-heard').classList.add('empty');

      el.attemptSentences.appendChild(block);
    });
  }

  // --------------------------------------------------------------------- init

  document.addEventListener('DOMContentLoaded', function start() {
    screens.login = $('login');
    screens.dashboard = $('dashboard');
    screens.create = $('create');
    screens.created = $('created');
    screens.detail = $('detail');
    screens.attempt = $('attempt');

    el.password = $('teacher-password');
    el.loginSubmit = $('login-submit');
    el.loginError = $('login-error');
    el.dashboardStatus = $('dashboard-status');
    el.list = $('assignment-list');
    el.createForm = $('create-form');
    el.createTitle = $('create-title');
    el.createClass = $('create-class');
    el.createLang = $('create-lang');
    el.createLesson = $('create-lesson');
    el.createDue = $('create-due');
    el.createAccuracy = $('create-accuracy');
    el.createAttempts = $('create-attempts');
    el.createFeedback = $('create-feedback');
    el.createText = $('create-text');
    el.createCount = $('create-count');
    el.createError = $('create-error');
    el.createSubmit = $('create-submit');
    el.createdCode = $('created-code');
    el.createdLink = $('created-link');
    el.detailTitle = $('detail-title');
    el.detailMeta = $('detail-meta');
    el.detailLink = $('detail-link');
    el.detailArchive = $('detail-archive');
    el.detailDelete = $('detail-delete');
    el.detailSentences = $('detail-sentences');
    el.detailAttempts = $('detail-attempts');
    el.attemptTitle = $('attempt-title');
    el.attemptMeta = $('attempt-meta');
    el.attemptSentences = $('attempt-sentences');

    LANGS.forEach(function addLang(lang) {
      var option = document.createElement('option');
      option.value = lang.code;
      option.textContent = lang.label;
      el.createLang.appendChild(option);
    });

    $('login-form').addEventListener('submit', signIn);
    $('new-assignment').addEventListener('click', openCreate);
    el.createForm.addEventListener('submit', submitCreate);
    el.createText.addEventListener('input', updateSentenceCount);

    el.createLang.addEventListener('change', function langChanged() {
      el.createLesson.value = '';
      loadLessonsForLang(el.createLang.value);
    });
    el.createLesson.addEventListener('change', function lessonChanged() {
      fillFromLesson(el.createLang.value, el.createLesson.value);
    });

    $('copy-link').addEventListener('click', function copyCreated(event) {
      copyToClipboard(el.createdLink.value, event.currentTarget);
    });
    $('detail-copy').addEventListener('click', function copyDetail(event) {
      copyToClipboard(el.detailLink.value, event.currentTarget);
    });

    // Back always means "the list", except from an attempt, which means "the
    // assignment I was reading".
    Array.prototype.forEach.call(document.querySelectorAll('[data-back]'), function bind(button) {
      button.addEventListener('click', function back() {
        var inAttempt = !screens.attempt.classList.contains('hidden');
        if (inAttempt && currentCode) openAssignment(currentCode);
        else openDashboard();
      });
    });

    show(A.hasTeacherPassword() ? 'dashboard' : 'login');
    if (A.hasTeacherPassword()) openDashboard();
  });
})();
