// Everything Speech to IPA needs to talk to the assignments back end, shared by
// the student app (app.js) and the teacher page (create/create.js).
//
// The back end is the same Cloudflare Worker that serves Dictation Time, and
// deliberately so: one roster, one teacher password, one place a class's work
// is stored. This app registers as `app: 'ipa'`, which is the only thing that
// keeps the two dashboards apart.
//
// Why the Worker rather than the Node server in server/: that server is on
// Render's free tier and sleeps after ~15 minutes idle. Reading aloud can
// survive a slow voice server — the browser has its own — but a class signing
// in at 8:55am cannot spend the first minute of the lesson watching a spinner.
//
// This is a plain script, not a module, because the rest of the app is: it
// attaches one global and nothing else.
(function attachAssignments(global) {
  'use strict';

  // Same Worker as Dictation Time. Override at runtime with
  // localStorage['ipa.workerBase'] to point a local page at `wrangler dev`.
  var DEFAULT_WORKER_BASE = 'https://dictation-api.eugenime.workers.dev';

  // The neural voice server, used only when this app is served from a static
  // host that has no APIs of its own — see resolveTtsBase in app.js.
  var DEFAULT_RENDER_BASE = 'https://speechtoipa.onrender.com';

  // Self-service doors for a student who can't get in. Neither is part of this
  // app: the roster is a Google Sheet, so signing up is a Form the teacher
  // reads, and the lookup is an Apps Script page that identifies the student by
  // the school Google account they are already signed into and shows them their
  // own row. That is why "forgot password" needs no email sending, no reset
  // tokens and no endpoints here — the school's Google sign-in is the proof of
  // identity. Same values as Dictation Time and pinplay; set either to '' to
  // hide that link.
  var STUDENT_SIGNUP_URL =
    'https://docs.google.com/forms/d/e/1FAIpQLSeTNwOWzYnSR6V6fsSghUkyZLvbhnE5vvcxBYxlj0b0FWpY-g/viewform';
  var STUDENT_LOGIN_LOOKUP_URL =
    'https://script.google.com/macros/s/AKfycbz5lL1e-bzNT8moViNmCzYEf2tiyCEU_j8BmHlQ_8Lvqhryj7dsoAo8yCiFoS4WWc7mqw/exec';

  function stripSlash(value) {
    return String(value || '').trim().replace(/\/+$/, '');
  }

  function override(key) {
    try {
      return stripSlash(localStorage.getItem(key));
    } catch (err) {
      return '';
    }
  }

  function workerBase() {
    return override('ipa.workerBase') || stripSlash(DEFAULT_WORKER_BASE);
  }

  function workerUrl(path) {
    return workerBase() + path;
  }

  function renderBase() {
    return override('ipa.ttsBase') || stripSlash(DEFAULT_RENDER_BASE);
  }

  function ApiError(message, options) {
    var opts = options || {};
    var error = new Error(message);
    error.name = 'ApiError';
    error.status = opts.status || 0;
    // Separated because they need different words on screen: a network error is
    // "try again", a config error is "this copy of the app can't do that".
    error.network = !!opts.network;
    return error;
  }

  /**
   * JSON call to the worker. Always fails fast rather than hanging: a student
   * mid-assignment needs to be told something is wrong while there is still
   * time to retry, not after the lesson has ended.
   */
  async function workerApi(path, options) {
    var opts = options || {};
    var controller = new AbortController();
    var timer = setTimeout(function abort() {
      controller.abort();
    }, opts.timeoutMs || 15000);

    var res;
    try {
      res = await fetch(workerUrl(path), {
        method: opts.method || 'GET',
        headers:
          opts.body === undefined
            ? opts.headers || {}
            : Object.assign({ 'Content-Type': 'application/json' }, opts.headers || {}),
        body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
        signal: controller.signal,
      });
    } catch (err) {
      throw ApiError("Can't reach the assignment server. Check your connection and try again.", {
        network: true,
      });
    } finally {
      clearTimeout(timer);
    }

    var text = await res.text();
    var data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch (err) {
      data = {};
    }

    if (!res.ok) {
      throw ApiError(data.error || 'Request failed (' + res.status + ')', { status: res.status });
    }
    return data;
  }

  // ------------------------------------------------------------------ teacher
  //
  // The password is held in memory only, never in storage: a shared staffroom
  // machine should forget it when the tab closes.

  var teacherPassword = '';

  function setTeacherPassword(value) {
    teacherPassword = String(value || '');
  }

  function getTeacherPassword() {
    return teacherPassword;
  }

  function hasTeacherPassword() {
    return !!teacherPassword;
  }

  function teacherPost(path, body) {
    return workerApi(path, {
      method: 'POST',
      body: Object.assign({}, body || {}, { password: teacherPassword }),
    });
  }

  // ------------------------------------------------------------------ student
  //
  // The attemptId is the student's capability for their attempt, so keeping it
  // lets a Chromebook that died mid-lesson carry on. Scoped per assignment code
  // and cleared once the work is handed in.

  function attemptStorageKey(code) {
    return 'ipa.attempt.' + code;
  }

  function rememberAttempt(code, attemptId, username) {
    try {
      localStorage.setItem(
        attemptStorageKey(code),
        JSON.stringify({ attemptId: attemptId, username: username })
      );
    } catch (err) {
      // Private mode or a full quota: resuming is a convenience, not a promise.
    }
  }

  function recallAttempt(code) {
    try {
      var saved = JSON.parse(localStorage.getItem(attemptStorageKey(code)));
      return saved && saved.attemptId ? saved : null;
    } catch (err) {
      return null;
    }
  }

  function forgetAttempt(code) {
    try {
      localStorage.removeItem(attemptStorageKey(code));
    } catch (err) {
      // Nothing to do; the entry is scoped to this assignment either way.
    }
  }

  /** The assignment code in the page URL, or '' for ordinary free practice. */
  function codeFromUrl() {
    try {
      var raw = new URLSearchParams(global.location.search).get('a') || '';
      return raw.trim().toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 8);
    } catch (err) {
      return '';
    }
  }

  /**
   * Absolute URL of the app root, with a trailing slash, from either page.
   * Worked out at runtime because the app is mounted differently in each of its
   * homes: the domain root on Render and locally, /speechtoipa/ on GitHub
   * Pages. The teacher page lives one level down at <root>/create/, so a
   * student link built there must climb out first or the class would get
   * …/create/?a=CODE and a 404.
   */
  function appRoot() {
    var path = global.location.pathname;
    var url = /\/create\/?$/.test(path)
      ? new URL('../', global.location.href)
      : new URL('./', global.location.href.replace(/\/index\.html$/, '/'));
    return url.href;
  }

  function studentLink(code) {
    return appRoot() + '?a=' + code;
  }

  function createPageUrl() {
    return appRoot() + 'create/';
  }

  global.Assignments = {
    // configuration
    workerBase: workerBase,
    workerUrl: workerUrl,
    renderBase: renderBase,
    STUDENT_SIGNUP_URL: STUDENT_SIGNUP_URL,
    STUDENT_LOGIN_LOOKUP_URL: STUDENT_LOGIN_LOOKUP_URL,

    // plumbing
    workerApi: workerApi,
    ApiError: ApiError,
    codeFromUrl: codeFromUrl,
    appRoot: appRoot,
    studentLink: studentLink,
    createPageUrl: createPageUrl,

    // teacher
    setTeacherPassword: setTeacherPassword,
    getTeacherPassword: getTeacherPassword,
    hasTeacherPassword: hasTeacherPassword,
    verifyTeacherPassword: function verifyTeacherPassword(password) {
      return workerApi('/api/teacher/verify', { method: 'POST', body: { password: password } });
    },
    createAssignment: function createAssignment(fields) {
      return teacherPost(
        '/api/teacher/assignments/create',
        Object.assign({ app: 'ipa' }, fields)
      );
    },
    listAssignments: function listAssignments() {
      return teacherPost('/api/teacher/assignments/list', { app: 'ipa' });
    },
    getAssignment: function getAssignment(code) {
      return teacherPost('/api/teacher/assignments/get', { code: code });
    },
    getAttempt: function getAttempt(code, studentKey, attemptId) {
      return teacherPost('/api/teacher/attempts/get', {
        code: code,
        studentKey: studentKey,
        attemptId: attemptId,
      });
    },
    setAssignmentStatus: function setAssignmentStatus(code, status) {
      return teacherPost('/api/teacher/assignments/status', { code: code, status: status });
    },
    deleteAssignment: function deleteAssignment(code) {
      return teacherPost('/api/teacher/assignments/delete', { code: code });
    },

    // student
    getMeta: function getMeta(code) {
      return workerApi('/api/assignments/' + code + '/meta');
    },
    startAttempt: function startAttempt(code, username, password) {
      return workerApi('/api/assignments/' + code + '/start', {
        method: 'POST',
        body: { username: username, password: password },
        // A roster lookup goes out to Apps Script, which is not always brisk.
        timeoutMs: 25000,
      });
    },
    reportSentence: function reportSentence(code, attemptId, index, score, text) {
      return workerApi('/api/assignments/' + code + '/answer', {
        method: 'POST',
        body: { attemptId: attemptId, index: index, score: score, text: text },
      });
    },
    submitAttempt: function submitAttempt(code, attemptId, scores, answers) {
      return workerApi('/api/assignments/' + code + '/submit', {
        method: 'POST',
        body: { attemptId: attemptId, scores: scores, answers: answers },
        timeoutMs: 25000,
      });
    },

    // resume bookkeeping
    rememberAttempt: rememberAttempt,
    recallAttempt: recallAttempt,
    forgetAttempt: forgetAttempt,
  };
})(typeof window !== 'undefined' ? window : globalThis);
