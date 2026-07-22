import http from 'node:http';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MsEdgeTTS, OUTPUT_FORMAT } from 'msedge-tts';
import { handleDictationRequest } from './dictation.js';

const PORT = Number(process.env.PORT) || 8787;
const MAX_TEXT_LENGTH = 500;
const SERVER_DIR = path.dirname(fileURLToPath(import.meta.url));
const CACHE_DIR = path.join(SERVER_DIR, 'cache');
// The server also serves the app itself, so `speechtoipa.bat` (or `npm start`)
// is all that's needed to run everything at one URL.
const APP_ROOT = path.join(SERVER_DIR, '..');
// Diagnostic log the browser app appends to (when ?debug=1), so the actual
// recognizer transcript and matching decisions can be inspected off-device.
const DEBUG_LOG_DIR = path.join(APP_ROOT, 'logs');
const DEBUG_LOG_FILE = path.join(DEBUG_LOG_DIR, 'debug.log');
const MAX_DEBUG_BODY = 256 * 1024;

const STATIC_MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.mp3': 'audio/mpeg',
  '.md': 'text/plain; charset=utf-8',
};

// One neural voice per app language, keyed by the primary language subtag the
// client sends (e.g. "en-US" -> "en", "ar-SA" -> "ar"). The app's only Arabic
// content is Moroccan Darija, so all Arabic requests use the ar-MA voice.
const VOICES = {
  en: 'en-US-JennyNeural',
  ar: 'ar-MA-MounaNeural',
  ca: 'ca-ES-JoanaNeural',
  fr: 'fr-FR-DeniseNeural',
  it: 'it-IT-IsabellaNeural',
  es: 'es-ES-ElviraNeural',
};
const DEFAULT_VOICE = VOICES.en;

function pickVoice(lang, voiceOverride) {
  if (voiceOverride && /^[a-zA-Z-]+$/.test(voiceOverride)) return voiceOverride;
  const primary = String(lang || '').toLowerCase().split('-')[0];
  return VOICES[primary] || DEFAULT_VOICE;
}

function cachePathFor(voice, text) {
  const hash = crypto.createHash('sha1').update(`${voice}|${text}`).digest('hex');
  return path.join(CACHE_DIR, `${hash}.mp3`);
}

async function synthesize(text, voice) {
  const tts = new MsEdgeTTS();
  await tts.setMetadata(voice, OUTPUT_FORMAT.AUDIO_24KHZ_48KBITRATE_MONO_MP3);
  const { audioStream } = await tts.toStream(text);

  const chunks = [];
  for await (const chunk of audioStream) {
    chunks.push(chunk);
  }
  const audio = Buffer.concat(chunks);
  if (!audio.length) {
    throw new Error('Synthesis returned no audio');
  }
  return audio;
}

function sendCors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

function sendJson(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

// Appends one JSON diagnostic entry (JSONL) from the browser app. Each
// recording session clears the file first (DELETE), so debug.log always holds
// just the most recent session — easy to read start to finish.
function handleDebugLog(req, res) {
  if (req.method === 'DELETE') {
    try {
      fs.mkdirSync(DEBUG_LOG_DIR, { recursive: true });
      fs.writeFileSync(DEBUG_LOG_FILE, '');
    } catch (err) {
      console.error('debug-log clear failed:', err.message || err);
    }
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method !== 'POST') {
    sendJson(res, 405, { error: 'Method not allowed' });
    return;
  }

  let body = '';
  let aborted = false;
  req.on('data', (chunk) => {
    body += chunk;
    if (body.length > MAX_DEBUG_BODY) {
      aborted = true;
      req.destroy();
    }
  });
  req.on('end', () => {
    if (aborted) return;
    try {
      const parsed = JSON.parse(body || '{}');
      fs.mkdirSync(DEBUG_LOG_DIR, { recursive: true });
      fs.appendFileSync(DEBUG_LOG_FILE, JSON.stringify(parsed) + '\n');
      res.writeHead(204);
      res.end();
    } catch (err) {
      sendJson(res, 400, { error: 'Invalid log payload' });
    }
  });
  req.on('error', () => {
    aborted = true;
  });
}

function serveStatic(pathname, res) {
  let decoded;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    sendJson(res, 400, { error: 'Bad request' });
    return;
  }
  if (decoded === '/') decoded = '/index.html';

  const filePath = path.normalize(path.join(APP_ROOT, decoded));
  const insideRoot = filePath.startsWith(APP_ROOT + path.sep) || filePath === path.join(APP_ROOT, 'index.html');
  if (!insideRoot) {
    sendJson(res, 403, { error: 'Forbidden' });
    return;
  }

  let stat;
  try {
    stat = fs.statSync(filePath);
  } catch {
    sendJson(res, 404, { error: 'Not found' });
    return;
  }
  if (!stat.isFile()) {
    sendJson(res, 404, { error: 'Not found' });
    return;
  }

  const ext = path.extname(filePath).toLowerCase();
  res.writeHead(200, {
    'Content-Type': STATIC_MIME[ext] || 'application/octet-stream',
    // no-store, not no-cache: the app is developed against a running server,
    // and stale app.js in an open tab has repeatedly masked fixes. There's no
    // validator (ETag/Last-Modified) to revalidate against, so forbid storing
    // outright — the files are tiny and local.
    'Cache-Control': 'no-store',
  });
  fs.createReadStream(filePath).pipe(res);
}

const server = http.createServer(async (req, res) => {
  sendCors(res);

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const url = new URL(req.url, `http://localhost:${PORT}`);

  if (url.pathname === '/health') {
    sendJson(res, 200, { ok: true, service: 'speechtoipa-tts', voices: VOICES });
    return;
  }

  if (url.pathname === '/voices') {
    sendJson(res, 200, VOICES);
    return;
  }

  if (url.pathname === '/debug-log') {
    handleDebugLog(req, res);
    return;
  }

  if (url.pathname === '/api/dictation') {
    await handleDictationRequest(req, res);
    return;
  }

  if (url.pathname !== '/tts') {
    serveStatic(url.pathname, res);
    return;
  }

  const text = (url.searchParams.get('text') || '').trim();
  const lang = url.searchParams.get('lang') || 'en';
  const voice = pickVoice(lang, url.searchParams.get('voice'));

  if (!text) {
    sendJson(res, 400, { error: 'Missing text parameter' });
    return;
  }
  if (text.length > MAX_TEXT_LENGTH) {
    sendJson(res, 413, { error: `Text too long (max ${MAX_TEXT_LENGTH} chars)` });
    return;
  }

  const cacheFile = cachePathFor(voice, text);

  try {
    let audio;
    if (fs.existsSync(cacheFile)) {
      audio = fs.readFileSync(cacheFile);
    } else {
      audio = await synthesize(text, voice);
      fs.mkdirSync(CACHE_DIR, { recursive: true });
      fs.writeFileSync(cacheFile, audio);
    }

    res.writeHead(200, {
      'Content-Type': 'audio/mpeg',
      'Content-Length': audio.length,
      'Cache-Control': 'public, max-age=31536000, immutable',
      'X-Voice': voice,
    });
    res.end(audio);
  } catch (err) {
    console.error(`Synthesis failed (voice=${voice}):`, err.message || err);
    sendJson(res, 502, { error: 'Synthesis failed', detail: String(err.message || err) });
  }
});

server.listen(PORT, () => {
  console.log(`speechtoipa running at http://127.0.0.1:${PORT}`);
  console.log('Neural voices:', VOICES);
});
