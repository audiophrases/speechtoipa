import http from 'node:http';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MsEdgeTTS, OUTPUT_FORMAT } from 'msedge-tts';

const PORT = Number(process.env.PORT) || 8787;
const MAX_TEXT_LENGTH = 500;
const CACHE_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'cache');

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

  if (url.pathname !== '/tts') {
    sendJson(res, 404, { error: 'Not found. Use /tts?text=...&lang=...' });
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
  console.log(`speechtoipa TTS server listening on http://127.0.0.1:${PORT}`);
  console.log('Voices:', VOICES);
});
