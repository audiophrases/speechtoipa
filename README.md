# Read • Listen • Speak

A lightweight web app for practicing reading, listening, and pronunciation.

## Running locally

**Windows (easiest):** double-click `speechtoipa.bat`. It installs dependencies on first run, starts the app together with the neural voice server, and opens your browser at `http://127.0.0.1:8787`. Keep the console window open while using the app; close it to stop.

Requires [Node.js](https://nodejs.org). Alternatively run `cd server && npm start` and open `http://127.0.0.1:8787` yourself, or serve the folder with any static file server (e.g. `python -m http.server 8000`) — the app then uses browser voices unless the neural server is also running.

## Voices and natural TTS

The best voice quality comes from the bundled **neural TTS server** (see below). Without it, the app uses the browser's built-in speech synthesis and automatically picks the most natural-sounding voice available: Edge's neural voices ("… Online (Natural)"), iOS/macOS "Premium"/"Enhanced" voices, and Chrome's remote Google voices all outrank plain local voices. Voice selection is fully automatic.

## Neural TTS server (recommended)

Browser voices are inconsistent across platforms — no browser ships Moroccan Arabic, and Catalan support is spotty. The `server/` directory contains a small Node server that streams Microsoft Edge neural voices (the same voices as Azure Speech) with no account or API key:

| Language | Voice |
| --- | --- |
| English (US) | `en-US-JennyNeural` |
| Moroccan Arabic (Darija) | `ar-MA-MounaNeural` |
| Catalan | `ca-ES-JoanaNeural` |
| French | `fr-FR-DeniseNeural` |
| Italian | `it-IT-IsabellaNeural` |
| Spanish | `es-ES-ElviraNeural` |

Run it with `speechtoipa.bat` (Windows) or:

```bash
cd server
npm install
npm start   # serves the app + voices on http://127.0.0.1:8787
```

The server also serves the app itself, so `http://127.0.0.1:8787` is all you need. The app always prefers whichever server is serving its own page (same origin) — this is what makes the exact same code work unmodified once deployed. If opened over a *different* `http://` origin or `file://` instead, it also probes `http://127.0.0.1:8787/health` as a secondary check. Synthesized clips are cached on disk (`server/cache/`) and in the browser, so repeated sentences play instantly.

### Deploying for students (Render)

For students who can't install anything locally (no admin rights, Chromebooks), deploy `server/` — which serves both the voices and the app itself — to Render's free tier:

1. Push this repo to GitHub.
2. In the [Render dashboard](https://dashboard.render.com): **New +** → **Blueprint** → pick the repo. Render reads [render.yaml](render.yaml) and pre-fills everything (no manual config).
3. Share the resulting `https://….onrender.com` URL with students — that's the whole "install."

No card is required on Render's free tier. It sleeps after ~15 minutes idle; the app pings `/health` on load and shows a "waking up" overlay during the ~30-60s cold start, so it doesn't look broken. While a tab stays open and visible, the app also pings periodically to keep the server from re-sleeping mid-lesson.

Because the page and the neural voice API are served from the same place, **no configuration is needed** for the deployed version to get the neural voices — it detects itself automatically. The `tts-base-url` meta tag / `window.TTS_BASE_URL` override is now only needed for the advanced case of pointing this frontend at a *separately* hosted TTS server:

```html
<meta name="tts-base-url" content="https://your-tts-server.example.com">
```

Note: the server uses the unofficial Edge Read Aloud endpoint (via `msedge-tts`). If Microsoft ever locks it down, the same voice names work on the official Azure Speech free tier (500K chars/month) — only the synthesis call in `server/server.js` needs swapping.

## Adding and revising lessons

Lesson content (sentences, translations, tokens) is not stored in this repo —
it lives in a shared Google Sheet that the app fetches live as CSV. See
[SCRIPTER_INSTRUCTIONS.md](SCRIPTER_INSTRUCTIONS.md) for the row schema and
the full workflow for adding new lessons or revising existing ones.

## Guided reading (Read Along-style)

While the mic is open, words light up live as they are read correctly and the next word to read is underlined. If the learner stumbles on a word, the app says "Try saying …" in the base language and then models the word slowly in the target voice (the mic is paused while the app speaks so it never transcribes itself). When the whole sentence is read correctly, the sentence flashes green, a short spoken praise plays, and the app automatically advances to the next sentence and reopens the mic so the learner can keep reading hands-free — or press Play first to listen.

## Text-to-speech (TTS) fallback service

If the browser lacks a locale-appropriate voice or does not support speech synthesis, the app automatically falls back to a lightweight HTTP TTS service. Speed and responsiveness are prioritized over premium audio quality.

### Default service
By default, the app uses Google's free Translate TTS endpoint (`https://translate.googleapis.com/translate_tts`). No sign-up or API key is required. The `lang` parameter is derived from the target language (e.g. `fr` or `en`), and the endpoint returns an audio payload such as `audio/mpeg`.

### Override the service URL
If you want to point to a different service (must expose `GET /tts?text=...&lang=...`), configure the base URL in one of these ways:

1. Add a meta tag in `index.html`:
   ```html
   <meta name="tts-base-url" content="https://your-tts-service.example.com">
   ```
2. Expose a global before `app.js` loads:
   ```html
   <script>window.TTS_BASE_URL = 'https://your-tts-service.example.com';</script>
   ```
3. Or set `window.__TTS_BASE_URL__` if you prefer a non-public global.

The app expects a `GET` endpoint at `${TTS_BASE_URL}/tts?text=...&lang=...` that returns an audio payload (e.g. `audio/mpeg`). Fetched clips are cached in-memory and, for short inputs (≈180 characters or fewer), in `localStorage` keyed by `(text, lang)`.
