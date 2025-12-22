# Read • Listen • Speak

A lightweight web app for practicing reading, listening, and pronunciation.

## Running locally

Open `index.html` in a modern browser or serve the folder with any static file server (e.g. `python -m http.server 8000`).

## Text-to-speech (TTS) fallback service

The app first prefers the browser's built-in speech synthesis. If the browser lacks a locale-appropriate voice or does not support speech synthesis, it automatically falls back to a lightweight HTTP TTS service. Speed and responsiveness are prioritized over premium audio quality.

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
