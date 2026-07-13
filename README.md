# Read • Listen • Speak

A lightweight web app for practicing reading, listening, and pronunciation.

## Running locally

Open `index.html` in a modern browser or serve the folder with any static file server (e.g. `python -m http.server 8000`).

## Voices and natural TTS

The app first prefers the browser's built-in speech synthesis and automatically ranks the most natural-sounding voices first: Edge's neural voices ("… Online (Natural)"), iOS/macOS "Premium"/"Enhanced" voices, and Chrome's remote Google voices all outrank plain local voices. Natural voices are marked with ✨ in the voice picker. For the best free voice quality, open the app in Microsoft Edge, whose neural voices are exposed through the standard Web Speech API.

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
