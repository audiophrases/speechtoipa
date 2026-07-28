# Read • Listen • Speak

A lightweight web app for practicing reading, listening, and pronunciation.
Teachers can also set a lesson as an assignment: students sign in with their
PinPlay logins, read it aloud, and hand it in.

## Where the app runs

Three pieces, in three places, for one reason each:

| Piece | Where | Why there |
| --- | --- | --- |
| The app itself | **GitHub Pages** | A static host has no cold start, so an assignment link opens instantly. |
| `/tts`, `/api/dictation` | **Render** (`server/`) | The Edge voices need a Node proxy. It may sleep; only free practice waits for it. |
| Assignments, logins, results | **Cloudflare Worker + R2** | Always on, so a class signing in at 8:55am waits for nothing. |

The Worker is the same one Dictation Time uses — one roster, one teacher
password, one place a class's work is stored. This app registers its
assignments as `app: 'ipa'`, which is the only thing keeping the two
dashboards apart. It lives in that repo, at
[`DictationApp/cloudflare/`](https://github.com/audiophrases/DictationApp).

A running copy works out its own back ends: [assignments.js](assignments.js)
holds the Worker address, and `detectTtsServer` in [app.js](app.js) prefers
whichever server is serving the page, falling back to Render's absolute URL
when there isn't one (i.e. on Pages). That is what keeps the same code working
from Render, from GitHub Pages and from `speechtoipa.bat` with no build step.

## Assignments

The teacher's side is its own page at **`/create/`**. Sign in with the teacher
password — the same one as Dictation Time — to build assignments and read
results.

Creating one: **New assignment**, pick a language and a lesson, and its
sentences fill in, one per line. Edit them, or ignore the lesson and type your
own. Set the class, the due date, how many attempts, and **the accuracy the
class is marked at** — the slider a learner can move in free practice is fixed
by the assignment, so everyone is judged alike. You get a six-character code
and a link.

Students open that link and sign in with **the same username and password they
use for PinPlay**, then read each sentence aloud. Unlike a dictation there is
no audio to prepare, so an assignment is ready the moment you create it.

**The marking happens in the browser, not on the server.** Only the recognizer
can judge a spoken sentence, so the app scores each reading and reports it.
What the Worker enforces is that a mark can only go *up*: a student's best
reading of a sentence stands, and a later worse take can't undo it. Each
sentence is reported as it is earned, so a Chromebook that dies mid-lesson
costs nothing — signing back in resumes the same attempt with its marks.

A student who can't get in has the same two doors as in PinPlay, linked under
the sign-in box: **Sign up** (the Google Form that feeds your roster sheet) and
**Forgot username/password** (the Apps Script page that identifies them by the
school Google account they are already signed into). Both live in
[assignments.js](assignments.js); set either to `''` to hide that link.

An assignment stores only its sentences. The per-word data — the translations
behind the tooltips, and for Darija the `ma_latn` transcriptions that *are* the
scoring backbone — is fetched back from the lesson sheet by the student's
browser and matched up by text. Without that a Darija assignment would be
marked against Arabic script the recognizer never returns, and would score near
zero however well it was read. Sentences typed by hand have no lesson behind
them and fall back to plain text, exactly as pasted custom text already does.

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

### Deploying for students

Two targets, and a push to `main` handles the first:

1. **GitHub Pages** — where students open the app, at
   `https://audiophrases.github.io/speechtoipa/`. Pages is set to serve the
   `main` branch root, and since the app is plain HTML/CSS/JS there is nothing
   to build: pushing publishes it. (That is why there is no Actions workflow
   here, unlike Dictation Time, which has a Vite build to run first.)
2. **Render** — the voices. In the
   [Render dashboard](https://dashboard.render.com): **New +** → **Blueprint** →
   pick the repo; [render.yaml](render.yaml) pre-fills everything. No card is
   needed on the free tier. Setting `ROOT_REDIRECT_URL` there sends anyone
   opening the Render address itself on to the Pages one (keeping any `?a=CODE`
   intact), while `/health`, `/tts` and `/api/*` keep answering normally.

Render's free tier sleeps after ~15 minutes idle. The app pings `/health` on
load and shows a "waking up" overlay during the ~30-60s cold start, and keeps
pinging while a visible tab stays open. Assignments are unaffected — they come
from the always-on Worker — so a class signing in never waits for Render.

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

[lessons.js](lessons.js) holds the sheet URLs and the CSV parsing, and is
loaded by both the app and the teacher page — the same lessons offered for
practice are the ones offered as assignments. Dictation Time reads the same
sheet, so editing a sentence there changes it in three places with nothing to
rebuild.

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
