# Pronunciation Practice

An interactive web exercise that helps students practise reading aloud. Pick a
paragraph, listen to the model reader, then record yourself. A bundled local
server transcribes the raw audio with [faster-whisper](https://github.com/guillaumekln/faster-whisper)
and produces IPA strings via `phonemizer` + `espeak-ng`, all without relying on
any external cloud APIs. The recognised transcript is compared against the
source text and each word is coloured green or red to highlight potential
pronunciation slips.

## Features
- Curated paragraph bank with approachable, imagery-rich sentences.
- One-click text-to-speech playback using the Web Speech synthesis engine.
- Microphone-powered practice session with feedback after each attempt.
- Word-by-word highlighting to make it easy to spot tricky phrases.
- Works entirely on your machine—no accounts or external services required.
- FastAPI-powered local inference endpoint so the browser can stay offline.

## Getting Started
1. Install the Python dependencies and speech engine:
   ```bash
   pip install -r requirements.txt
   # On Debian/Ubuntu
   sudo apt-get install espeak-ng
   ```
   The first transcription will automatically download the requested
   faster-whisper model (default: `small`) into your local cache. To control the
   location, set `FWHISPER_ASSET_DIR` before running the server.
2. Launch the offline inference endpoint:
   ```bash
   python -m speechtoipa.local_server --host 127.0.0.1 --port 8000 --model-size small
   ```
   Use `--device cuda` and an appropriate `--compute-type` if you have a GPU.
   The server also serves `index.html`, so you can point your browser straight
   at `http://127.0.0.1:8000/`.
3. Choose a paragraph from the dropdown and press **Play model reading** to hear
   the reference pronunciation.
4. When you're ready, press **Start practice**. Your microphone audio is
   recorded locally, converted to WAV in the browser, and sent to the local
   server for transcription.
5. Review the transcript and colour-coded feedback, then repeat as needed.

## How it Works
- **Speech synthesis:** The Web Speech API (`speechSynthesis`) reads the
  paragraph at a slightly reduced pace to encourage mindful listening.
- **Speech recognition:** The browser captures raw audio with the
  `MediaRecorder` API, converts it to WAV, and posts the data to the local
  FastAPI endpoint.
- **Transcription & phonemisation:** The Python server feeds the audio into
  `speechtoipa.pipeline.transcribe_audio_to_ipa`, which uses faster-whisper to
  recognise text and `espeak-ng` (via `phonemizer`) to derive IPA strings.
- **Comparison:** The transcript is normalised (lowercased, punctuation
  removed) and compared token by token to the source paragraph. Matching words
  are marked green; mismatches remain red so students can focus on improvement.

## Extending the Exercise
- Add longer stories or curriculum-aligned passages to the `PARAGRAPHS` array in
  `index.html`.
- Experiment with different speech synthesis voices via
  `speechSynthesis.getVoices()`.
- Persist session history by storing attempts in `localStorage`, or wire in a
  backend if you need instructor dashboards.

## Accessibility & Privacy
- Audio never leaves your machine; it only flows between the browser and the
  locally running FastAPI server.
- Buttons include emoji cues and large tap targets for touch users.
- ARIA live regions announce updates so screen readers stay informed.

Enjoy practising and refining your pronunciation!
