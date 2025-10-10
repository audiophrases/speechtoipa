# Reading Pronunciation Practice

An interactive web exercise that helps students practice reading aloud. Pick a
paragraph, listen to the model reader, then record yourself. The site captures
your microphone audio with the browser's Speech Recognition API, compares the
recognized transcript against the source text, and colours each word green or
red to highlight potential pronunciation slips.

## Features
- Curated paragraph bank with approachable, imagery-rich sentences.
- One-click text-to-speech playback using the Web Speech synthesis engine.
- Microphone-powered practice session with live feedback after each attempt.
- Word-by-word highlighting to make it easy to spot tricky phrases.
- Works entirely in the browser—no accounts, servers, or external services.

## Getting Started
1. Open `index.html` in a modern Chromium-based browser (Chrome, Edge, Brave).
2. Choose a paragraph from the dropdown.
3. Press **Play model reading** to hear the reference pronunciation.
4. When you're ready, press **Start practice** and read the paragraph aloud.
5. Review the transcript and colour-coded feedback, then repeat as needed.

> **Note:** Speech recognition currently works best in desktop Chrome. Other
> browsers may not expose the necessary APIs or may require enabling
> experimental flags.

## How it Works
- **Speech synthesis:** The Web Speech API (`speechSynthesis`) reads the
  paragraph at a slightly reduced pace to encourage mindful listening.
- **Speech recognition:** `SpeechRecognition`/`webkitSpeechRecognition`
  listens through the microphone and returns a transcript once you finish.
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
- All interactions happen locally in the browser. No audio leaves the device.
- Buttons include emoji cues and large tap targets for touch users.
- ARIA live regions announce updates so screen readers stay informed.

Enjoy practising and refining your pronunciation!
