# Speech-to-IPA Tool: Design Exploration

## Problem Statement and Goals
- **Objective:** Build a tool that can ingest raw speech audio in any language (including low-resource tongues and nonsensical "gibberish") and output the corresponding pronunciation in the International Phonetic Alphabet (IPA).
- **Key challenges:**
  - Covering diverse phonetic inventories, suprasegmental features (tone, stress, length), and speaker accents.
  - Handling audio with no associated lexicon or established orthography.
  - Operating across noisy, real-time, or batch audio scenarios.
  - Providing confidence scoring and diagnostics for linguists or downstream systems.
- **Primary users:** Linguists documenting languages, accessibility applications, speech researchers, and creative tools (e.g., voice acting or synthetic speech systems).

## Quick Prototype: Faster-Whisper + IPA Conversion

The repository now includes a small Python package that wires together
[faster-whisper](https://github.com/guillaumekln/faster-whisper) for rapid speech
recognition and [`phonemizer`](https://github.com/bootphon/phonemizer) with the
`espeak-ng` backend for text-to-IPA conversion. It is intentionally lightweight
so that we can iterate quickly before worrying about large-scale feedback or UI
polish.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# system dependencies (Ubuntu/Debian examples)
sudo apt-get update && sudo apt-get install -y ffmpeg espeak-ng
```

- **FFmpeg** is required so faster-whisper can decode common audio formats.
- **espeak-ng** powers the IPA conversion inside `phonemizer`. If you are on
  macOS use `brew install ffmpeg espeak` instead.

### Usage

```bash
python -m speechtoipa.cli path/to/audio.wav --model small --segments
```

The command prints the transcript, IPA rendering, and (optionally) each segment
with timestamps. Use `--output result.json` to capture structured output.

Key options:

| Flag | Purpose |
| ---- | ------- |
| `--model` | Pick any faster-whisper checkpoint (`tiny`, `base`, `small`, etc.). |
| `--language` | Hint Whisper if you already know the language. |
| `--ipa-language` | Force a specific espeak-ng code (defaults to detected language). |
| `--device` / `--compute-type` | Switch to GPU or float precision manually. |
| `--disable-vad` | Skip the built-in voice-activity detector. |
| `--segments` | Print per-segment IPA output for review. |

The JSON payload written with `--output` includes per-segment timings, IPA, and
model metadata so downstream tooling can consume it easily.

### Troubleshooting

- **Mobile browser cache (Opera):** If the web demo fails to load in Opera for
  Android/iOS even though other browsers work, clear the browser cache and
  reload. Opera has been observed to hold on to stale service worker assets,
  preventing the engine from initialising until the cache is reset.

### What’s next?

- Swap in distilled checkpoints or quantised CTranslate2 models for embedded
  devices.
- Support streaming input and diarisation once we graduate from this CLI.
- Layer in richer post-processing (tone marks, articulatory features) once we
  have feedback on the IPA accuracy across multiple languages.

## End-to-End Processing Pipeline
1. **Ingestion & Normalization**
   - Accept streaming (WebRTC/microphone) and batch (upload) audio.
   - Normalize sample rate (e.g., 16 kHz), manage channel conversion, and capture metadata (speaker ID, environment, device).
2. **Segmentation & Filtering**
   - Voice activity detection to trim silence/noise and segment long audio.
   - Optional diarization for multi-speaker inputs and language ID hints.
   - Apply denoising or source separation (e.g., Demucs) when SNR is low.
3. **Feature Extraction**
   - Use self-supervised front-ends (wav2vec 2.0, HuBERT, MMS, Whisper encoder) for robust multilingual representations.
   - Augment with classical features (MFCC, pitch/energy contours) for tone/stress detection.
4. **Core Phonetic Recognition**
   - Predict context-independent phonetic tokens with neural sequence models.
   - Integrate suprasegmental predictors (pitch classifier, duration model) for diacritics.
   - Produce time-aligned phones (start/end timestamps) for later alignment.
5. **IPA Mapping & Post-processing**
   - Map universal phone set (e.g., IPA93, X-SAMPA) to IPA glyphs; handle language-specific allophones via configurable mapping tables.
   - Add markers for uncertain segments (e.g., using parentheses or color coding in UI) based on confidence scores.
   - Collapse repeated phones, smooth transitions, and stitch segments for final transcript.
6. **Outputs & Integrations**
   - Provide IPA transcript, per-phone timings, confidence, and optional audio + phone alignment visualization.
   - Offer export formats (JSON, TextGrid, subtitle-like formats) and API/web UI access.

## Modeling Strategies to Explore
### 1. Transfer Learning from Multilingual Foundation Models
- Fine-tune models like Whisper, XLS-R, or MMS to output IPA tokens directly using CTC/seq2seq objectives.
- Add language embeddings or adapters for high-resource languages while keeping zero-shot ability.
- Use phoneme inventories derived from CMUdict, PanPhon, or PHOIBLE to initialize output layers.

### 2. Modular Pipeline with Language-Conditioned IPA Mapping
- Stage 1: Universal acoustic model outputs language-independent phones (X-SAMPA).
- Stage 2: Lightweight adapter uses language ID (predicted or provided) to map phones to IPA, adding diacritics and allophonic details.
- Benefits: Easy to add new language-specific post-processing without retraining the core recognizer.

### 3. Unsupervised / Few-shot Phone Discovery for Gibberish or New Languages
- Use vector-quantized self-supervised models (wav2vec-U, HuBERT soft clusters) to discover pseudo-phone units.
- Cluster units via articulatory feature models (PanPhon features, articulatory embeddings) and map them to closest IPA symbols.
- Implement human-in-the-loop refinement: linguists can relabel clusters, and system updates mapping tables dynamically.

## Data Strategy
- **Multilingual corpora:** Common Voice, GlobalPhone, BABEL, MLS, OpenSLR archives, and speech corpora with phonetic transcriptions (TIMIT, VoxForge, LibriSpeech-IPA variants).
- **Low-resource augmentation:** Crowdsource minimal pairs, use unsupervised aligners (Montreal Forced Aligner) on text corpora to bootstrap labels, or leverage articulatory synthesizer to generate pseudo-labeled audio.
- **Data curation:** Balance phone coverage, remove noisy labels, and track metadata (language, speaker, channel, tone).
- **Augmentation:** Speed perturbation, pitch shifting, additive noise, reverberation, code-switching synthesis for robust performance.

## IPA Representation and Mapping Details
- Maintain a canonical symbol inventory and alias table (IPA ↔ X-SAMPA ↔ ARPABET) to interoperate with other tools.
- Capture suprasegmentals: tone (Chao digits or tone letters), stress marks, gemination (length marks), nasalization, and secondary articulations.
- Use rule-based layers for coarticulation (e.g., automatic /n/ → [ŋ] before velars) to improve readability while keeping traceable logs of applied rules.
- Provide fallback representation for uncertain phones (e.g., "◊" placeholder or best-N suggestions) to address gibberish/unseen articulations.

## Evaluation Plan
- **Metrics:** Phone Error Rate (PER), articulatory feature accuracy, suprasegmental accuracy (tone/stress), and time alignment error.
- **Benchmarks:** Use corpora with gold IPA or phonetic annotations, plus synthetic benchmarks for rare phones.
- **Human review:** Provide linguist-friendly UI for correction, collect feedback to drive active learning loops.
- **Robustness tests:** Vary noise levels, channel types, speaker accents, and intentionally provide non-linguistic vocalizations to ensure graceful handling.

## Deployment & Tooling Considerations
- **Core stack:** PyTorch for modeling, torchaudio or librosa for preprocessing, Hugging Face Transformers for foundation models.
- **Serving:** Real-time inference via ONNX Runtime or TensorRT; batch jobs orchestrated with Kubernetes or serverless functions.
- **Client interfaces:**
  - Web app showing waveform, spectrogram, IPA timeline, and editable output.
  - REST/gRPC API for programmatic access.
  - Plug-ins for Praat/ELAN to aid linguists.
- **Edge/offline mode:** Quantize models (int8) and prune for mobile/embedded scenarios; optionally ship a distilled model with limited footprint.

## Implementation Phases
1. **Prototype (4–6 weeks):**
   - Assemble multilingual dataset subset with IPA labels.
   - Fine-tune Whisper/XLS-R to emit IPA using CTC decoding.
   - Build CLI tool for batch transcription and inspect accuracy on known languages.
2. **Alpha Tooling (6–8 weeks):**
   - Integrate VAD, diarization, and streaming input.
   - Implement IPA mapping tables and suprasegmental inference.
   - Create simple web UI for visualization and correction.
3. **Robustness & Gibberish Handling (ongoing):**
   - Add unsupervised unit discovery pipeline for unlabeled data.
  - Launch human-in-the-loop labeling interface to refine mapping tables.
   - Introduce active learning to prioritize low-confidence segments.
4. **Production Hardening:**
   - Optimize inference latency, add monitoring (confidence drift, PER via spot checks).
   - Provide APIs, documentation, and packaging for offline deployments.

## Open Research Questions
- Best way to represent non-linguistic vocalizations: should they map to IPA diacritics or special tokens?
- How to infer articulatory features (e.g., airstream mechanism) directly from embeddings without labels?
- Strategies for covering highly tonal or click-rich languages without extensive labeled data.
- Approaches to calibrate confidence scores across languages and speakers.

## Next Steps
- Prioritize data acquisition and labeling agreements for underrepresented phonetic inventories.
- Experiment with multitask objectives (phone + articulatory feature prediction) on top of a self-supervised encoder.
- Evaluate existing forced aligners for generating weak IPA labels to bootstrap training.
- Design UI wireframes focusing on linguist workflows to guide development of correction tools.
