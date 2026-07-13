# Adding and revising lessons

This app has no lesson content in the repo. All lessons and sentences live in a
shared Google Sheet — one tab per target language — that the app fetches live
as CSV every time it loads. There is no code to write and nothing to build or
deploy: edit the sheet, refresh the app, done.

Target languages and their sheet tabs (`gid`s) are wired up in
`MASTER_CSV_URLS` in [app.js](app.js) — that's the source of truth for which
spreadsheet and tabs are live. Currently: English, French, Catalan, Italian,
Moroccan Darija (`ma`).

## The row schema

Every tab uses the same columns:

```
lesson_title, lesson_id, sentence_id, token_id, <target_lang>, ca, en, fr, es, it, ma[, ma_latn]
```

Two kinds of rows, always in this order — a sentence row, then its token rows
directly below:

**Sentence row** (`token_id` empty):
| lesson_title | lesson_id | sentence_id | token_id | en | ca | fr | es | it | ma |
|---|---|---|---|---|---|---|---|---|---|
| A1 - Introductions | en_a1_u1_introductions | en_a1_01_001 | | Hi, my name is Marc. | Hola, em dic Marc. | Bonjour, je m'appelle Marc. | Hola, me llamo Marc. | Ciao, mi chiamo Marc. | سلام، سميتي مارك. |

- `lesson_title` — only needs to be set on each lesson's *first* row; it becomes the label shown in the Lesson dropdown.
- The target-language column (`en` in the English tab, `fr` in the French tab, etc.) holds the full sentence text.
- Every other language column holds that sentence's translation, shown as a hover tooltip.

**Token rows** (`token_id` filled), one per word in the sentence, right after it:
| lesson_title | lesson_id | sentence_id | token_id | en | ca | fr | es | it | ma |
|---|---|---|---|---|---|---|---|---|---|
| | en_a1_u1_introductions | en_a1_01_001 | en_a1_01_001_001 | Hi | Hola | Salut | Hola | Ciao | سلام |
| | en_a1_u1_introductions | en_a1_01_001 | en_a1_01_001_002 | my | el meu | mon | mi | mio | ديالي |
| | en_a1_u1_introductions | en_a1_01_001 | en_a1_01_001_003 | name | nom | nom | nombre | nome | سمية |

Token rows drive the word-by-word tooltip, the live word-by-word highlighting
while the mic is listening, and pronunciation scoring.

**Darija (`ma`) tabs only** have one extra column, `ma_latn`: a Latin-script
transcription of the same word/sentence (e.g. `salam, smiyti Mark.`). This is
not cosmetic — Arabic-script speech recognition is unreliable, so the app
actually scores the learner's pronunciation against `ma_latn`, not the Arabic
script. Every Darija sentence and token row needs it filled in.

## Two rules that will silently break a sentence if you skip them

1. **Every token's text must appear verbatim, in order, inside the sentence
   text.** The app finds each word by searching for its exact substring in the
   sentence (`"Hi, my name is Marc.".indexOf("my")`). If a token is spelled
   differently from how it appears in the sentence — different punctuation,
   capitalization, a typo — that word is silently skipped: no highlighting, no
   scoring, no tooltip, and no error anywhere.
2. **`token_id` must sort correctly as *text*, not as a number.** Keep the
   zero-padded suffix style already used everywhere: `..._001`, `..._002`, …
   `..._010`. Without padding, `_2` sorts after `_10` and words come out in
   the wrong order.

Sentences within a lesson play in **row order in the sheet** — the numeric
part of `sentence_id` is just a label, not a sort key. To reorder sentences,
drag the rows.

## How to add a new lesson

1. Pick a `lesson_id` that doesn't already exist in that language's tab (follow the existing pattern, e.g. `en_a1_u21_directions`).
2. Add your sentence rows and their token rows, following the schema above.
3. Fill in translations for every other language column — they power the tooltips and the spoken coaching/praise phrases (which speak in the learner's base language).
4. Save. Reload the app (a hard refresh) — the new lesson appears in the Lesson dropdown automatically. No code, build step, or deploy involved.

## How to revise an existing sentence

1. Edit the sentence-row cell for the language(s) you're changing.
2. If you changed the target-language wording, update its token rows to match — add, remove, or re-split token rows so every token is still an exact substring of the new sentence text, in order.
3. Save and hard-refresh the app to see the change.

The app only fetches each language's sheet once per page load (cached in
memory), and Google's "publish to web" has its own short propagation delay
after you save — if an edit doesn't show up immediately, wait a few seconds
and refresh again.

## Optional columns for tricky words

Two columns are already supported by the app but not currently used in any
tab — add them to a tab if you need them:

- `pronunciation_aliases` — pipe-separated alternate spellings/pronunciations accepted for a token (e.g. `Mohammed|Mohamed|Muhammad`).
- `is_proper_noun` — set to `true`/`1`/`yes` on a token row to relax how strictly that word's pronunciation is scored (useful for names).

## What NOT to touch

There is no lesson content checked into this repo (no CSV or JSON files to
edit locally) — the Google Sheet is the only place lesson content lives. If
you see references to a "master CSV" or "sheet" anywhere in `app.js`
(`MASTER_CSV_URLS`, `ensureMasterRowsForLang`), that's the fetch logic — the
content itself is not there.
