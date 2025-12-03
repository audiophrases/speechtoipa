#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const COURSE_CODE = 'CA_A1';
const SENTENCE_CSV = path.join(__dirname, '..', 'data', 'CA_A1_SENTENCES.csv');
const TOKENS_DIR = path.join(__dirname, '..', 'data');
const OUTPUT_JSON = path.join(__dirname, '..', 'data', 'ca_a1_course.json');

function parseCsv(content) {
  const rows = [];
  let current = '';
  let inQuotes = false;
  const pushCell = (row, cell) => row.push(cell);
  const pushRow = (row) => {
    rows.push(row.map((cell) => cell.replace(/^"|"$/g, '')));
  };

  let row = [];
  for (let i = 0; i < content.length; i++) {
    const char = content[i];
    const next = content[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      pushCell(row, current);
      current = '';
    } else if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') {
        i++;
      }
      pushCell(row, current);
      pushRow(row);
      row = [];
      current = '';
    } else {
      current += char;
    }
  }

  if (current.length || row.length) {
    pushCell(row, current);
    pushRow(row);
  }

  return rows.filter((r) => r.some((cell) => cell.trim().length));
}

function loadCsvObjects(filePath) {
  const raw = fs.readFileSync(filePath, 'utf8');
  const trimmed = raw.replace(/^\uFEFF/, '');
  const rows = parseCsv(trimmed);
  if (!rows.length) return [];
  const headers = rows[0];
  return rows.slice(1).map((cells) => {
    const obj = {};
    headers.forEach((h, idx) => {
      obj[h.trim()] = (cells[idx] || '').trim();
    });
    return obj;
  });
}

function discoverTokenFiles() {
  return fs
    .readdirSync(TOKENS_DIR)
    .filter((name) => /^CA_A1_U\d+_TOKENS\.csv$/.test(name))
    .map((name) => path.join(TOKENS_DIR, name))
    .sort();
}

function toNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function buildTranslations(row) {
  const languages = ['es', 'en', 'fr', 'it', 'ma'];
  return languages.reduce((acc, code) => {
    const key = `${code}_sentence`;
    if (row[key]) acc[code] = row[key];
    return acc;
  }, {});
}

function buildTokenTranslations(row) {
  const languages = ['es', 'en', 'fr', 'it', 'ma'];
  return languages.reduce((acc, code) => {
    const key = `${code}_gloss`;
    if (row[key]) acc[code] = row[key];
    return acc;
  }, {});
}

function buildSentences(sentenceRows, tokenRows) {
  const tokensBySentence = tokenRows.reduce((acc, row) => {
    const sid = row.sentence_id;
    if (!sid) return acc;
    if (!acc[sid]) acc[sid] = [];
    acc[sid].push(row);
    return acc;
  }, {});

  const sentences = sentenceRows.map((row) => {
    const unit = toNumber(row.unit);
    const sentenceNumber = toNumber(row.sentence_number);
    const tokens = (tokensBySentence[row.sentence_id] || [])
      .sort((a, b) => toNumber(a.token_number) - toNumber(b.token_number))
      .map((token) => ({
        surface: token.ca_token,
        translations: buildTokenTranslations(token),
      }));

    return {
      id: row.sentence_id,
      unit,
      theme: row.theme,
      title: row.title,
      sentenceNumber,
      text: row.ca_sentence,
      translations: buildTranslations(row),
      tokens,
    };
  });

  return sentences.sort((a, b) => {
    if (a.unit === b.unit) {
      return (a.sentenceNumber || 0) - (b.sentenceNumber || 0);
    }
    return (a.unit || 0) - (b.unit || 0);
  });
}

function main() {
  if (!fs.existsSync(SENTENCE_CSV)) {
    console.error(`Sentence CSV not found at ${SENTENCE_CSV}`);
    process.exit(1);
  }

  const sentenceRows = loadCsvObjects(SENTENCE_CSV);
  const tokenFiles = discoverTokenFiles();
  const tokenRows = tokenFiles.flatMap((file) => loadCsvObjects(file));

  console.log(`Loaded ${sentenceRows.length} sentences from CA_A1_SENTENCES.`);
  console.log(`Loaded ${tokenRows.length} tokens from ${tokenFiles.length} token sheet(s).`);

  const sentences = buildSentences(sentenceRows, tokenRows);
  const output = {
    lang: 'ca',
    course: COURSE_CODE,
    level: 'A1',
    source: {
      sentencesCsv: path.basename(SENTENCE_CSV),
      tokenSheets: tokenFiles.map((f) => path.basename(f)),
    },
    sentences,
  };

  fs.writeFileSync(OUTPUT_JSON, JSON.stringify(output, null, 2), 'utf8');
  console.log(`Wrote ${sentences.length} sentences to ${OUTPUT_JSON}`);
}

main();
