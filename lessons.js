// The lesson sheet: where the course content actually lives.
//
// Both pages need it — the student app to practise a lesson, the teacher page
// to offer them when building an assignment — so it is loaded once here rather
// than parsed twice. Loaded before app.js and before create.js, which use these
// as ordinary globals (both are plain scripts, not modules).
//
// The sheet is published read-only, so it answers with
// `Access-Control-Allow-Origin: *` and needs no key or proxy. Editing a
// sentence there changes it in both apps with nothing to rebuild — see
// SCRIPTER_INSTRUCTIONS.md for the row schema.

const MASTER_CSV_URLS = {
  ca: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=1216373156&single=true&output=csv',
  en: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=1053057720&single=true&output=csv',
  fr: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=484976070&single=true&output=csv',
  it: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=1338439854&single=true&output=csv',
  ma: 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQl1GNJGHAilkpQn3KiB0HnrUGEXSQp_dwo6A548izQXL-iAtAIHB2g3_o6VYAOv6UFuUOcISzJQO61/pub?gid=710375040&single=true&output=csv',
};

// One tab is a few hundred KB, so a language is fetched at most once per page.
const MASTER_ROWS_BY_LANG = {};

function parseCourseCsv(text) {
  const rows = [];
  let current = '';
  let inQuotes = false;
  let row = [];

  const pushCell = () => {
    row.push(current);
    current = '';
  };
  const pushRow = () => {
    if (row.length) {
      rows.push(row.map((cell) => cell.replace(/^"|"$/g, '')));
      row = [];
    }
  };

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      pushCell();
    } else if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') {
        i++;
      }
      pushCell();
      pushRow();
    } else {
      current += char;
    }
  }

  if (current.length || row.length) {
    pushCell();
    pushRow();
  }

  if (!rows.length) return [];

  const headers = rows[0].map((h) => h.trim());
  return rows
    .slice(1)
    .filter((cells) => cells.some((c) => c && c.trim().length))
    .map((cells) => {
      const obj = {};
      headers.forEach((h, idx) => {
        obj[h] = (cells[idx] || '').trim();
      });
      if (obj.pronunciation_aliases) {
        obj.pronunciation_aliases = obj.pronunciation_aliases
          .split('|')
          .map((alias) => alias.trim())
          .filter(Boolean);
      }
      return obj;
    });
}

async function ensureMasterRowsForLang(lang) {
  if (MASTER_ROWS_BY_LANG[lang]) return MASTER_ROWS_BY_LANG[lang];

  const url = MASTER_CSV_URLS[lang];
  if (!url) return null;

  const res = await fetch(url);
  if (!res.ok) {
    console.error('Failed to fetch master CSV for', lang, res.status);
    return null;
  }
  const text = await res.text();
  const rows = parseCourseCsv(text);
  MASTER_ROWS_BY_LANG[lang] = rows;

  console.log('Loaded master rows for', lang, 'count =', rows.length);
  return rows;
}

// Every lesson in a language, in sheet order: { id, label }. A lesson's rows
// carry its title on each line, so the first occurrence wins.
function lessonsFromRows(rows) {
  const lessons = new Map();
  (rows || []).forEach((row) => {
    if (!row.lesson_id || lessons.has(row.lesson_id)) return;
    lessons.set(row.lesson_id, {
      id: row.lesson_id,
      label: row.lesson_title || row.lesson_id,
    });
  });
  return Array.from(lessons.values());
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { parseCourseCsv, lessonsFromRows, MASTER_CSV_URLS };
}
