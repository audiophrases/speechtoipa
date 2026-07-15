const test = require('node:test');
const assert = require('node:assert');

const {
  filterUnexpectedRepeats,
  tokenizeText,
  rankVoicesForLang,
  getVoiceNaturalness,
  splitIntoSentences,
  buildDarijaSpokenTokens,
  applyOutOfOrderConfirmation,
} = require('./app.js');

test('drops repeated sequences once expected counts are met', () => {
  const targetTokens = tokenizeText('hi my name is marc', 'en');
  const spoken = 'hi my hi my hi my name is marc';

  const { filteredTokens } = filterUnexpectedRepeats(spoken, targetTokens, 'en');

  assert.deepStrictEqual(filteredTokens, tokenizeText('hi my name is marc', 'en'));
});

test('keeps legitimate consecutive duplicates but trims extras', () => {
  const targetTokens = tokenizeText('very very good job', 'en');
  const spoken = 'very very very good job';

  const { filteredTokens } = filterUnexpectedRepeats(spoken, targetTokens, 'en');

  assert.deepStrictEqual(filteredTokens, tokenizeText('very very good job', 'en'));
});

test('removes extra occurrences even when they are not consecutive', () => {
  const targetTokens = tokenizeText('to be or not to be', 'en');
  const spoken = 'to be or not to be to be';

  const { filteredTokens } = filterUnexpectedRepeats(spoken, targetTokens, 'en');

  assert.deepStrictEqual(filteredTokens, tokenizeText('to be or not to be', 'en'));
});

test('strips elision apostrophes so "l’a" matches the recognizer’s "la"', () => {
  const target = tokenizeText('l’a fait la France face à l’Espagne', 'fr');
  const spoken = tokenizeText("la fait la france face a l'espagne", 'fr');

  assert.deepStrictEqual(target, spoken);
});

test('strips apostrophes from English contractions', () => {
  assert.deepStrictEqual(tokenizeText("don't stop", 'en'), ['dont', 'stop']);
});

test('normalizes digit tokens to match spelled-out numbers', () => {
  const spelled = tokenizeText('I have two apples', 'en');
  const digits = tokenizeText('I have 2 apples', 'en');

  assert.deepStrictEqual(digits, spelled);
});

test('tokenizes digit sequences as number tokens', () => {
  assert.deepStrictEqual(tokenizeText('1 2 3 4 5', 'en'), [
    'one',
    'two',
    'three',
    'four',
    'five',
  ]);
});

test('normalizes French digit tokens to match spelled-out numbers', () => {
  const spelled = tokenizeText('quatorze', 'fr');
  const digits = tokenizeText('14', 'fr');

  assert.deepStrictEqual(digits, spelled);
});

test('tokenizes French digit sequences into number words', () => {
  assert.deepStrictEqual(tokenizeText('1 2 3 4 5', 'fr'), [
    'un',
    'deux',
    'trois',
    'quatre',
    'cinq',
  ]);
});

test('normalizes time tokens to match o\'clock phrase in English', () => {
  const timeTokens = tokenizeText('1:00', 'en');
  const wordTokens = tokenizeText("one o'clock", 'en');

  assert.deepStrictEqual(timeTokens, wordTokens);
});

test('normalizes time tokens to match hour phrases in French', () => {
  const timeTokens = tokenizeText('1:00', 'fr');
  const wordTokens = tokenizeText('une heure', 'fr');

  assert.deepStrictEqual(timeTokens, wordTokens);
});

test('scores neural and premium voices above plain ones', () => {
  assert.strictEqual(
    getVoiceNaturalness({ name: 'Microsoft Aria Online (Natural)', localService: false }) > 0,
    true
  );
  assert.strictEqual(
    getVoiceNaturalness({ name: 'Google US English', localService: false }) > 0,
    true
  );
  assert.strictEqual(
    getVoiceNaturalness({ name: 'Microsoft David - English (United States)', localService: true }),
    0
  );
});

test('splits a pasted paragraph into individual sentences', () => {
  const text = 'Hi, my name is Marc. I am from Spain! Do you like pizza?';
  assert.deepStrictEqual(splitIntoSentences(text), [
    'Hi, my name is Marc.',
    'I am from Spain!',
    'Do you like pizza?',
  ]);
});

test('splits one-sentence-per-line custom text even without punctuation', () => {
  const text = 'Hello there\nHow are you\nGoodbye';
  assert.deepStrictEqual(splitIntoSentences(text), [
    'Hello there',
    'How are you',
    'Goodbye',
  ]);
});

test('splits Darija custom text on Arabic sentence punctuation', () => {
  const text = 'سلام. كيف حالك؟';
  assert.deepStrictEqual(splitIntoSentences(text), ['سلام.', 'كيف حالك؟']);
});

test('falls back to the whole trimmed text when nothing splits it', () => {
  assert.deepStrictEqual(splitIntoSentences('   just one phrase   '), ['just one phrase']);
});

test('returns an empty list for blank custom text', () => {
  assert.deepStrictEqual(splitIntoSentences(''), []);
  assert.deepStrictEqual(splitIntoSentences('   \n  '), []);
});

test('Darija Arabic-script matching tolerates a single misheard letter', () => {
  const sentence = {
    tokens: [
      { surface: 'سلام', transcription: 'salam' },
      { surface: 'لباس', transcription: 'labas' },
    ],
  };

  // Recognizer got the second word's first letter wrong (ل -> ن), as
  // real speech recognizers commonly do with similar-sounding consonants.
  const { spokenTokens } = buildDarijaSpokenTokens('سلامنباس', sentence);

  assert.deepStrictEqual(spokenTokens, ['salam', 'labas']);
});

test('Darija Arabic-script matching still rejects an unrelated word', () => {
  const sentence = {
    tokens: [
      { surface: 'سلام', transcription: 'salam' },
      { surface: 'لباس', transcription: 'labas' },
    ],
  };

  const { spokenTokens } = buildDarijaSpokenTokens('سلامكتاب', sentence);

  assert.deepStrictEqual(spokenTokens, ['salam']);
});

test('out-of-order buffer does not block later words after a single skipped word', () => {
  // Mirrors "l'a fait la France..." where "l'a" fails the match threshold
  // (French elision vs. the recognizer's "la") but every word after it was
  // spoken correctly and in order — they must all light up immediately.
  const wordStatus = ['pending', 'correct', 'correct', 'correct', 'correct', 'correct', 'correct'];
  const sinceMap = new Map();

  applyOutOfOrderConfirmation(wordStatus, 0, wordStatus.length, false, sinceMap, 1000);

  assert.deepStrictEqual(wordStatus, [
    'pending',
    'correct',
    'correct',
    'correct',
    'correct',
    'correct',
    'correct',
  ]);
  assert.strictEqual(sinceMap.size, 0);
});

test('out-of-order buffer holds back a match that jumps over two+ unspoken words', () => {
  const wordStatus = ['correct', 'pending', 'pending', 'correct'];
  const sinceMap = new Map();

  applyOutOfOrderConfirmation(wordStatus, 0, wordStatus.length, false, sinceMap, 1000);

  assert.deepStrictEqual(wordStatus, ['correct', 'pending', 'pending', 'pending']);
  assert.strictEqual(sinceMap.get(3), 1000);
});

test('out-of-order buffer confirms a jump-ahead match once it survives the delay', () => {
  const sinceMap = new Map([[3, 1000]]);
  const wordStatus = ['correct', 'pending', 'pending', 'correct'];

  applyOutOfOrderConfirmation(wordStatus, 0, wordStatus.length, false, sinceMap, 2000);

  assert.strictEqual(wordStatus[3], 'correct');
  assert.strictEqual(sinceMap.has(3), false);
});

test('out-of-order buffer is bypassed for final recognition results', () => {
  const wordStatus = ['correct', 'pending', 'pending', 'correct'];
  const sinceMap = new Map();

  applyOutOfOrderConfirmation(wordStatus, 0, wordStatus.length, true, sinceMap, 1000);

  assert.deepStrictEqual(wordStatus, ['correct', 'pending', 'pending', 'correct']);
  assert.strictEqual(sinceMap.size, 0);
});

test('ranks natural voices above local standard voices for a language', () => {
  const voices = [
    { name: 'Microsoft David - English (United States)', lang: 'en-US', localService: true },
    { name: 'Google US English', lang: 'en-US', localService: false },
    { name: 'Microsoft Aria Online (Natural) - English (United States)', lang: 'en-US', localService: false },
    { name: 'Google français', lang: 'fr-FR', localService: false },
  ];

  const ranked = rankVoicesForLang(voices, 'en-US');

  assert.strictEqual(
    ranked[0].name,
    'Microsoft Aria Online (Natural) - English (United States)'
  );
  // Off-language voices always rank last.
  assert.strictEqual(ranked[ranked.length - 1].name, 'Google français');
});
