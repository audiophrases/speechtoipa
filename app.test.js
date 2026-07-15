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
  findMatchesForTargetTokens,
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

test('matches an elided "l’a" against the recognizer’s fused "la"', () => {
  const targetTokens = tokenizeText('l’a fait la France face à l’Espagne', 'fr');
  const spokenTokens = tokenizeText("la fait la france face a l'espagne", 'fr');

  const matches = findMatchesForTargetTokens(targetTokens, spokenTokens, { langCode: 'fr' });

  assert.strictEqual(matches.filter(Boolean).length, targetTokens.length);
});

test('matches an English contraction against its fused recognized form', () => {
  const targetTokens = tokenizeText("don't stop", 'en');
  const spokenTokens = tokenizeText('dont stop', 'en');

  const matches = findMatchesForTargetTokens(targetTokens, spokenTokens, { langCode: 'en' });

  assert.strictEqual(matches.filter(Boolean).length, targetTokens.length);
});

test('tokenizeText still keeps apostrophes intact (unaffected by scoring-level fix)', () => {
  assert.deepStrictEqual(tokenizeText("don't stop", 'en'), ["don't", 'stop']);
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
  const tokens = ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6'];
  const wordStatus = ['pending', 'correct', 'correct', 'correct', 'correct', 'correct', 'correct'];
  const sinceMap = new Map();

  applyOutOfOrderConfirmation(wordStatus, tokens, 0, false, sinceMap, 1000);

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
  const tokens = ['w0', 'w1', 'w2', 'w3'];
  const wordStatus = ['correct', 'pending', 'pending', 'correct'];
  const sinceMap = new Map();

  applyOutOfOrderConfirmation(wordStatus, tokens, 0, false, sinceMap, 1000);

  assert.deepStrictEqual(wordStatus, ['correct', 'pending', 'pending', 'pending']);
  assert.strictEqual(sinceMap.get(3), 1000);
});

test('out-of-order buffer confirms a jump-ahead match once it survives the delay', () => {
  const tokens = ['w0', 'w1', 'w2', 'w3'];
  const sinceMap = new Map([[3, 1000]]);
  const wordStatus = ['correct', 'pending', 'pending', 'correct'];

  applyOutOfOrderConfirmation(wordStatus, tokens, 0, false, sinceMap, 2000);

  assert.strictEqual(wordStatus[3], 'correct');
  assert.strictEqual(sinceMap.has(3), false);
});

test('out-of-order buffer is bypassed for final recognition results', () => {
  const tokens = ['w0', 'w1', 'w2', 'w3'];
  const wordStatus = ['correct', 'pending', 'pending', 'correct'];
  const sinceMap = new Map();

  applyOutOfOrderConfirmation(wordStatus, tokens, 0, true, sinceMap, 1000);

  assert.deepStrictEqual(wordStatus, ['correct', 'pending', 'pending', 'correct']);
  assert.strictEqual(sinceMap.size, 0);
});

test('a repeated word never lights out of order, even after the delay', () => {
  // "the cat and the dog": the learner has said "the" once, so "the" exists
  // in the transcript permanently — a leftover/echoed copy of it is stable
  // evidence that would always outlive the confirmation delay. The second
  // "the" must stay pending while "cat"/"and" are unspoken, no matter how
  // long the phantom match persists.
  const tokens = ['the', 'cat', 'and', 'the', 'dog'];
  const wordStatus = ['correct', 'pending', 'pending', 'correct', 'pending'];
  // Simulate the phantom match having already survived far past the delay.
  const sinceMap = new Map([[3, 0]]);

  applyOutOfOrderConfirmation(wordStatus, tokens, 0, false, sinceMap, 60000);

  assert.deepStrictEqual(wordStatus, ['correct', 'pending', 'pending', 'pending', 'pending']);
  assert.strictEqual(sinceMap.has(3), false);
});

test('a repeated word lights instantly once the reading reaches it', () => {
  const tokens = ['the', 'cat', 'and', 'the', 'dog'];
  const wordStatus = ['correct', 'correct', 'correct', 'correct', 'pending'];
  const sinceMap = new Map();

  applyOutOfOrderConfirmation(wordStatus, tokens, 0, false, sinceMap, 1000);

  assert.deepStrictEqual(wordStatus, ['correct', 'correct', 'correct', 'correct', 'pending']);
  assert.strictEqual(sinceMap.size, 0);
});

test('a repeated word still lights on final results regardless of order', () => {
  const tokens = ['the', 'cat', 'and', 'the', 'dog'];
  const wordStatus = ['correct', 'pending', 'pending', 'correct', 'pending'];
  const sinceMap = new Map();

  applyOutOfOrderConfirmation(wordStatus, tokens, 0, true, sinceMap, 1000);

  assert.deepStrictEqual(wordStatus, ['correct', 'pending', 'pending', 'correct', 'pending']);
});

test('short words no longer fuzzy-match inside longer words', () => {
  // At Accuracy 50% the coverage rule used to score "a" as a perfect match
  // against ANY word containing the letter a — which is how the second "a"
  // in a sentence lit up green from leftover transcript tokens.
  const matches = findMatchesForTargetTokens(['a'], ['fait'], { langCode: 'fr' });
  assert.strictEqual(matches[0], null);

  const matchesEt = findMatchesForTargetTokens(['et'], ['le'], { langCode: 'fr' });
  assert.strictEqual(matchesEt[0], null);
});

test('recognizer-merged words still match their first target word', () => {
  // "bon dia" often comes back merged as "bondia" — the prefix exception
  // keeps that working despite the stricter length guard.
  const matches = findMatchesForTargetTokens(['bon', 'dia'], ['bondia'], { langCode: 'ca' });
  assert.notStrictEqual(matches[0], null);
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
