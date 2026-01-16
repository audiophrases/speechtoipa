const test = require('node:test');
const assert = require('node:assert');

const {
  filterUnexpectedRepeats,
  tokenizeText,
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
