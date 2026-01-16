const test = require('node:test');
const assert = require('node:assert');

const {
  filterUnexpectedRepeats,
  tokenizeText,
} = require('./app.js');

test('drops repeated sequences once expected counts are met', () => {
  const targetTokens = tokenizeText('hi my name is marc');
  const spoken = 'hi my hi my hi my name is marc';

  const { filteredTokens } = filterUnexpectedRepeats(spoken, targetTokens);

  assert.deepStrictEqual(filteredTokens, tokenizeText('hi my name is marc'));
});

test('keeps legitimate consecutive duplicates but trims extras', () => {
  const targetTokens = tokenizeText('very very good job');
  const spoken = 'very very very good job';

  const { filteredTokens } = filterUnexpectedRepeats(spoken, targetTokens);

  assert.deepStrictEqual(filteredTokens, tokenizeText('very very good job'));
});

test('removes extra occurrences even when they are not consecutive', () => {
  const targetTokens = tokenizeText('to be or not to be');
  const spoken = 'to be or not to be to be';

  const { filteredTokens } = filterUnexpectedRepeats(spoken, targetTokens);

  assert.deepStrictEqual(filteredTokens, tokenizeText('to be or not to be'));
});

test('normalizes digit tokens to match spelled-out numbers', () => {
  const spelled = tokenizeText('I have two apples');
  const digits = tokenizeText('I have 2 apples');

  assert.deepStrictEqual(digits, spelled);
});

test('tokenizes digit sequences as number tokens', () => {
  assert.deepStrictEqual(tokenizeText('1 2 3 4 5'), ['1', '2', '3', '4', '5']);
});
