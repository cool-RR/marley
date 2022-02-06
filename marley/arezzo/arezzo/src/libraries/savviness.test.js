const formatSavvy = require('./savviness')


test('list0', () => {
  expect(
    formatSavvy(
      []
    )
  ).toBe(
    "[]"
  )
});
test('list1', () => {
  expect(
    formatSavvy(
      [1, 2]
    )
  ).toBe(
    "[1, 2]"
  )
});
test('list2', () => {
  expect(
    formatSavvy(
      [1, null, 'foo', [9]]
    )
  ).toBe(
    '[1, None, "foo", [9]]'
  )
});

test('tuple0', () => {
  expect(
    formatSavvy(
      {'tuple': [[]]}
    )
  ).toBe(
    "()"
  )
});
test('tuple1', () => {
  expect(
    formatSavvy(
      {'tuple': [[5]]}
    )
  ).toBe(
    "(5,)"
  )
});
test('tuple2', () => {
  expect(
    formatSavvy(
      {'tuple': [[5, 6, 'foo']]}
    )
  ).toBe(
    '(5, 6, "foo")'
  )
});

test('dict0', () => {
  expect(
    formatSavvy(
      {'dict': [[]]}
    )
  ).toBe(
    "{}"
  )
});
test('dict1', () => {
  expect(
    formatSavvy(
      {'dict': [[[0, 1]]]}
      )
  ).toBe(
    "{0: 1}"
  )
});
test('dict2', () => {
  expect(
    formatSavvy(
      {'dict': [[[0, 1], [2, 'foo']]]})
  ).toBe(
    '{0: 1, 2: "foo"}'
  )
});
test('dict3', () => {
  expect(
    formatSavvy(
      {
        'dict': [
          [
            [
              null,
              {
                'marley.worlds.blackjack.core.ThresholdPolicy':
                  ['31vdh9ru680v.0001000', 0]
              }
            ]
          ]
        ]
      }
    )
  ).toBe(
    '{None: marley.worlds.blackjack.core.ThresholdPolicy("31vdh9ru680v.0001000", 0)}'
  )
});

test('custom0', () => {
  expect(
    formatSavvy(
      {'hola.Senora': []})
  ).toBe(
    'hola.Senora()'
  )
});
test('custom1', () => {
  expect(
    formatSavvy(
      {'hola.Senora': [1, null, [2, 3]]})
  ).toBe(
    'hola.Senora(1, None, [2, 3])'
  )
});

test('bytes0', () => {
  expect(
    formatSavvy(
      {'base64.b64decode': ['']})
  ).toBe(
    'base64.b64decode("")'
  )
});
test('bytes1', () => {
  expect(
    formatSavvy(
      {'base64.b64decode': ['hi']})
  ).toBe(
    'base64.b64decode("hi")'
  )
});
