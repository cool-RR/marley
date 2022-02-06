function formatSavvy(savvy) {
  let key, value, entries, formattedValue, fluffs, inner
  console.log('Starting formatSavvy:', typeof(savvy), savvy)

  if (savvy === null) {
    return 'None'
  } else if (typeof(savvy) === 'string') {
    return JSON.stringify(savvy)
  } else if (typeof(savvy) === 'number') {
    return savvy.toString()
  } else if (Array.isArray(savvy)) {
      fluffs = savvy.map(formatSavvy)
      inner = fluffs.join(', ')
      return '[' + inner + ']'
  } else if (typeof(savvy) === 'object') {
    entries = Object.entries(savvy)
    key = entries['0'][0]
    value = entries['0'][1]
    if (key == 'dict') {
      fluffs = value[0].map(item => (formatSavvy(item[0]) + ': ' + formatSavvy(item[1])))
      inner = fluffs.join(', ')
      return '{' + inner + '}'
    } else if (key == 'tuple') {
      if (value[0].length == 0) {
        return '()'
      } else if (value[0].length == 1) {
        return '(' + formatSavvy(value[0][0]) + ',)'
      } else {
        fluffs = value[0].map(formatSavvy)
        inner = fluffs.join(', ')
        return '(' + inner + ')'
      }
    } else if (key == 'bytes') {
      return 'b' + JSON.stringify(value[0])
    } else {
      inner = value.map(formatSavvy).join(', ')
      return key + '(' + inner + ')'
    }
  } else {
    console.log('Unrecognized type:', typeof(savvy), savvy)
    return 'Unrecognized type:' + typeof(savvy) + savvy
  }
}
module.exports = formatSavvy;
