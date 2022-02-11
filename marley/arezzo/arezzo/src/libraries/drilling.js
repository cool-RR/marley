function removeAsterisksFromDrillDown(drillDown) {
  let result = []
  drillDown.forEach(
    drill => {result.push(drill.replace('*', ''))}
  )
  return result
}
module.exports = removeAsterisksFromDrillDown;
