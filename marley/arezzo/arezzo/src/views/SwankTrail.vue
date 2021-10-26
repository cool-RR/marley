<template>
  <div>
    <div class="swank-trail">
      <Swank v-for="(squanchy, index) in squanchies" :key="squanchy.vueKey" :ass="0" :jamKindName="squanchy.jamKindName" :jamId="squanchy.jamId" :jamIndex="squanchy.jamIndex" :drill="squanchy.drill" :parentDrillDown="squanchy.parentDrillDown" :rootJamKindName="squanchies[0].jamKindName" :rootJamIndex="squanchies[0].jamIndex" :rootJamId="squanchies[0].jamId" :class="{is_active: index == activeSquanchyIndex, is_parent_and_button: index == activeSquanchyIndex - 1, is_parent_ancestor: index < activeSquanchyIndex - 1, is_child_and_button: index == activeSquanchyIndex + 1, is_child_descendant: index > activeSquanchyIndex + 1}" v-on="(index == activeSquanchyIndex - 1) ? {click: activateParent} : ((index == activeSquanchyIndex + 1) ? { click: activateChild} : {})" />
    </div>
  </div>
</template>

<script>
import _ from 'lodash';

import JamService from '@/services/JamService.js'
import Swank from '@/components/Swank.vue';

function zip(arrays) {
  return arrays[0].map(function(_,i){
    return arrays.map(function(array){return array[i]})
  });
}

function getLengthOfMatchingHead(x, y) {
  let zipped = zip([x || [], y || []])
  let zip_item
  let i
  for (i = 0; i < zipped.length; i++) {
    zip_item = zipped[i]
    if (!(_.isEqual(zip_item[0], zip_item[1]))) {
      return i
    }
  }
  return i
}

function removeAsterisksFromDrillDown(drillDown) {
  let result = []
  drillDown.forEach(
    drill => {result.push(drill.replace('*', ''))}
  )
  return result
}



export default {
  name: 'SwankTrail',
  components: {
    Swank,
  },
  props: {
  'jamKindName': String,
  'jamId': String,
  'jamIndexString': String,
  'drillDown': [Array, String],
  },
  computed: {
    jamIndex() {
      return parseInt(this.jamIndexString)
    },
    activeSquanchyIndex() {
      let drillDown = this.drillDown || []
      for (let i = 0; i < drillDown.length; i++) {
        if (drillDown[i].endsWith('*')) {
          return i + 1
        }
      }
      return 0
    },
    cleanDrillDown() {
      return removeAsterisksFromDrillDown(this.drillDown || [])
    }
  },
  data() {
    return {
      wipDrillDown: null,
      squanchies: [],
    }
  },
  created() {
    this.squanchies = [
      {
        jamKindName: this.jamKindName,
        jamId: this.jamId,
        jamIndex: this.jamIndex,
        drill: null,
        parentDrillDown: [],
        vueKey: this.jamKindName + '/' + this.jamId + '[' + this.jamIndex + ']',
      }
    ]
    this.handleCleanDrillDownChange(this.drillDown, [])
  },
  watch: {
    cleanDrillDown(newCleanDrillDown, oldCleanDrillDown) {
      this.handleCleanDrillDownChange(newCleanDrillDown, oldCleanDrillDown)
    },
  },
  methods: {
    handleCleanDrillDownChange(newCleanDrillDown, oldCleanDrillDown) {
      let lengthOfMatchingHead = getLengthOfMatchingHead(newCleanDrillDown, oldCleanDrillDown)
      this.startDrillDown(lengthOfMatchingHead)
    },
    startDrillDown(lengthOfMatchingHead) {
      let nSquanchiesToPop = this.squanchies.length - lengthOfMatchingHead - 1
      let i
      for (i = 0; i < nSquanchiesToPop; i++) {
        this.squanchies.pop()
      }
      for (i = 1; i < this.squanchies.length; i++) {
        let squanchy = this.squanchies[i]
        squanchy.parentDrillDown = this.drillDown.slice(0, i)
        squanchy.drill = this.drillDown[i]
      }
      this.wipDrillDown = (this.drillDown || []).slice(lengthOfMatchingHead)
      this.continueDrillDown()
    },
    continueDrillDown() {
      if (this.wipDrillDown.length == 0) {
        return
      }
      let drill = this.wipDrillDown.shift()
      let indexedDrillRegex = /^(.*)\[([0-9]+)\]\*?$/
      let lastSquanchy
      let jam
      let fieldName
      let fullFieldName
      let fieldValue
      let jamIndex
      let match
      let parentDrillDown
      lastSquanchy = this.squanchies[this.squanchies.length - 1]
      lastSquanchy.drill = drill
      let jamPromise = JamService.getJam(lastSquanchy.jamKindName, lastSquanchy.jamId,
                                         lastSquanchy.jamIndex)
      jamPromise.then(
        response => {
          jam = response.data
          parentDrillDown = Array.prototype.concat(lastSquanchy.parentDrillDown, [drill])
          match = indexedDrillRegex.exec(drill)

          if (match) {
            fieldName = match[1]
            fullFieldName = fieldName + '.parchment'
            fieldValue = jam[fullFieldName]
            jamIndex = parseInt(match[2])
            this.squanchies.push(
              {
                jamKindName: fieldValue[0],
                jamId: fieldValue[1],
                jamIndex: jamIndex + fieldValue[2],
                drill: null,
                parentDrillDown: parentDrillDown,
                vueKey: fieldValue[0] + '/' + fieldValue[1] + '[' + jamIndex + ']',
              }
            )
          } else {
            fieldName = drill
            fullFieldName = fieldName + '.swank'
            fieldValue = jam[fullFieldName]
            this.squanchies.push(
              {
                jamKindName: fieldValue[0],
                jamId: fieldValue[1],
                jamIndex: fieldValue[2],
                drill: null,
                parentDrillDown: parentDrillDown,
                vueKey: fieldValue[0] + '/' + fieldValue[1] + '[' + fieldValue[2] + ']',
              }
            )
          }
          setTimeout(this.continueDrillDown)
        }
      )
    },
    activateChild(event) {
      let newActiveSquanchyIndex = this.activeSquanchyIndex + 1
      let newDrillDown = this.drillDown.slice()
      newDrillDown[newActiveSquanchyIndex - 1] += '*'
      if (newActiveSquanchyIndex >= 2) {
        newDrillDown[newActiveSquanchyIndex - 2] =
          newDrillDown[newActiveSquanchyIndex - 2].slice(0, -1)
      }
      this.$router.push(
        {
          name: 'SwankTrail',
          params: {
            jamKindName: this.jamKindName,
            jamId: this.jamId,
            jamIndexString: this.jamIndexString,
            drillDown: newDrillDown,
          }
        }
      )
    },
    activateParent(event) {
      let newActiveSquanchyIndex = this.activeSquanchyIndex - 1
      let newDrillDown = this.drillDown.slice()
      newDrillDown[newActiveSquanchyIndex] = newDrillDown[newActiveSquanchyIndex].slice(0, -1)
      if (newActiveSquanchyIndex >= 1) {
        newDrillDown[newActiveSquanchyIndex - 1] += '*'
      }
      this.$router.push(
        {
          name: 'SwankTrail',
          params: {
            jamKindName: this.jamKindName,
            jamId: this.jamId,
            jamIndexString: this.jamIndexString,
            drillDown: newDrillDown,
          }
        }
      )
    },
  }
};

</script>

<style scoped>
.swank-trail {
  display: flex;
  flex-direction: row;
  align-items: center;
  position: absolute;
  top: 80px;
  bottom: 80px;
  left: 0px;
  right: 0px;
}
</style>