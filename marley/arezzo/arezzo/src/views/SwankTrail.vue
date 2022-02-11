<template>
  <div>
    <div class="swank-trail">
      <Swank v-for="(squanchy, index) in squanchies" :key="squanchy.vueKey" :jamKindName="squanchy.jamKindName" :jamId="squanchy.jamId" :jamIndex="squanchy.jamIndex" :drill="squanchy.drill" :headDeasteriskedDrillDown="squanchy.headDeasteriskedDrillDown" :rootJamKindName="squanchies[0].jamKindName" :rootJamIndex="squanchies[0].jamIndex" :rootJamId="squanchies[0].jamId" :class="{is_active: index == activeSquanchyIndex, is_parent_and_button: index == activeSquanchyIndex - 1, is_parent_ancestor: index < activeSquanchyIndex - 1, is_child_and_button: index == activeSquanchyIndex + 1, is_child_descendant: index > activeSquanchyIndex + 1}" v-on="(index == activeSquanchyIndex - 1) ? {click: activateParent} : ((index == activeSquanchyIndex + 1) ? { click: activateChild} : {})" />
    </div>
  </div>
</template>

<script>
import _ from 'lodash';

import JamService from '@/services/JamService.js'
import removeAsterisksFromDrillDown from '@/libraries/drilling.js'
import Swank from '@/components/Swank.vue';

function zip(arrays) {
  return arrays[0].map(function(_,i){
    return arrays.map(function(array){return array[i]})
  });
}

Array.prototype.equals = function (array) {
    if (!array)
        return false;

    if (this.length != array.length)
        return false;

    for (var i = 0, l=this.length; i < l; i++) {
        // Check if we have nested arrays
        if (this[i] instanceof Array && array[i] instanceof Array) {
            // recurse into the nested arrays
            if (!this[i].equals(array[i]))
                return false;
        }
        else if (this[i] != array[i]) {
            // Warning - two different object instances will never be equal: {x:20} != {x:20}
            return false;
        }
    }
    return true;
}

// Hide method from for-in loops
Object.defineProperty(Array.prototype, 'equals', {enumerable: false});


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
    jamTriplet() {
      return [this.jamKindName, this.jamId, this.jamIndex]
    },
    jamTripletAndCleanDrilldown() {
      return [this.jamKindName, this.jamId, this.jamIndex, this.cleanDrillDown]
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
    this.initialize()
  },
  watch: {
    jamTripletAndCleanDrilldown(newJamTripletAndCleanDrilldown, oldJamTripletAndCleanDrilldown) {
      let oldJamTriplet = oldJamTripletAndCleanDrilldown.slice(0, 3)
      let newJamTriplet = newJamTripletAndCleanDrilldown.slice(0, 3)
      let oldCleanDrillDown = oldJamTripletAndCleanDrilldown[3]
      let newCleanDrillDown = newJamTripletAndCleanDrilldown[3]

      if (!oldJamTriplet.equals(newJamTriplet)) {
        this.initialize()
      } else {
        if (!oldCleanDrillDown.equals(newCleanDrillDown)) {
          this.handleCleanDrillDownChange(newCleanDrillDown, oldCleanDrillDown)
        }
      }
    }
  },
  methods: {
    initialize() {
      this.squanchies = [
        {
          jamKindName: this.jamKindName,
          jamId: this.jamId,
          jamIndex: this.jamIndex,
          drill: null,
          headDeasteriskedDrillDown: [],
          vueKey: this.jamKindName + '/' + this.jamId + '[' + this.jamIndex + ']',
        }
      ]
      this.handleCleanDrillDownChange(this.drillDown, [])
    },
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
        squanchy.headDeasteriskedDrillDown = removeAsterisksFromDrillDown(
          this.drillDown.slice(0, i)
        )
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
      let lastSquanchy, jam, fieldName, fullFieldName, fieldValue, jamIndex, match
      let headDeasteriskedDrillDown
      lastSquanchy = this.squanchies[this.squanchies.length - 1]
      lastSquanchy.drill = drill
      let jamPromise = JamService.getJam(lastSquanchy.jamKindName, lastSquanchy.jamId,
                                         lastSquanchy.jamIndex)
      jamPromise.then(
        response => {
          jam = response.data
          headDeasteriskedDrillDown = removeAsterisksFromDrillDown(
            Array.prototype.concat(lastSquanchy.headDeasteriskedDrillDown, [drill])
          )
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
                headDeasteriskedDrillDown: headDeasteriskedDrillDown,
                vueKey: fieldValue[0] + '/' + fieldValue[1] + '[' + jamIndex + ']',
              }
            )
          } else {
            fieldName = drill.endsWith('*') ? drill.slice(0, -1) : drill
            fullFieldName = fieldName + '.swank'
            fieldValue = jam[fullFieldName]
            this.squanchies.push(
              {
                jamKindName: fieldValue[0],
                jamId: fieldValue[1],
                jamIndex: fieldValue[2],
                drill: null,
                headDeasteriskedDrillDown: headDeasteriskedDrillDown,
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