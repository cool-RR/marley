<template>
  <div v-if="fieldTypeName == 'simple'" class="simple-field">
    {{ formattedSimpleValue }}
  </div>

  <div v-else-if="fieldTypeName == 'savvy'" class="swank-field">
    {{ formattedSavvy }}
  </div>

  <div v-else-if="fieldTypeName == 'swank'" class="swank-ref-field">
    <button @click="goToSwank()">
      {{ swankFieldShortJamKindName }}
    </button>
  </div>

  <div v-else-if="fieldTypeName == 'parchment'" class="parchment-ref-field">
    <button @click="parchmentFieldShowDialog()">
      <img v-if="parchmentFieldLength === null" class="loading" src="@/assets/loading.gif" />
      <span v-else>
        {{ parchmentFieldLength }}x
      </span>

      {{ parchmentFieldShortJamKindName }}
    </button>
  </div>

  <div v-else>Error: Field type not recognized</div>
</template>

<script>
import JamService from '@/services/JamService.js'
import formatSavvy from '@/libraries/savviness.js'

export default {
  name: 'SwankFieldValue',
  components: {
  },
  props: {
    'rootJamKindName': String,
    'rootJamId': String,
    'rootJamIndex': Number,
    'drill': String,
    'parentDrillDown': Array,
    'fieldName': String,
    'fieldTypeName': String,
    'value': {
      required: true,
    },
  },
  data() {
    return {
      parchmentFieldLength: null,
    }
  },
  watch: {
    value: {
      handler (newValue, oldValue) {
        if (this.fieldTypeName == 'parchment') {
          if (this.parchmentFieldEndIndex === null) {
            JamService.getParchmentInfo(
              this.parchmentFieldJamKindName,
              this.parchmentFieldJamId
            ).then(
              response => {
                this.parchmentFieldLength = response.data.length
              }
            )
          } else {
            // This parchment field is bounded.
            this.parchmentFieldLength =
              this.parchmentFieldEndIndex - this.parchmentFieldStartIndex
          }
        }
      },
      immediate: true,
    }
  },
  computed: {
    parchmentFieldShortJamKindName() {
      let split_name = this.parchmentFieldJamKindName.split('.')
      return split_name[split_name.length - 1]
    },
    parchmentFieldJamKindName() {
      if (this.fieldTypeName == 'parchment') {
        return this.value[0]
      }
    },
    parchmentFieldJamId() {
      if (this.fieldTypeName == 'parchment') {
        return this.value[1]
      }
    },
    parchmentFieldStartIndex() {
      if (this.fieldTypeName == 'parchment') {
        return this.value[2]
      }
    },
    parchmentFieldEndIndex() {
      if (this.fieldTypeName == 'parchment') {
        return this.value[3]
      }
    },
    swankFieldShortJamKindName() {
      if (this.fieldTypeName == 'swank') {
        let split_jam_kind_name = this.value[0].split('.')
        return split_jam_kind_name[split_jam_kind_name.length - 1]
      }
    },
    formattedSavvy() {
      if (this.fieldTypeName == 'savvy') {
        return formatSavvy(this.value)
      }
    },
    formattedSimpleValue() {
      if (this.fieldTypeName == 'simple') {
        if (typeof(this.value) == 'string') {
          return JSON.stringify(this.value)
        } else {
          return this.value
        }
      }
    },
  },
  methods: {
    parchmentFieldShowDialog() {
      let i = null
      while ((i === null) || (isNaN(i)) || (i < 0) || (i >= this.parchmentFieldLength)) {
        i = parseInt(
          window.prompt('Choose ' + this.parchmentFieldShortJamKindName +
                        ' number, from 0 to ' + (this.parchmentFieldLength - 1))
        )
      }
      let newDrillDown = this.parentDrillDown.slice()
      if (newDrillDown.length) {
        let lastDrill = newDrillDown.pop()
        newDrillDown.push(lastDrill.slice(0, -1))
      }
      newDrillDown.push(this.fieldName + '[' + i + ']*')
      this.$router.push(
        {
          name: 'SwankTrail',
          params: {
            jamKindName: this.rootJamKindName,
            jamId: this.rootJamId,
            jamIndex: this.rootJamIndex,
            drillDown: newDrillDown
          }
        }
      )
    },
    goToSwank() {
      let newDrillDown = this.parentDrillDown.slice()
      if (newDrillDown.length) {
        let lastDrill = newDrillDown.pop()
        newDrillDown.push(lastDrill.slice(0, -1))
      }
      newDrillDown.push(this.fieldName + '*')
      this.$router.push(
        {
          name: 'SwankTrail',
          params: {
            jamKindName: this.rootJamKindName,
            jamId: this.rootJamId,
            jamIndex: this.rootJamIndex,
            drillDown: newDrillDown
          }
        }
      )
    }
  }
};
</script>

<style scoped>
img.loading {
  width: 30px;
  height: auto;
  position: relative;
  bottom: -3px
}

.is_drilled {
  border: 2px solid red;
}
</style>
