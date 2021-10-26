<template>
  <div v-if="fieldTypeName == 'simple'" class="simple-field">
    {{ value }}
  </div>

  <div v-else-if="fieldTypeName == 'savvy'" class="swank-field">
    Savvy: {{ value }}
  </div>

  <div v-else-if="fieldTypeName == 'swank'" class="swank-ref-field">
    Swank: {{ value }}
  </div>

  <div v-else-if="fieldTypeName == 'parchment'" class="parchment-ref-field">
    <button @click="parchmentRefFieldShowDialog()">
      <img v-if="parchmentRefFieldLength === null" class="loading" src="@/assets/loading.gif" />
      <span v-else>
        {{ parchmentRefFieldLength }}x
      </span>

      {{ parchmentRefFieldShortJamKindName }}
    </button>
  </div>

  <div v-else>Error: Field type not recognized</div>
</template>

<script>
import JamService from '@/services/JamService.js'

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
      parchmentRefFieldLength: null,
    }
  },
  watch: {
    value: {
      handler (newValue, oldValue) {
        if (this.fieldTypeName == 'parchment') {
          if (this.parchmentRefFieldEndIndex === null) {
            JamService.getParchmentInfo(
              this.parchmentRefFieldJamKindName,
              this.parchmentRefFieldJamId
            ).then(
              response => {
                this.parchmentRefFieldLength = response.data.length
              }
            )
          } else {
            // This parchment field is bounded.
            this.parchmentRefFieldLength =
              this.parchmentRefFieldEndIndex - this.parchmentRefFieldStartIndex
          }
        }
      },
      immediate: true,
    }
  },
  computed: {
    parchmentRefFieldShortJamKindName() {
      let split_name = this.parchmentRefFieldJamKindName.split('.')
      return split_name[split_name.length - 1]
    },
    parchmentRefFieldJamKindName() {
      if (this.fieldTypeName == 'parchment') {
        return this.value[0]
      }
    },
    parchmentRefFieldJamId() {
      if (this.fieldTypeName == 'parchment') {
        return this.value[1]
      }
    },
    parchmentRefFieldStartIndex() {
      if (this.fieldTypeName == 'parchment') {
        return this.value[2]
      }
    },
    parchmentRefFieldEndIndex() {
      if (this.fieldTypeName == 'parchment') {
        return this.value[3]
      }
    },
  },
  methods: {
    parchmentRefFieldShowDialog() {
      let i = null
      while ((i === null) || (isNaN(i)) || (i < 0) || (i >= this.parchmentRefFieldLength)) {
        i = parseInt(
          window.prompt('Choose ' + this.parchmentRefFieldShortJamKindName +
                        ' number, from 0 to ' + (this.parchmentRefFieldLength - 1))
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
