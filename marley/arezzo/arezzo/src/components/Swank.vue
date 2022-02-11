<template>
  <div class="swank">
    <table class="top-bar">
      <tr>
        <td>
          <div class="title">
            <router-link :to="{name: 'SwankTrail', params: {'jamKindName': jamKindName, 'jamId': jamId, 'jamIndex': jamIndex}}">
              {{ shortJamKindName}}/<b>{{ jamId }}</b>[{{ jamIndex }}]
            </router-link>
            <br/>
          </div>
        </td>
        <td style="text-align: right;">
          <a class="favorite" @click="toggleFavorite()">
            <img :src="favoriteImage" />
          </a>
        </td>
      </tr>
    </table>
    <table>
      <tr v-for="(value, key) in jam" :key="key">
        <td>
          {{ key.split('.')[0] }}:
        </td>
        <td>
          <SwankFieldValue :key="key.split('.')[0]" :fieldName="key.split('.')[0]" :rootJamKindName="rootJamKindName" :rootJamId="rootJamId" :rootJamIndex="rootJamIndex" :headDeasteriskedDrillDown="headDeasteriskedDrillDown" :value="value" :fieldTypeName="key.split('.')[1]" :class="{is_drilled: key == drillFullFieldName}" />
        </td>
      </tr>
    </table>
  </div>
</template>

<script>
import _ from 'lodash';

import JamService from '@/services/JamService.js'
import ArezzoService from '@/services/ArezzoService.js'
import SwankFieldValue from '@/components/SwankFieldValue.vue'

function containsObject(list, object) {
    var i;
    for (i = 0; i < list.length; i++) {
        if (_.isEqual(list[i], obj)) {
            return true;
        }
    }
    return false;
}

export default {
  name: 'Swank',
  components: {
    SwankFieldValue,
  },
  props: {
    'rootJamKindName': String,
    'rootJamId': String,
    'rootJamIndex': Number,

    'jamKindName': String,
    'jamId': String,
    'jamIndex': Number,

    'drill': String,
    'headDeasteriskedDrillDown': Array,
  },
  data() {
    return {
      jam: null,
      isFavorite: null
    }
  },
  created() {
    JamService.getJam(this.jamKindName, this.jamId, this.jamIndex)
    .then(response => {
      let jam = response.data
      this.jam = jam
    })
    .catch(error => {
      console.log(error)
    })

    this.updateFavorite()
  },
  computed: {
    favoriteEntry() {
      return (
        this.rootJamKindName + '/' + this.rootJamId + '/' + this.rootJamIndex + '/' +
        this.headDeasteriskedDrillDown.join('/') +
        (this.headDeasteriskedDrillDown.length ? '*' : '')
      )
    },
    favoriteImage() {
      if (this.isFavorite === null) {
        return require('@/assets/loading.gif')
      }
      else if (this.isFavorite) {
        return require('@/assets/full_star.png')
      }
      else {
        return require('@/assets/empty_star.png')
      }
    },
    drillFieldName() {
      return this.drill.split('[')[0]
    },
    drillFullFieldName() {
      if ((this.drill === null) || (this.drill === undefined)) {
        return null
      } else {
        let fieldType = (this.drill.includes('[') ? 'parchment' : 'swank')
        return this.drillFieldName + '.' + fieldType
      }
    },
    shortJamKindName() {
      let split_name = this.jamKindName.split('.')
      return split_name[split_name.length - 1]
    }
  },
  methods: {
    updateFavorite() {
      ArezzoService.getFavorites()
      .then(response => {
        let favorites = response.data
        this.isFavorite = favorites.includes(this.favoriteEntry)
      })
      .catch(error => {
        console.log(error)
      })
    },
    toggleFavorite(event) {
      let request
      if (this.isFavorite === null) {
        return
      }
      if (this.isFavorite === true) {
        this.isFavorite = null
        request = ArezzoService.deleteFavorite(this.favoriteEntry)
      }
      else if (this.isFavorite === false) {
        this.isFavorite = null
        request = ArezzoService.addFavorite(this.favoriteEntry)
      }
      request.then(response => {this.updateFavorite()})
             .catch(error => {console.log(error)})

    },
  }
};
</script>

<style scoped>
.swank {
  border: 1px solid #39495c;
  margin-bottom: 18px;
  background: black;
  transition: all 0.5s linear;
  overflow-x: hidden;
  overflow-y: auto;
  height: 100%;
}

.swank.is_active {
  flex: 1;
  margin-left: 10px;
  margin-right: 10px;
}

.swank:not(.is_active) > * {
  display: none;
}

.swank.is_parent_ancestor, .swank.is_child_descendant {
  opacity: 0;
  flex: 0.0001;
}

.swank.is_parent_ancestor, .swank.is_parent_and_button {
}

.swank.is_child_descendant, .swank.is_child_and_button {
}

.swank.is_child_and_button, .swank.is_parent_and_button {
  cursor: pointer;
  opacity: 0.6;
  flex: 0.01 30px;
}

.swank.is_parent_ancestor, .swank.is_child_descendant, .swank.is_child_and_button, .swank.is_parent_and_button {
  width: 30px;
}

table {
  text-align: left;
}

td:first-child {
  font-family: consolas, "Liberation Mono", courier, monospace;
}

a.favorite {
  cursor: pointer;
}
a.favorite img {
  width: 30px;
  height: 30px;
}

table.top-bar {
  width: 100%
}

</style>
