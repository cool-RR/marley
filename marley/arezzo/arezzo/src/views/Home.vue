<template>
  <h2>Favorites:</h2>
  <ul>
    <li v-for="favorite in favorites">
      <router-link :to="{ name: 'SwankTrail', params: { jamKindName: favorite.split('/')[0], jamId: favorite.split('/')[1], jamIndexString: favorite.split('/')[2], drillDown: favorite.replace(/(.*?\/){3}/, '').split('/')}}">
        {{ favorite }}
      </router-link>
    </li>
  </ul>
</template>

<script>
import ArezzoService from '@/services/ArezzoService.js'

export default {
  name: 'Home',
  data() {
    return {
      favorites: [],
    }
  },
  created() {
    ArezzoService.getFavorites()
    .then(response => {
      this.favorites = response.data
    })
    .catch(error => {
      console.log(error)
    })
  },
  components: {
  },
};

</script>

<style scoped>
ul {
  list-style-type: none;
}

li {
  margin: 10px;
}
</style>