<template>
  <div id="nav">
    <router-link :to="{ name: 'Home' }">Marley</router-link>
  </div>
  <router-view />
</template>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #B7B7B7;
}

#nav {
  padding: 5px;
}

#nav a {
  text-decoration: none;
  font-size: 35px;
  color: #445577;
}

#nav a.router-link-exact-active {
  color: #42b983;
}

h4 {
  font-size: 20px;
}

button {
  filter: invert(1);
}

body {
  background: black;
}

a {
  color: #79b
}

</style>
