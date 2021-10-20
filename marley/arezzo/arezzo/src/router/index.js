import { createRouter, createWebHistory } from 'vue-router';
import Home from '../views/Home.vue';
import SwankTrail from '../views/SwankTrail.vue';
import About from '../views/About.vue';

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home,
  },
  {
    path: '/swank/:jamKindName/:jamId/:jamIndexString/:drillDown*',
    name: 'SwankTrail',
    props: true,
    component: SwankTrail,
  },
  {
    path: '/about',
    name: 'About',
    component: About,
  },
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes,
});

export default router;
