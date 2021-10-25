import axios from 'axios'

const apiClient = axios.create({
  baseURL: 'http://127.0.0.1:22390/api/',
  withCredentials: false,
  headers: {
    Accept: 'application/json',
    'Content-Type': 'application/json'
  }
})


function getFavorites() {
  return apiClient.get('/favorites')
}

function addFavorite(favorite) {
  return apiClient.post('/favorites/' + favorite)
}

function deleteFavorite(favorite) {
  return apiClient.delete('/favorites/' + favorite)
}

export default {
  getFavorites: getFavorites,
  addFavorite: addFavorite,
  deleteFavorite: deleteFavorite,
}