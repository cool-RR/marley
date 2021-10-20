import axios from 'axios'

const apiClient = axios.create({
  baseURL: '/api/jam/',
  withCredentials: false,
  headers: {
    Accept: 'application/json',
    'Content-Type': 'application/json'
  }
})

//function getJams(jam_kind_name, jam_id, jam_index_start, jam_index_end) {
  //return apiClient.get(`/${jam_kind_name}/${jam_id}/${jam_index_start}..${jam_index_end}`)
//}

function getJam(jam_kind_name, jam_id, jam_index) {
  return apiClient.get(`/${jam_kind_name}/${jam_id}/${jam_index}..${jam_index + 1}`)
}

function getParchmentInfo(jam_kind_name, jam_id) {
  return apiClient.get(`/${jam_kind_name}/${jam_id}`)
}


export default {
  getJam: getJam,
  getParchmentInfo: getParchmentInfo,
}