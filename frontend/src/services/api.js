
import axios from 'axios'

// Backend base URL
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// Create Axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Check backend health
export async function checkBackendHealth(endpoint) {
  try {
    const response = await api.get(endpoint)
    return response.data
  } catch (error) {
    console.error('Backend health check failed:', error)
    throw error
  }
}

// Send prediction request
export async function requestPrediction(endpoint, payload) {
  try {
    const response = await api.post(endpoint, payload)
    return response.data
  } catch (error) {
    console.error('Prediction request failed:', error)
    throw error
  }
}

export default api
