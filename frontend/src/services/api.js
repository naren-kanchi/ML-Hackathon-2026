
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
export async function checkBackendHealth() {
  try {
    const response = await api.get('/health')
    return response.data
  } catch (error) {
    console.error('Backend health check failed:', error)
    throw error
  }
}

// Retrieve the trip IDs available to the server-side evaluator.
export async function fetchTrips() {
  try {
    const response = await api.get('/api/v1/trips')
    return response.data
  } catch (error) {
    console.error('Trip list request failed:', error)
    throw error
  }
}

// Inference is keyed by a trip ID; ground-truth route files stay server-side.
export async function requestPrediction(tripId) {
  if (!/^\d{8}$/.test(tripId)) {
    throw new Error('Trip ID must contain exactly 8 digits')
  }
  try {
    const response = await api.get(`/api/v1/trips/${encodeURIComponent(tripId)}`)
    return response.data
  } catch (error) {
    console.error('Prediction request failed:', error)
    throw error
  }
}

export default api
