import { useEffect, useMemo } from 'react'
import { CircleMarker, MapContainer, Polyline, TileLayer, useMap } from 'react-leaflet'

const DEFAULT_CENTER = [12.9716, 77.5946]

function toPosition(coordinate) {
  if (!Array.isArray(coordinate) || coordinate.length < 2) return null
  const [lng, lat] = coordinate.map(Number)
  return Number.isFinite(lat) && Number.isFinite(lng) && Math.abs(lat) <= 90 && Math.abs(lng) <= 180
    ? [lat, lng]
    : null
}

function FitRoute({ positions }) {
  const map = useMap()
  useEffect(() => {
    const timer = window.setTimeout(() => map.invalidateSize(), 0)
    if (positions.length > 1) map.fitBounds(positions, { padding: [36, 36], maxZoom: 16 })
    else if (positions.length === 1) map.setView(positions[0], 15)
    return () => window.clearTimeout(timer)
  }, [map, positions])
  return null
}

export default function RouteMapLeaflet({ gpsPoints = [], predictedRoute = [] }) {
  const gps = useMemo(() => gpsPoints.map(toPosition).filter(Boolean), [gpsPoints])
  const route = useMemo(() => predictedRoute
    .flatMap((segment) => Array.isArray(segment?.coordinates) ? segment.coordinates : [])
    .map(toPosition).filter(Boolean), [predictedRoute])
  const positions = useMemo(() => [...gps, ...route], [gps, route])

  return (
    <MapContainer center={gps[0] ?? route[0] ?? DEFAULT_CENTER} zoom={13} scrollWheelZoom className="h-full min-h-[420px] w-full bg-[#e5e7eb]" style={{ height: '100%', width: '100%' }}>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <FitRoute positions={positions} />
      {gps.map((point, index) => <CircleMarker key={`gps-${index}`} center={point} radius={4} pathOptions={{ color: '#fff', weight: 1, fillColor: '#3b82f6', fillOpacity: 0.95 }} />)}
      {route.length > 1 && <Polyline positions={route} pathOptions={{ color: '#8ac926', weight: 6, opacity: 0.95 }} />}
    </MapContainer>
  )
}
