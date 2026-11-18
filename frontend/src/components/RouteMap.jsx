
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Polyline,
  Popup,
  useMap,
} from 'react-leaflet'

import { useEffect } from 'react'

// Automatically adjust the map when real route data arrives.
function FitRouteBounds({ gpsPoints, predictedRoute }) {
  const map = useMap()

  useEffect(() => {
    const coordinates = [
      ...gpsPoints,
      ...predictedRoute,
    ]

    if (coordinates.length > 0) {
      map.fitBounds(coordinates, { padding: [40, 40] })
    }
  }, [gpsPoints, predictedRoute, map])

  return null
}

function RouteMap({
  gpsPoints = [],
  predictedRoute = [],
}) {
  return (
    <div className="w-full h-full overflow-hidden rounded-xl">

      <MapContainer
        center={[20, 0]}
        zoom={2}
        scrollWheelZoom={true}
        className="w-full h-full"
      >

        {/* OpenStreetMap background */}
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <FitRouteBounds
          gpsPoints={gpsPoints}
          predictedRoute={predictedRoute}
        />

        {/* Original GPS observations */}
        {gpsPoints.map((point, index) => (
          <CircleMarker
            key={`gps-${index}`}
            center={point}
            radius={5}
            pathOptions={{
              color: '#60a5fa',
              fillColor: '#3b82f6',
              fillOpacity: 0.9,
            }}
          >
            <Popup>GPS Observation {index + 1}</Popup>
          </CircleMarker>
        ))}

        {/* Predicted route */}
        {predictedRoute.length > 1 && (
          <Polyline
            positions={predictedRoute}
            pathOptions={{
              color: '#22c55e',
              weight: 5,
              opacity: 0.9,
            }}
          >
            <Popup>Predicted Route</Popup>
          </Polyline>
        )}

      </MapContainer>

    </div>
  )
}

export default RouteMap
