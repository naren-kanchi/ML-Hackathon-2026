import { useEffect, useMemo, useRef, useState } from 'react'
import { LngLatBounds, Map as MapLibreMap, NavigationControl } from 'maplibre-gl'

const EMPTY_DATA = []
const FALLBACK_STYLE = {
  version: 8,
  sources: {
    osm: {
      type: 'raster',
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      attribution: '© OpenStreetMap contributors',
    },
  },
  layers: [{ id: 'osm-raster', type: 'raster', source: 'osm' }],
}

function featureCollection(features = []) {
  return { type: 'FeatureCollection', features }
}

function updateSources(map, gpsPoints, routeLine) {
  map.getSource('pathsnap-gps')?.setData(featureCollection(gpsPoints.map((coordinates) => ({
    type: 'Feature',
    geometry: { type: 'Point', coordinates },
    properties: {},
  }))))
  map.getSource('pathsnap-predicted-route')?.setData(featureCollection(routeLine.length > 1 ? [{
    type: 'Feature',
    geometry: { type: 'LineString', coordinates: routeLine },
    properties: {},
  }] : []))

  const boundsPoints = [...gpsPoints, ...routeLine]
  if (boundsPoints.length > 1) {
    const bounds = boundsPoints.reduce((result, coordinate) => result.extend(coordinate),
      new LngLatBounds(boundsPoints[0], boundsPoints[0]))
    map.fitBounds(bounds, { padding: 50, maxZoom: 16, duration: 500 })
  } else if (boundsPoints.length === 1) {
    map.easeTo({ center: boundsPoints[0], zoom: 15, duration: 500 })
  }
}

export default function RouteMap({ gpsPoints = EMPTY_DATA, predictedRoute = EMPTY_DATA }) {
  const mapContainer = useRef(null)
  const mapRef = useRef(null)
  const dataRef = useRef({ gpsPoints, routeLine: [] })
  const [mapError, setMapError] = useState('')
  const [styleReady, setStyleReady] = useState(false)
  const routeLine = useMemo(() => {
    const coordinates = []
    predictedRoute.forEach((segment) => {
      segment.coordinates?.forEach((coordinate) => {
        const previous = coordinates.at(-1)
        if (!previous || previous[0] !== coordinate[0] || previous[1] !== coordinate[1]) {
          coordinates.push(coordinate)
        }
      })
    })
    return coordinates
  }, [predictedRoute])
  dataRef.current = { gpsPoints, routeLine }

  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return undefined
    const map = new MapLibreMap({
      container: mapContainer.current,
      // Inline style avoids a separate style JSON request. OSM raster tiles are
      // widely reachable; the route overlays work with either this or OpenFreeMap.
      style: FALLBACK_STYLE,
      center: gpsPoints[0] ?? routeLine[0] ?? [77.5946, 12.9716],
      zoom: 15,
      pitch: 0,
      bearing: 0,
    })
    mapRef.current = map
    map.addControl(new NavigationControl({ visualizePitch: true }), 'top-right')
    map.on('error', (event) => {
      const message = event.error?.message ?? 'The map style or map tiles could not be loaded.'
      console.error('MapLibre error:', event.error ?? event)
      setMapError(message)
    })
    map.on('load', () => {
      setStyleReady(true)
      setMapError('')
      map.addSource('pathsnap-gps', { type: 'geojson', data: featureCollection() })
      map.addLayer({
        id: 'pathsnap-gps-points',
        type: 'circle',
        source: 'pathsnap-gps',
        paint: {
          'circle-radius': 4,
          'circle-color': '#60a5fa',
          'circle-stroke-color': '#0f172a',
          'circle-stroke-width': 1,
        },
      })
      map.addSource('pathsnap-predicted-route', { type: 'geojson', data: featureCollection() })
      map.addLayer({
        id: 'pathsnap-predicted-line',
        type: 'line',
        source: 'pathsnap-predicted-route',
        paint: { 'line-color': '#22c55e', 'line-width': 5, 'line-opacity': 0.9 },
      })
      updateSources(map, dataRef.current.gpsPoints, dataRef.current.routeLine)
      // The map sits in a dashboard panel; force a layout pass after its style loads.
      requestAnimationFrame(() => map.resize())
    })
    const resizeObserver = new ResizeObserver(() => map.resize())
    resizeObserver.observe(mapContainer.current)
    return () => {
      resizeObserver.disconnect()
      map.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (map?.isStyleLoaded() && map.getSource('pathsnap-gps')) {
      updateSources(map, gpsPoints, routeLine)
    }
  }, [gpsPoints, routeLine])

  return (
    <div className="relative w-full h-full min-h-[420px] rounded-2xl overflow-hidden bg-[#0b1220]">
      <div ref={mapContainer} className="absolute inset-0" />
      {mapError && (
        <div className="absolute inset-x-4 bottom-4 z-10 rounded-lg border border-amber-400/30 bg-slate-950/90 px-4 py-3 text-sm text-amber-100 shadow-lg">
          <p className="font-semibold">Map tiles could not be loaded</p>
          <p className="mt-1 text-xs text-slate-300">{mapError}. Check your internet connection or whether tile.openstreetmap.org is blocked.</p>
        </div>
      )}
      {!styleReady && !mapError && (
        <div className="absolute left-4 bottom-4 z-10 rounded-lg bg-slate-950/80 px-3 py-2 text-xs text-slate-300">
          Loading street map…
        </div>
      )}
    </div>
  )
}
