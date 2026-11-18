import { useEffect, useMemo, useRef } from 'react'
import { LngLatBounds, Map as MapLibreMap, NavigationControl } from 'maplibre-gl'

const EMPTY_DATA = []

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

export default function RouteMap3D({ gpsPoints = EMPTY_DATA, predictedRoute = EMPTY_DATA }) {
  const mapContainer = useRef(null)
  const mapRef = useRef(null)
  const dataRef = useRef({ gpsPoints, routeLine: [] })
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
      style: 'https://tiles.openfreemap.org/styles/dark',
      center: gpsPoints[0] ?? routeLine[0] ?? [77.5946, 12.9716],
      zoom: 15,
      pitch: 60,
      bearing: -20,
      antialias: true,
    })
    mapRef.current = map
    map.addControl(new NavigationControl({ visualizePitch: true }), 'top-right')
    map.on('load', () => {
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
    })
    return () => {
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

  return <div ref={mapContainer} className="w-full h-[600px] rounded-2xl overflow-hidden" />
}
