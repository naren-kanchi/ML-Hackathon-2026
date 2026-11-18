import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Cartesian3,
  Color,
  createOsmBuildingsAsync,
  EllipsoidTerrainProvider,
  HeightReference,
  ImageryLayer,
  Ion,
  Math as CesiumMath,
  OpenStreetMapImageryProvider,
  Rectangle,
  Terrain,
  Viewer,
} from 'cesium'
import RouteMapLeaflet from './RouteMapLeaflet'

const DEFAULT_CENTER = [77.5946, 12.9716]
const TOKEN = import.meta.env.VITE_CESIUM_ION_TOKEN?.trim()

function validCoordinate(value) {
  if (!Array.isArray(value) || value.length < 2) return false
  const [longitude, latitude] = value.map(Number)
  return Number.isFinite(longitude) && Number.isFinite(latitude)
    && Math.abs(longitude) <= 180 && Math.abs(latitude) <= 90
}

export default function RouteMap({ gpsPoints = [], predictedRoute = [] }) {
  const container = useRef(null)
  const viewer = useRef(null)
  const [ready, setReady] = useState(false)
  const [fallback, setFallback] = useState(false)
  const [mapMessage, setMapMessage] = useState('')
  const [ionBuildingsReady, setIonBuildingsReady] = useState(false)

  const route = useMemo(() => predictedRoute
    .flatMap((segment) => Array.isArray(segment?.coordinates) ? segment.coordinates : [])
    .filter(validCoordinate)
    .map(([longitude, latitude]) => [Number(longitude), Number(latitude)]), [predictedRoute])
  const gps = useMemo(() => gpsPoints.filter(validCoordinate)
    .map(([longitude, latitude]) => [Number(longitude), Number(latitude)]), [gpsPoints])

  useEffect(() => {
    if (!container.current || viewer.current || fallback) return undefined
    let disposed = false
    let cesiumViewer

    try {
      if (TOKEN) Ion.defaultAccessToken = TOKEN
      cesiumViewer = new Viewer(container.current, {
        baseLayer: new ImageryLayer(new OpenStreetMapImageryProvider({ url: 'https://tile.openstreetmap.org/' })),
        terrainProvider: TOKEN ? undefined : new EllipsoidTerrainProvider(),
        terrain: TOKEN ? Terrain.fromWorldTerrain() : undefined,
        baseLayerPicker: false,
        geocoder: false,
        homeButton: false,
        sceneModePicker: false,
        navigationHelpButton: false,
        animation: false,
        timeline: false,
        fullscreenButton: false,
        infoBox: false,
        selectionIndicator: false,
        scene3DOnly: true,
        requestRenderMode: false,
      })
      viewer.current = cesiumViewer
      setReady(true)
      cesiumViewer.scene.globe.depthTestAgainstTerrain = true
      cesiumViewer.cesiumWidget.errorEvent.addEventListener((error) => {
        console.error('Cesium rendering error:', error)
        if (!disposed) setMapMessage(`Cesium map resource error: ${error.message ?? error}`)
      })
      cesiumViewer.camera.setView({
        destination: Cartesian3.fromDegrees(DEFAULT_CENTER[0], DEFAULT_CENTER[1], 14000),
        orientation: {
          heading: CesiumMath.toRadians(0),
          pitch: CesiumMath.toRadians(-42),
          roll: 0,
        },
      })

      if (TOKEN) {
        createOsmBuildingsAsync().then((buildings) => {
          if (disposed) return
          cesiumViewer.scene.primitives.add(buildings)
          setIonBuildingsReady(true)
          cesiumViewer.scene.requestRender()
        }).catch((error) => {
          console.error('Cesium OSM buildings failed to load:', error)
          if (!disposed) setMapMessage(`3D buildings could not load: ${error.message}`)
        })
      } else {
        setMapMessage('Add VITE_CESIUM_ION_TOKEN to frontend/.env.local to enable 3D buildings and terrain.')
      }
    } catch (error) {
      console.error('Cesium viewer failed to initialize:', error)
      setMapMessage(`Cesium could not start: ${error.message}`)
      setFallback(true)
    }

    const resizeObserver = new ResizeObserver(() => cesiumViewer?.resize())
    if (container.current) resizeObserver.observe(container.current)
    return () => {
      disposed = true
      resizeObserver.disconnect()
      cesiumViewer?.destroy()
      viewer.current = null
    }
  }, [fallback])

  useEffect(() => {
    const activeViewer = viewer.current
    if (!ready || !activeViewer || activeViewer.isDestroyed()) return
    const entities = activeViewer.entities
    entities.values
      .filter((entity) => entity.id?.startsWith('pathsnap-'))
      .forEach((entity) => entities.remove(entity))

    const pointsToFrame = [...gps, ...route]
    gps.forEach(([longitude, latitude], index) => {
      entities.add({
        id: `pathsnap-gps-${index}`,
        position: Cartesian3.fromDegrees(longitude, latitude, 8),
        point: {
          pixelSize: 7,
          color: Color.fromCssColorString('#3b82f6'),
          outlineColor: Color.WHITE,
          outlineWidth: 2,
          heightReference: HeightReference.CLAMP_TO_GROUND,
        },
      })
    })
    if (route.length > 1) {
      entities.add({
        id: 'pathsnap-route-line',
        polyline: {
          positions: Cartesian3.fromDegreesArray(route.flat()),
          width: 7,
          material: Color.fromCssColorString('#b6ff3b'),
          clampToGround: true,
        },
      })
    }
    if (pointsToFrame.length > 1) {
      const west = Math.min(...pointsToFrame.map(([longitude]) => longitude))
      const east = Math.max(...pointsToFrame.map(([longitude]) => longitude))
      const south = Math.min(...pointsToFrame.map(([, latitude]) => latitude))
      const north = Math.max(...pointsToFrame.map(([, latitude]) => latitude))
      activeViewer.camera.flyTo({
        destination: Rectangle.fromDegrees(west, south, east, north),
        duration: 1.1,
        orientation: { heading: 0, pitch: CesiumMath.toRadians(-35), roll: 0 },
      })
    } else if (pointsToFrame.length === 1) {
      activeViewer.camera.flyTo({ destination: Cartesian3.fromDegrees(...pointsToFrame[0], 3500), duration: 1.1 })
    }
  }, [gps, route, ready])

  if (fallback) {
    return <div className="relative h-full min-h-[420px] w-full overflow-hidden rounded-xl">
      <RouteMapLeaflet gpsPoints={gpsPoints} predictedRoute={predictedRoute} />
      <div className="absolute left-3 top-3 z-[1000] rounded-lg border border-amber-400/30 bg-black/85 px-3 py-2 text-xs text-amber-100">Cesium could not initialize. Showing the 2D map. {mapMessage}</div>
    </div>
  }

  return <div className="relative h-full min-h-[420px] w-full overflow-hidden rounded-xl bg-[#dce3e7]">
    <div ref={container} className="absolute inset-0" />
    {(!ready || (TOKEN && !ionBuildingsReady)) && !mapMessage && <div className="absolute bottom-3 left-3 z-10 rounded-lg bg-black/80 px-3 py-2 text-xs text-white">Loading Cesium 3D globe…</div>}
    {mapMessage && <div className="absolute bottom-3 left-3 right-3 z-10 rounded-lg border border-amber-400/40 bg-black/85 px-3 py-2 text-xs text-amber-100">{mapMessage}</div>}
    <div className="absolute left-3 top-3 z-10 rounded-lg border border-[#b6ff3b]/30 bg-black/75 px-3 py-2 text-[11px] text-[#d8ff9b]">CesiumJS 3D globe{ionBuildingsReady ? ' · buildings enabled' : ''}</div>
  </div>
}
