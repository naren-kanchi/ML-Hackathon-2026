
import { useEffect, useRef } from 'react';
import { Map as MapLibreMap, NavigationControl } from 'maplibre-gl';
export default function RouteMap3D() {
  const mapContainer = useRef(null);
  const mapRef = useRef(null);

  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;

   const map = new MapLibreMap({
      container: mapContainer.current,

      style: 'https://tiles.openfreemap.org/styles/dark',

      center: [77.5946, 12.9716],

      zoom: 15,

      pitch: 60,

      bearing: -20,

      antialias: true,
    });

    mapRef.current = map;

    map.addControl(
      new NavigationControl({
        visualizePitch: true,
      }),
      'top-right'
    );

    map.on('load', () => {
      console.log('PathSnap 3D Map Loaded!');
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  return (
    <div
      ref={mapContainer}
      style={{
        width: '100%',
        height: '600px',
        borderRadius: '16px',
        overflow: 'hidden',
      }}
    />
  );
}
