# PathSnap evaluator UI

React/Vite evaluator dashboard connected to the FastAPI backend.

The map uses CesiumJS. OpenStreetMap imagery works without a token. To enable Cesium World Terrain and 3D OpenStreetMap buildings, create a public Cesium ion token, copy `.env.example` to `.env.local`, and set `VITE_CESIUM_ION_TOKEN`. Restrict the token to `http://localhost:5173` during local development.

## Start

From the repository root, start the API in one terminal. In another PowerShell terminal:

```powershell
cd frontend
npm install
$env:VITE_API_BASE_URL = "http://localhost:8000"
npm run dev -- --host 0.0.0.0
```

The UI checks `GET /health`, gets available IDs from `GET /api/v1/trips`, then requests a prediction with `GET /api/v1/trips/{trip_id}`. Its API client is `src/services/api.js`.

For two laptops, point `VITE_API_BASE_URL` at the API laptop's LAN address and configure `PATHSNAP_CORS_ORIGINS` on the API laptop to include the UI origin, such as `http://192.168.1.20:5173`.
