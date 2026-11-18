
import { defineConfig, normalizePath } from 'vite'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const projectRoot = path.dirname(fileURLToPath(import.meta.url))
const cesiumSource = normalizePath(path.join(projectRoot, 'node_modules/cesium/Build/Cesium'))
const cesiumBaseUrl = 'cesiumStatic'
const cesiumDirectories = ['ThirdParty', 'Workers', 'Assets', 'Widgets']

function cesiumStaticAssets() {
  return {
    name: 'pathsnap-cesium-static-assets',
    configureServer(server) {
      server.middlewares.use(`/${cesiumBaseUrl}`, (request, response, next) => {
        let relativePath
        try {
          relativePath = decodeURIComponent((request.url ?? '/').split('?')[0]).replace(/^\/+/, '')
        } catch {
          next()
          return
        }
        const filePath = path.resolve(cesiumSource, relativePath)
        if (!filePath.startsWith(`${path.resolve(cesiumSource)}${path.sep}`)) {
          next()
          return
        }
        fs.stat(filePath, (error, stats) => {
          if (error || !stats.isFile()) {
            next()
            return
          }
          const mimeTypes = {
            '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm',
            '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.svg': 'image/svg+xml',
            '.css': 'text/css', '.ktx2': 'application/octet-stream', '.bin': 'application/octet-stream',
          }
          response.setHeader('Content-Type', mimeTypes[path.extname(filePath).toLowerCase()] ?? 'application/octet-stream')
          fs.createReadStream(filePath).pipe(response)
        })
      })
    },
    closeBundle() {
      const destinationRoot = path.join(projectRoot, 'dist', cesiumBaseUrl)
      for (const directory of cesiumDirectories) {
        fs.cpSync(path.join(cesiumSource, directory), path.join(destinationRoot, directory), { recursive: true, force: true })
      }
    },
  }
}

export default defineConfig({
  define: {
    CESIUM_BASE_URL: JSON.stringify(`/${cesiumBaseUrl}/`),
  },
  plugins: [
    react(),
    tailwindcss(),
    cesiumStaticAssets(),
  ],
})
