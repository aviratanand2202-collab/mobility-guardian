import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // NOTE: Dev-only proxy!
      // This proxy forwards /api and /ws to http://localhost:8000 during local Vite development.
      // Real CORS middleware will be needed on the backend before any production build
      // or demo from another machine (e.g. app.add_middleware(CORSMiddleware, ...)).
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
