import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '192.168.0.150', // Explicitly bind to the static IP
    port: 5173,
    strictPort: true, // Fail if port 5173 is already in use
  },
});