// import { defineConfig } from 'vite'
// import react from '@vitejs/plugin-react'

// // https://vite.dev/config/
// export default defineConfig({
//   plugins: [react()],
// })

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist', // Ensures correct output directory
    assetsDir: 'assets', // Keeps assets organized
  },
  server: {
    strictPort: true,
    port: 5454, // Default Vite port
  },
  resolve: {
    alias: {
      '@': '/src', // Adjust based on your project structure
    },
  },
});
