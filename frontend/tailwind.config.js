/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#1a1a1a',
          800: '#242424',
          700: '#2d2d2d',
          600: '#3d3d3d',
        },
        accent: {
          pink: '#e91e8c',
          purple: '#9c27b0',
        }
      },
    },
  },
  plugins: [],
}
