/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#eef4ff',
          100: '#d9e5ff',
          200: '#bcd0ff',
          300: '#8eb0ff',
          400: '#5985ff',
          500: '#345dfb',
          600: '#1f3df0',
          700: '#1a2fdc',
          800: '#1c2cb2',
          900: '#1c2c8c',
          950: '#151d55',
        },
      },
      fontFamily: {
        sans: [
          'Inter',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'Arial',
          'sans-serif',
        ],
        mono: [
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Consolas',
          'Liberation Mono',
          'monospace',
        ],
      },
      keyframes: {
        'caret-blink': {
          '0%, 70%, 100%': { opacity: '1' },
          '20%, 50%': { opacity: '0' },
        },
        'slide-in-right': {
          from: { transform: 'translateX(1.5rem)', opacity: '0' },
          to: { transform: 'translateX(0)', opacity: '1' },
        },
        'fade-in': {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
      },
      animation: {
        'caret-blink': 'caret-blink 1.1s steps(1) infinite',
        'slide-in-right': 'slide-in-right 160ms ease-out',
        'fade-in': 'fade-in 140ms ease-out',
      },
    },
  },
  plugins: [],
};
