/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        'editor-bg': '#ffffff',
        'editor-bg-dark': '#1E1E1E',
        'editor-sidebar': '#f3f4f6',
        'editor-sidebar-dark': '#252525',
        'editor-highlight': '#e5e7eb',
        'editor-highlight-dark': '#2D2D2D',
        'editor-accent': '#0D9488',
        'editor-accent-dark': '#0D9488',
        'editor-text': '#1f2937',
        'editor-text-dark': '#D4D4D4',
        'editor-muted': '#6b7280',
        'editor-muted-dark': '#9ca3af',
        'editor-comment': '#6B7280',
        'editor-comment-dark': '#6A9955',
        'editor-string': '#0F766E',
        'editor-string-dark': '#CE9178',
        'editor-number': '#059669',
        'editor-number-dark': '#B5CEA8',
        'editor-keyword': '#2563EB',
        'editor-keyword-dark': '#569CD6'
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: '65ch',
            color: 'inherit',
            a: {
              color: '#0D9488',
              '&:hover': {
                color: '#0F766E',
              },
            },
            pre: {
              backgroundColor: '#1E1E1E',
              color: '#D4D4D4',
              fontFamily: 'JetBrains Mono, monospace',
            },
            code: {
              color: '#D4D4D4',
              fontFamily: 'JetBrains Mono, monospace',
            },
          },
        },
      },
    },
  },
  plugins: [],
}