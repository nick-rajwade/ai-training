import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import 'katex/dist/katex.min.css'
import App from './App.jsx'
import { BookmarkProvider } from './contexts/BookmarkContext'
import ErrorBoundary from './components/ErrorBoundary'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <BookmarkProvider>
        <App />
      </BookmarkProvider>
    </ErrorBoundary>
  </StrictMode>,
)
