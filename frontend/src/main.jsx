import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ClerkProvider } from '@clerk/clerk-react'
import './index.css'
import App from './App.jsx'

// ====== USER FILL THESE IN ======
const PUBLISHABLE_KEY = "pk_test_c2luY2VyZS1zYWxtb24tNjcuY2xlcmsuYWNjb3VudHMuZGV2JA"
// ================================

if (!PUBLISHABLE_KEY || PUBLISHABLE_KEY === "pk_test_YOUR_CLERK_PUBLISHABLE_KEY_HERE") {
  console.warn("Clerk publishable key is missing. Authentication UI may not render properly.");
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ClerkProvider publishableKey={PUBLISHABLE_KEY}>
      <App />
    </ClerkProvider>
  </StrictMode>,
)
