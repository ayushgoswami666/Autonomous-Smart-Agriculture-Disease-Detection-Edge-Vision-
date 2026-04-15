import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { SignedIn, SignedOut } from '@clerk/clerk-react'
import LandingPage from './pages/LandingPage'
import Dashboard from './pages/Dashboard'
import AuthPage from './pages/AuthPage'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={
          <>
            <SignedIn>
              <Navigate to="/dashboard" replace />
            </SignedIn>
            <SignedOut>
              <LandingPage />
            </SignedOut>
          </>
        } />
        
        {/* Role-based Auth Routes */}
        <Route path="/:role/:mode" element={
          <>
            <SignedIn>
              <Navigate to="/dashboard" replace />
            </SignedIn>
            <SignedOut>
              <AuthPage />
            </SignedOut>
          </>
        } />

        <Route path="/dashboard" element={
          <>
            <SignedIn>
              <Dashboard />
            </SignedIn>
            <SignedOut>
              <Navigate to="/" replace />
            </SignedOut>
          </>
        } />
        
        {/* Catch all route - returns to landing */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  )
}

export default App
