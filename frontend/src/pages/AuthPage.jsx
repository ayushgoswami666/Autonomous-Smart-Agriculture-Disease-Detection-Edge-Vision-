import { SignIn, SignUp } from '@clerk/clerk-react'
import { useParams, Link } from 'react-router-dom'
import './AuthPage.css'

export default function AuthPage() {
  const { role, mode } = useParams()

  const isSignIn = mode === 'signin'
  const roleTitle = role === 'farmer' ? 'Farmer' : 'Agri Specialist'
  const roleIcon = role === 'farmer' ? '👨‍🌾' : '🔬'

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <Link to="/" className="auth-logo">
            🌿 Agri Vision
          </Link>
          <h2>{isSignIn ? 'Welcome Back' : 'Join Us'}</h2>
          <p className="auth-subtitle">
            {isSignIn ? `Sign in as a ${roleTitle}` : `Create your ${roleTitle} account`} {roleIcon}
          </p>
        </div>

        <div className="auth-form-wrapper">
          {isSignIn ? (
            <SignIn
              routing="path"
              path={`/${role}/signin`}
              signUpUrl={`/${role}/signup`}
              afterSignInUrl="/dashboard"
            />
          ) : (
            <SignUp
              routing="path"
              path={`/${role}/signup`}
              signInUrl={`/${role}/signin`}
              afterSignUpUrl="/dashboard"
            />
          )}
        </div>

        <div className="auth-footer">
          <p>Need help? <a href="mailto:support@agrivision.com">Contact Support</a></p>
        </div>
      </div>
    </div>
  )
}
