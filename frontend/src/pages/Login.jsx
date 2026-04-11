import { SignIn } from '@clerk/clerk-react'

export default function Login() {
  return (
    <div className="login-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', backgroundColor: '#f8fafc' }}>
      <div style={{ textAlign: 'center', width: '100%' }}>
        <h1 style={{ color: 'var(--primary)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>🌿 Agri-Vision</h1>
        <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>Secure Edge Node Access</p>
        
        <div style={{ display: 'flex', justifyContent: 'center', animation: 'fadeInUp 0.6s ease-out' }}>
          <SignIn routing="hash" />
        </div>
      </div>
    </div>
  )
}
