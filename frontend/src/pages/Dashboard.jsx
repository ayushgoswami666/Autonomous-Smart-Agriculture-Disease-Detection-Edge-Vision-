import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useUser, useAuth } from '@clerk/clerk-react'

export default function Dashboard() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)
  const navigate = useNavigate()
  
  const { user } = useUser()
  const { getToken, signOut } = useAuth()

  // Use Clerk publicMetadata or fallback
  const role = user?.publicMetadata?.role || 'FARMER'
  const username = user?.fullName || user?.primaryEmailAddress?.emailAddress || 'Guest'
  const isSpecialist = role === 'AGRI-SPECIALIST'

  const handleLogout = async () => {
    await signOut()
    navigate('/')
  }

  const handleFileChange = (e) => {
    const selected = e.target.files[0]
    if (selected && selected.type.startsWith('image/')) {
      setFile(selected)
      setPreview(URL.createObjectURL(selected))
      setResults(null)
      setError(null)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const dropped = e.dataTransfer.files[0]
    if (dropped && dropped.type.startsWith('image/')) {
      setFile(dropped)
      setPreview(URL.createObjectURL(dropped))
      setResults(null)
      setError(null)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  const runAnalysis = async () => {
    if (!file) return

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const token = await getToken()
      const response = await fetch('http://127.0.0.1:8000/api/predict', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      })
      
      if (response.status === 401) {
        handleLogout()
        return
      }

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Analysis failed. Edge node unreachable.')
      }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const resetAnalysis = () => {
    setFile(null)
    setPreview(null)
    setResults(null)
    setError(null)
  }

  return (
    <>
      <nav className="navbar">
        <div className="nav-brand">🌿 Agri-Vision Dashboard</div>
        <div className="nav-controls">
          <span className={`role-badge ${isSpecialist ? 'specialist' : ''}`}>{role}</span>
          <span style={{ fontSize: '0.875rem', fontWeight: 500 }}>{username}</span>
          <button onClick={handleLogout} className="btn" style={{ padding: '0.4rem 1rem', fontSize: '0.875rem', marginTop: 0 }}>Logout</button>
        </div>
      </nav>

      <div className="app-container" style={{ paddingTop: 0, minHeight: 'calc(100vh - 80px)' }}>
        <header className="header" style={{ marginBottom: '2rem' }}>
          <h1 style={{ fontSize: '2rem' }}>Edge Node Diagnostics</h1>
          <p>Scan and process leaf samples locally</p>
        </header>

        <main className="main-content">
          {/* Left Side: Upload & Preview */}
          <div className="card">
            <h2 className="card-title">📷 Specimen Upload</h2>
            
            {!preview ? (
              <div 
                className="dropzone" 
                onClick={() => fileInputRef.current.click()}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
              >
                <div className="dropzone-icon">📥</div>
                <p className="dropzone-text">Click or drag a leaf image here</p>
                <p className="dropzone-subtext">JPG, PNG up to 10MB</p>
              </div>
            ) : (
              <div className="preview-container">
                <img src={preview} alt="Leaf Preview" className="preview-image" />
              </div>
            )}

            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              className="hidden-input" 
              accept="image/*" 
            />

            {preview && !results && !loading && (
              <div style={{ display: 'flex', gap: '1rem' }}>
                <button 
                  className="btn btn-primary" 
                  onClick={runAnalysis}
                >
                  🔍 Analyze Specimen
                </button>
                <button 
                  className="btn" 
                  onClick={resetAnalysis}
                  style={{ backgroundColor: '#e2e8f0', color: '#0f172a' }}
                >
                  Clear
                </button>
              </div>
            )}

            {error && (
              <div className="error-msg" style={{ marginTop: '1rem', marginBottom: 0 }}>
                🚨 {error}
              </div>
            )}
          </div>

          {/* Right Side: Results */}
          <div className="card">
            <h2 className="card-title">🩺 Diagnostic Report</h2>
            
            {!file && !results && !loading && (
              <div style={{ textAlign: 'center', padding: '3rem', color: '#64748b' }}>
                <p>Upload a specimen on the left to begin analysis.</p>
              </div>
            )}

            {loading && (
              <div className="spinner-container">
                <div className="spinner"></div>
                <p>Processing with Edge Node...</p>
              </div>
            )}

            {results && (
              <div style={{ animation: 'fadeInUp 0.6s ease-out' }}>
                <div className="result-header">
                  <div className="result-class">{results.top_class}</div>
                  <div className="result-confidence" style={{ backgroundColor: results.top_confidence > 0.8 ? 'var(--primary-light)' : (results.top_confidence > 0.5 ? '#fef3c7' : '#fee2e2'), color: results.top_confidence > 0.8 ? 'var(--primary-hover)' : (results.top_confidence > 0.5 ? 'var(--warning)' : 'var(--danger)')}}>
                    {Math.round(results.top_confidence * 100)}% Confidence
                  </div>
                </div>

                {isSpecialist && (
                  <div style={{ marginBottom: '2rem', padding: '1rem', backgroundColor: '#f8fafc', borderRadius: '0.5rem', border: '1px solid #e2e8f0' }}>
                    <h4 style={{ marginBottom: '1rem', color: '#0f172a', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Full Confidence Matrix</h4>
                    {results.predictions.map((pred, i) => (
                      <div key={i} className="confidence-bar-container">
                        <div className="confidence-label">
                          <span>{pred.class}</span>
                          <span>{Math.round(pred.confidence * 100)}%</span>
                        </div>
                        <div className="confidence-track">
                          <div 
                            className="confidence-fill" 
                            style={{ width: `${Math.round(pred.confidence * 100)}%`, backgroundColor: i === 0 ? 'var(--primary)' : '#94a3b8' }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {results.treatment ? (
                  <div className="treatment-section" style={{ marginTop: isSpecialist ? '1rem' : '0', paddingTop: isSpecialist ? '1rem' : '0' }}>
                    <h3 style={{ marginBottom: '1rem' }}>🔬 Agri-Cure Protocol</h3>
                    
                    <span className={`status-badge status-${results.treatment.status.replace(/[^a-zA-Z]/g, '') || 'Healthy'}`}>
                      {results.treatment.status}
                    </span>

                    <div className="treatment-item">
                      <h4>🩺 Recommended Treatment</h4>
                      <p>{results.treatment.treatment}</p>
                    </div>

                    <div className="treatment-item">
                      <h4>🛡️ Prevention Plan</h4>
                      <p>{results.treatment.prevention}</p>
                    </div>
                  </div>
                ) : (
                  <div className="treatment-section">
                    <p style={{ color: '#64748b' }}>No specific treatment protocol found for this class.</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </main>
      </div>
    </>
  )
}
