import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const fileInputRef = useRef(null);

  const API_URL ='http://localhost:8000';

  // Track mouse position for interactive background effects
  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const classInfo = {
    akiec: {
      name: 'Actinic Keratosis',
      color: '#FF6B6B',
      description: 'Rough, scaly spots on skin exposed to sun',
      risk: 'Moderate'
    },
    bcc: {
      name: 'Basal Cell Carcinoma',
      color: '#FF4757',
      description: 'Most common type of skin cancer',
      risk: 'High'
    },
    bkl: {
      name: 'Benign Keratosis',
      color: '#FFA502',
      description: 'Common, harmless brown growths',
      risk: 'Low'
    },
    df: {
      name: 'Dermatofibroma',
      color: '#FFD93D',
      description: 'Benign fibrous growth in the skin',
      risk: 'Low'
    },
    mel: {
      name: 'Melanoma',
      color: '#6C3483',
      description: 'Most serious type of skin cancer',
      risk: 'Very High'
    },
    nv: {
      name: 'Nevus',
      color: '#1DD1A1',
      description: 'Common mole, usually harmless',
      risk: 'Very Low'
    },
    vasc: {
      name: 'Vascular Lesion',
      color: '#EE5A6F',
      description: 'Blood vessel abnormalities in skin',
      risk: 'Low'
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file');
      return;
    }

    setImage(file);
    setError(null);
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('drag-over');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) {
      processFile(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      setError('Please select an image');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', image);

      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (response.data.success) {
        const resultData = {
          prediction: response.data.prediction,
          confidence: response.data.confidence,
          probabilities: response.data.probabilities,
          timestamp: new Date().toLocaleString(),
          preview: preview,
        };
        setResult(resultData);
        setHistory([resultData, ...history.slice(0, 4)]);
      } else {
        setError(response.data.error || 'Prediction failed');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Error connecting to server');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const riskLevelColor = (risk) => {
    switch (risk) {
      case 'Very High':
        return '#6C3483';
      case 'High':
        return '#FF4757';
      case 'Moderate':
        return '#FFA502';
      case 'Low':
        return '#1DD1A1';
      default:
        return '#95A5A6';
    }
  };

  return (
    <div className="app">
      {/* Interactive Background */}
      <div className="background-container">
        <div className="medical-grid"></div>
        <div 
          className="interactive-cursor"
          style={{
            left: `${mousePosition.x}px`,
            top: `${mousePosition.y}px`,
          }}
        ></div>
        <div className="floating-orbs">
          <div className="orb orb-1"></div>
          <div className="orb orb-2"></div>
          <div className="orb orb-3"></div>
          <div className="orb orb-4"></div>
        </div>
        <div className="dna-helix"></div>
      </div>

      {/* Navigation */}
      <nav className="navbar">
        <div className="nav-container">
          <div className="nav-logo">
            <div className="logo-icon">
              <svg viewBox="0 0 40 40" width="32" height="32">
                {/* Skin with lesion detection */}
                <circle cx="20" cy="20" r="18" fill="none" stroke="currentColor" strokeWidth="2" />
                <path d="M20 8 Q28 14 28 20 Q28 28 20 28 Q12 28 12 20 Q12 14 20 8" fill="none" stroke="currentColor" strokeWidth="2" />
                {/* Detection point */}
                <circle cx="20" cy="20" r="4" fill="currentColor" opacity="0.8" />
                {/* Scanning circles */}
                <circle cx="20" cy="20" r="6" fill="none" stroke="currentColor" strokeWidth="1" opacity="0.5" />
              </svg>
            </div>
            <span>DermAssist</span>
          </div>
          <ul className="nav-links">
            <li><a href="#features">Features</a></li>
            <li><a href="#about">About</a></li>
            <li><a href="#contact">Contact</a></li>
          </ul>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="hero">
        <div className="hero-particles">
          {[...Array(20)].map((_, i) => (
            <div 
              key={i} 
              className="particle"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                '--delay': `${i * 0.1}s`
              }}
            ></div>
          ))}
        </div>
        
        <div className="hero-content">
          <h1>Skin Lesion Analysis</h1>
          <p>Intelligent dermatological classification powered by advanced AI</p>
          <button 
            className="cta-button"
            onClick={() => fileInputRef.current?.click()}
          >
            <span className="button-text">Get Started</span>
            <span className="button-icon">→</span>
          </button>
        </div>

        <div className="hero-visual">
          <div className="medical-scanner">
            <div className="scanner-ring"></div>
            <div className="scanner-ring delay-1"></div>
            <div className="scanner-ring delay-2"></div>
          </div>
        </div>
      </header>

      <main className="container">
        {/* Upload Section */}
        <section className="upload-section">
          <div className="upload-card">
            <div className="card-glow"></div>
            <h2>Upload Your Image</h2>
            <div
              className="upload-area"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                hidden
              />
              {preview ? (
                <div className="preview-container">
                  <div className="preview-frame">
                    <img src={preview} alt="preview" className="preview-image" />
                    <div className="preview-overlay"></div>
                  </div>
                  <button
                    className="change-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      fileInputRef.current?.click();
                    }}
                  >
                    Change Image
                  </button>
                </div>
              ) : (
                <div className="upload-placeholder">
                  <div className="upload-icon">
                    <svg viewBox="0 0 24 24" width="64" height="64">
                      <path
                        fill="currentColor"
                        d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"
                      />
                    </svg>
                  </div>
                  <p>Drag and drop or click to upload</p>
                  <span>PNG, JPG or GIF (max 10MB)</span>
                </div>
              )}
            </div>

            {error && (
              <div className="error-message">
                <svg viewBox="0 0 24 24" width="20" height="20">
                  <path
                    fill="currentColor"
                    d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"
                  />
                </svg>
                {error}
              </div>
            )}

            {preview && (
              <div className="button-group">
                <button
                  className="analyze-btn"
                  onClick={handleSubmit}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <span>Analyze Image</span>
                      <span className="btn-icon">🔬</span>
                    </>
                  )}
                </button>
                <button className="secondary-btn" onClick={handleClear}>
                  Clear
                </button>
              </div>
            )}
          </div>
        </section>

        {/* Results Section */}
        {result && (
          <section className="results-section">
            <h2>Analysis Results</h2>
            <div className="results-container">
              {/* Main Result Card */}
              <div
                className="main-result"
                style={{
                  borderColor: classInfo[result.prediction].color,
                  backgroundColor: classInfo[result.prediction].color + '08'
                }}
              >
                <div className="result-glow" style={{
                  backgroundColor: classInfo[result.prediction].color + '20'
                }}></div>
                <div className="result-header">
                  <div
                    className="result-badge"
                    style={{ backgroundColor: classInfo[result.prediction].color }}
                  >
                    <svg viewBox="0 0 24 24" width="24" height="24">
                      <path
                        fill="white"
                        d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"
                      />
                    </svg>
                  </div>
                  <div>
                    <h3>{classInfo[result.prediction].name}</h3>
                    <p className="confidence-score">
                      Confidence: <span className="confidence-value">{(result.confidence * 100).toFixed(1)}%</span>
                    </p>
                  </div>
                </div>
                <p className="result-description">
                  {classInfo[result.prediction].description}
                </p>
                <div className="risk-indicator">
                  <span className="risk-label">Risk Level:</span>
                  <span
                    className="risk-badge"
                    style={{ color: riskLevelColor(classInfo[result.prediction].risk) }}
                  >
                    {classInfo[result.prediction].risk}
                  </span>
                </div>
              </div>

              {/* Probabilities Chart */}
              <div className="probabilities-card">
                <div className="card-glow"></div>
                <h3>Classification Probabilities</h3>
                <div className="probability-list">
                  {Object.entries(result.probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([className, prob], idx) => (
                      <div key={className} className="probability-item" style={{
                        '--item-delay': `${idx * 50}ms`
                      }}>
                        <div className="prob-info">
                          <span className="prob-label">
                            {classInfo[className].name}
                          </span>
                          <span className="prob-value">
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="prob-bar-container">
                          <div
                            className="prob-bar"
                            style={{
                              width: `${prob * 100}%`,
                              backgroundColor: classInfo[className].color,
                            }}
                          ></div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>

            {/* Medical Disclaimer */}
            <div className="disclaimer">
              <div className="disclaimer-icon">⚠️</div>
              <div>
                <strong>Medical Disclaimer:</strong> This tool is for informational purposes only and should not replace professional medical advice. Always consult with a dermatologist for accurate diagnosis and treatment.
              </div>
            </div>
          </section>
        )}

        {/* History Section */}
        {history.length > 0 && (
          <section className="history-section">
            <h2>Recent Analyses</h2>
            <div className="history-grid">
              {history.map((item, idx) => (
                <div
                  key={idx}
                  className="history-card"
                  style={{
                    borderLeftColor: classInfo[item.prediction].color,
                    '--card-delay': `${idx * 50}ms`
                  }}
                >
                  <div className="history-image-container">
                    <img src={item.preview} alt="analysis" />
                    <div className="history-overlay"></div>
                  </div>
                  <div className="history-info">
                    <h4>{classInfo[item.prediction].name}</h4>
                    <p>{(item.confidence * 100).toFixed(1)}% confident</p>
                    <span className="history-time">{item.timestamp}</span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Features Section */}
        <section id="features" className="features-section">
          <h2>Why Choose DermAssist?</h2>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-glow"></div>
              <div className="feature-icon" style={{ color: '#667EEA' }}>
                <svg viewBox="0 0 24 24" width="32" height="32">
                  <path
                    fill="currentColor"
                    d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z"
                  />
                </svg>
              </div>
              <h3>AI-Powered</h3>
              <p>Advanced deep learning model trained on thousands of dermatological images</p>
            </div>
            <div className="feature-card">
              <div className="feature-glow"></div>
              <div className="feature-icon" style={{ color: '#FF6B6B' }}>
                <svg viewBox="0 0 24 24" width="32" height="32">
                  <path
                    fill="currentColor"
                    d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm0-13c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5z"
                  />
                </svg>
              </div>
              <h3>Instant Results</h3>
              <p>Get classification results in milliseconds with confidence scores</p>
            </div>
            <div className="feature-card">
              <div className="feature-glow"></div>
              <div className="feature-icon" style={{ color: '#1DD1A1' }}>
                <svg viewBox="0 0 24 24" width="32" height="32">
                  <path
                    fill="currentColor"
                    d="M12 1C5.92 1 1 5.92 1 12s4.92 11 11 11 11-4.92 11-11S18.08 1 12 1zm-2 16l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
                  />
                </svg>
              </div>
              <h3>Reliable</h3>
              <p>90.62% accuracy with optimized inference for medical applications</p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <p>&copy; 2024 DermAssist. Powered by Advanced AI.</p>
          <p className="footer-disclaimer">
            For medical guidance, please consult a licensed dermatologist.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
