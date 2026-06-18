import React, { useState, useRef } from 'react';
import axios from 'axios';
import './DermAssist.css';

function DermAssist() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const fileInputRef = useRef(null);
  const API_URL = 'https://skin-lesion-api-p7ga.onrender.com';;

  const classMetadata = {
    akiec: {
      name: 'Actinic Keratosis',
      abbreviation: 'AK',
      risk: 'medium',
      description: 'Precancerous lesion caused by sun exposure. Requires professional evaluation and may benefit from treatment.',
      color: 'amber'
    },
    bcc: {
      name: 'Basal Cell Carcinoma',
      abbreviation: 'BCC',
      risk: 'high',
      description: 'Most common form of skin cancer. Typically slow-growing and highly treatable when identified early.',
      color: 'red'
    },
    bkl: {
      name: 'Benign Keratosis',
      abbreviation: 'BK',
      risk: 'low',
      description: 'Common, harmless skin growth. Usually appears as brown, black, or tan waxy bumps.',
      color: 'green'
    },
    df: {
      name: 'Dermatofibroma',
      abbreviation: 'DF',
      risk: 'low',
      description: 'Benign fibrous nodule, typically painless. Does not require treatment unless cosmetically concerning.',
      color: 'green'
    },
    mel: {
      name: 'Melanoma',
      abbreviation: 'MEL',
      risk: 'critical',
      description: 'Most serious form of skin cancer. Early detection and treatment significantly improve outcomes.',
      color: 'red'
    },
    nv: {
      name: 'Nevus (Mole)',
      abbreviation: 'NV',
      risk: 'low',
      description: 'Common, benign skin lesion present from birth or acquired over time. Monitoring for changes is recommended.',
      color: 'green'
    },
    vasc: {
      name: 'Vascular Lesion',
      abbreviation: 'VASC',
      risk: 'low',
      description: 'Benign lesion composed of blood vessels. Usually requires no treatment unless for cosmetic reasons.',
      color: 'green'
    }
  };

  const processFile = (file) => {
    if (file.size > 10 * 1024 * 1024) {
      setError('Image size exceeds 10 MB limit');
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

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('drag-active');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-active');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-active');
    const file = e.dataTransfer.files[0];
    if (file) {
      processFile(file);
    }
  };

  const handleAnalyze = async () => {
    if (!image) {
      setError('Please select an image');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', image);

      const response = await axios.post("https://skin-lesion-api-p7ga.onrender.com/predict", formData, {
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
        setError(response.data.error || 'Analysis failed');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to connect to server');
    } finally {
      setLoading(false);
    }
  };


  const handleReset = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getRiskLevel = (risk) => {
    const risks = {
      'low': { label: 'Low Risk', class: 'risk-low' },
      'medium': { label: 'Medium Risk', class: 'risk-medium' },
      'high': { label: 'High Risk', class: 'risk-high' },
      'critical': { label: 'Critical', class: 'risk-critical' }
    };
    return risks[risk] || risks['low'];
  };

  return (
    <div className="derm-app">
      {/* HEADER */}
      <header className="derm-header">
        <div className="derm-header-content">
          <div className="derm-logo">
            <svg viewBox="0 0 32 32" className="logo-icon">
              <circle cx="16" cy="16" r="14" fill="none" stroke="currentColor" strokeWidth="1.5" />
              <circle cx="16" cy="16" r="9" fill="none" stroke="currentColor" strokeWidth="1.5" />
              <circle cx="16" cy="16" r="4" fill="currentColor" />
              <line x1="16" y1="2" x2="16" y2="6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              <line x1="16" y1="26" x2="16" y2="30" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            <span className="logo-text">DermAssist</span>
          </div>
          <nav className="derm-nav">
            <a href="#about">About</a>
            <a href="#classes">Lesion Types</a>
            <a href="#research">Research</a>
          </nav>
        </div>
      </header>

      {/* HERO */}
      <section className="derm-hero">
        <div className="hero-content">
          <h1 className="hero-title">Clinical Skin Lesion Analysis</h1>
          <p className="hero-subtitle">
            AI-assisted dermoscopic image classification using distilled MobileNetV2. 
            For clinical research and educational purposes.
          </p>
          <div className="hero-actions">
            <button 
              className="btn-primary"
              onClick={() => fileInputRef.current?.click()}
            >
              Upload Image
            </button>
            <a href="#classes" className="btn-secondary">Learn About Lesion Types</a>
          </div>
        </div>
      </section>

      {/* MAIN CONTENT */}
      <main className="derm-main">
        {/* UPLOAD SECTION */}
        <section className="derm-section upload-section">
          <div className="section-container">
            <div className="section-header">
              <h2>Image Analysis</h2>
              <p>Upload a dermoscopic image for classification</p>
            </div>

            <div className="upload-container">
              <div
                className="upload-box"
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
                  <div className="upload-preview">
                    <img src={preview} alt="selected" className="preview-img" />
                    <div className="preview-overlay">
                      <button
                        className="preview-action"
                        onClick={(e) => {
                          e.stopPropagation();
                          fileInputRef.current?.click();
                        }}
                      >
                        Change Image
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="upload-content">
                    <svg className="upload-icon" viewBox="0 0 24 24">
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z" fill="currentColor" />
                    </svg>
                    <p className="upload-text">Drag and drop image or click to select</p>
                    <p className="upload-hint">JPG, PNG, or GIF (max 10 MB)</p>
                  </div>
                )}
              </div>

              {error && (
                <div className="error-message">
                  <svg viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="currentColor" />
                  </svg>
                  <span>{error}</span>
                </div>
              )}

              {preview && (
                <div className="upload-actions">
                  <button
                    className="btn-primary"
                    onClick={handleAnalyze}
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <span className="spinner"></span>
                        Analyzing...
                      </>
                    ) : (
                      'Analyze Image'
                    )}
                  </button>
                  <button className="btn-ghost" onClick={handleReset}>
                    Clear
                  </button>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* RESULTS SECTION */}
        {result && (
          <section className="derm-section results-section">
            <div className="section-container">
              <div className="section-header">
                <h2>Analysis Results</h2>
              </div>

              <div className="results-grid">
                {/* PRIMARY RESULT */}
                <div className="result-card primary-result">
                  <div className="result-header-row">
                    <div className={`risk-badge ${getRiskLevel(classMetadata[result.prediction].risk).class}`}>
                      {getRiskLevel(classMetadata[result.prediction].risk).label}
                    </div>
                  </div>

                  <div className="result-title">
                    <h3>{classMetadata[result.prediction].name}</h3>
                    <p className="result-code">({classMetadata[result.prediction].abbreviation})</p>
                  </div>

                  <div className="confidence-display">
                    <span className="confidence-label">Model Confidence</span>
                    <span className="confidence-value">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  <p className="result-description">
                    {classMetadata[result.prediction].description}
                  </p>

                  <div className="result-disclaimer">
                    <svg viewBox="0 0 24 24">
                      <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" fill="currentColor" />
                    </svg>
                    <p>
                      This classification is generated by machine learning and should be reviewed by a qualified dermatologist. 
                      Not for diagnostic purposes.
                    </p>
                  </div>
                  
                </div>

                {/* PROBABILITY DISTRIBUTION */}
                <div className="result-card">
                  <h3 className="card-title">Classification Distribution</h3>
                  <div className="probability-chart">
                    {Object.entries(result.probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([className, prob], index) => {
                        const metadata = classMetadata[className];
                        const isTopPrediction = className === result.prediction;
                        return (
                          <div
                            key={className}
                            className={`probability-row ${isTopPrediction ? 'top-prediction' : ''}`}
                          >
                            <div className="prob-label">
                              <span className="prob-name">{metadata.abbreviation}</span>
                              <span className="prob-fullname">{metadata.name}</span>
                            </div>
                            <div className="prob-bar-container">
                              <div
                                className={`prob-bar prob-${metadata.color}`}
                                style={{ width: `${prob * 100}%` }}
                              />
                            </div>
                            <span className="prob-percent">
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        );
                      })}
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        

        {/* LESION TYPES SECTION */}
        <section id="classes" className="derm-section classes-section">
          <div className="section-container">
            <div className="section-header">
              <h2>Skin Lesion Classification</h2>
              <p>The HAM10000 dataset includes seven types of skin lesions</p>
            </div>

            <div className="classes-grid">
              {Object.entries(classMetadata).map(([code, meta]) => (
                <div key={code} className="class-card">
                  <div className={`class-header class-${meta.color}`}>
                    <span className="class-abbr">{meta.abbreviation}</span>
                  </div>
                  <h4>{meta.name}</h4>
                  <p className="class-description">{meta.description}</p>
                  <div className={`class-risk ${getRiskLevel(meta.risk).class}`}>
                    {getRiskLevel(meta.risk).label}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* RESEARCH SECTION */}
        <section id="research" className="derm-section research-section">
          <div className="section-container">
            <div className="section-header">
              <h2>Research Model</h2>
              <p>Technical specifications of the distilled MobileNetV2 classifier</p>
            </div>

            <div className="research-grid">
              <div className="research-card">
                <div className="research-icon">📊</div>
                <h4>Model Architecture</h4>
                <p className="research-value">MobileNetV2 (Distilled)</p>
                <p className="research-desc">Lightweight CNN architecture optimized for mobile deployment</p>
              </div>

              <div className="research-card">
                <div className="research-icon">🎯</div>
                <h4>Test Accuracy</h4>
                <p className="research-value">90.62%</p>
                <p className="research-desc">Evaluated on HAM10000 dataset</p>
              </div>

              <div className="research-card">
                <div className="research-icon">⚡</div>
                <h4>Inference Speed</h4>
                <p className="research-value">2.5× Faster</p>
                <p className="research-desc">Compared to full-size teacher model</p>
              </div>

              <div className="research-card">
                <div className="research-icon">📦</div>
                <h4>Model Compression</h4>
                <p className="research-value">5.3× Smaller</p>
                <p className="research-desc">Knowledge distillation reduces model size</p>
              </div>

              <div className="research-card">
                <div className="research-icon">✅</div>
                <h4>External Validation</h4>
                <p className="research-value">PH2 Dataset</p>
                <p className="research-desc">Cross-validation on independent dataset</p>
              </div>

              <div className="research-card">
                <div className="research-icon">🔧</div>
                <h4>Framework</h4>
                <p className="research-value">PyTorch → ONNX</p>
                <p className="research-desc">Optimized runtime for web deployment</p>
              </div>
            </div>

            <div className="research-note">
              <p>
                <strong>Citation:</strong> This work implements knowledge distillation techniques for efficient skin lesion classification. 
                The distilled model maintains high accuracy while achieving significant compression and speed improvements suitable for mobile and web deployment.
              </p>
            </div>
          </div>
        </section>

        {/* RECENT ANALYSES */}
        {history.length > 0 && (
          <section className="derm-section history-section">
            <div className="section-container">
              <div className="section-header">
                <h2>Recent Analyses</h2>
              </div>

              <div className="history-grid">
                {history.map((item, idx) => (
                  <div key={idx} className="history-card">
                    <img src={item.preview} alt="analysis" className="history-img" />
                    <div className="history-info">
                      <h4>{classMetadata[item.prediction].abbreviation}</h4>
                      <p className="history-name">{classMetadata[item.prediction].name}</p>
                      <p className="history-confidence">{(item.confidence * 100).toFixed(1)}% confidence</p>
                      <p className="history-time">{item.timestamp}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
      </main>

      {/* FOOTER */}
      <footer className="derm-footer">
        <div className="footer-content">
          <p className="footer-text">
            DermAssist is a research demonstration tool. Not for clinical diagnosis. 
            Always consult a qualified dermatologist.
          </p>
          <p className="footer-secondary">
            Knowledge distillation-based approach to efficient skin lesion classification for mobile deployment
          </p>
        </div>
      </footer>
    </div>
  );
}

export default DermAssist;
