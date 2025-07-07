import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('url');
  const [url, setUrl] = useState('');
  const [customPrompt, setCustomPrompt] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageDrop = (e) => {
    console.log('Image dropped');
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      console.log('Valid image file:', file.name, file.size, 'bytes');
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        console.log('Image preview generated');
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
      setError(null);
    } else {
      console.log('Invalid file type:', file?.type);
      setError('Please select a valid image file');
    }
  };

  const handleImageSelect = (e) => {
    console.log('Image selected from input');
    const file = e.target.files[0];
    if (file) {
      console.log('Selected file:', file.name, file.size, 'bytes', file.type);
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        console.log('Image preview generated from input');
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
      setError(null);
    }
  };

  const handleUrlAnalysis = async () => {
    console.log("=== URL ANALYSIS STARTED ===");
    console.log("URL:", url);
    console.log("Custom prompt:", customPrompt);

    if (!url) {
      setError('Please enter a valid URL');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      console.log('Sending request to /analyze-url...');
      const response = await axios.post('/analyze-url', {
        url: url,
        prompt: customPrompt
      });
      
      console.log('URL analysis response:', response.data);
      setResults(response.data);
    } catch (error) {
      console.error('URL analysis error:', error);
      console.error('Error response:', error.response?.data);
      setError(error.response?.data?.error || 'Failed to analyze URL');
    } finally {
      setLoading(false);
    }
  };

  const handleImageAnalysis = async () => {
    console.log("=== IMAGE ANALYSIS STARTED ===");
    console.log("Selected image:", selectedImage);
    console.log("Custom prompt:", customPrompt);
    
    if (!selectedImage) {
      console.log('No image selected');
      setError('Please select an image');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          console.log("Image converted to base64, length:", e.target.result.length);
          
          const requestData = {
            image: e.target.result,
            prompt: customPrompt
          };
          
          console.log("Sending request to /analyze-image...");
          console.log("Request data keys:", Object.keys(requestData));
          console.log("Image data preview:", e.target.result.substring(0, 100) + "...");
          
          const response = await axios.post('/analyze-image', requestData, {
            headers: {
              'Content-Type': 'application/json',
            },
            timeout: 30000
          });
          
          console.log("Response received:", response.data);
          setResults(response.data);
          
        } catch (error) {
          console.error("Analysis error:", error);
          console.error("Error response:", error.response?.data);
          console.error("Error status:", error.response?.status);
          setError(error.response?.data?.error || 'Failed to analyze image');
        } finally {
          setLoading(false);
        }
      };
      
      reader.onerror = (error) => {
        console.error("FileReader error:", error);
        setError('Failed to read image file');
        setLoading(false);
      };
      
      console.log("Starting to read image file...");
      reader.readAsDataURL(selectedImage);
      
    } catch (error) {
      console.error("Outer error:", error);
      setError('Failed to process image');
      setLoading(false);
    }
  };

  const testBackendConnection = async () => {
    try {
      console.log('Testing backend connection...');
      const response = await axios.get('/health');
      console.log('Backend health check:', response.data);
      alert(`Backend Status: ${JSON.stringify(response.data, null, 2)}`);
    } catch (error) {
      console.error('Backend connection failed:', error);
      alert('Backend connection failed. Check if backend is running on port 5000.');
    }
  };

  const clearResults = () => {
    setResults(null);
    setError(null);
  };

  const copyToClipboard = (text) => {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).then(() => {
        alert('ALT text copied to clipboard!');
      }).catch(err => {
        console.error('Failed to copy text: ', err);
        fallbackCopyToClipboard(text);
      });
    } else {
      fallbackCopyToClipboard(text);
    }
  };

  const fallbackCopyToClipboard = (text) => {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
      document.execCommand('copy');
      alert('ALT text copied to clipboard!');
    } catch (err) {
      console.error('Fallback: Unable to copy', err);
      alert('Unable to copy to clipboard. Please copy manually.');
    }
    document.body.removeChild(textArea);
  };

  return (
    <div className="App">
      <div className="container mt-5">
        <div className="row justify-content-center">
          <div className="col-lg-10">
            <div className="card shadow-lg">
              <div className="card-header bg-primary text-white">
                <h1 className="h3 mb-0 text-center">
                  <i className="fas fa-eye me-2"></i>
                  ALT Tag Generator
                </h1>
                <p className="text-center mb-0 mt-2">
                  Generate accessibility-friendly ALT tags using AI analysis
                </p>
                <div className="text-center mt-2">
                  <button 
                    className="btn btn-sm btn-light me-2" 
                    onClick={testBackendConnection}
                  >
                    Test Backend
                  </button>
                  {results && (
                    <button 
                      className="btn btn-sm btn-outline-light" 
                      onClick={clearResults}
                    >
                      Clear Results
                    </button>
                  )}
                </div>
              </div>
              
              <div className="card-body">
                {/* Navigation Tabs */}
                <ul className="nav nav-tabs mb-4" role="tablist">
                  <li className="nav-item" role="presentation">
                    <button
                      className={`nav-link ${activeTab === 'url' ? 'active' : ''}`}
                      onClick={() => setActiveTab('url')}
                      type="button"
                    >
                      <i className="fas fa-link me-2"></i>
                      Analyze Website
                    </button>
                  </li>
                  <li className="nav-item" role="presentation">
                    <button
                      className={`nav-link ${activeTab === 'image' ? 'active' : ''}`}
                      onClick={() => setActiveTab('image')}
                      type="button"
                    >
                      <i className="fas fa-image me-2"></i>
                      Upload Image
                    </button>
                  </li>
                </ul>

                {/* URL Analysis Tab */}
                {activeTab === 'url' && (
                  <div className="tab-content">
                    <div className="mb-4">
                      <label htmlFor="urlInput" className="form-label">
                        <strong>Website URL</strong>
                      </label>
                      <input
                        type="url"
                        className="form-control"
                        id="urlInput"
                        placeholder="https://example.com"
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                      />
                    </div>
                    
                    <div className="mb-4">
                      <label htmlFor="customPrompt" className="form-label">
                        <strong>Custom Instructions (Optional)</strong>
                      </label>
                      <textarea
                        className="form-control"
                        id="customPrompt"
                        rows="3"
                        placeholder="Add any specific requirements or context for ALT tag generation..."
                        value={customPrompt}
                        onChange={(e) => setCustomPrompt(e.target.value)}
                      />
                    </div>
                    
                    <button
                      className="btn btn-primary btn-lg w-100"
                      onClick={handleUrlAnalysis}
                      disabled={loading}
                    >
                      {loading ? (
                        <>
                          <span className="spinner-border spinner-border-sm me-2"></span>
                          Analyzing Website...
                        </>
                      ) : (
                        <>
                          <i className="fas fa-search me-2"></i>
                          Analyze Website
                        </>
                      )}
                    </button>
                  </div>
                )}

                {/* Image Upload Tab */}
                {activeTab === 'image' && (
                  <div className="tab-content">
                    <div className="mb-4">
                      <label className="form-label">
                        <strong>Upload Image</strong>
                      </label>
                      <div
                        className="upload-area border-dashed border-2 border-primary p-4 text-center"
                        onDrop={handleImageDrop}
                        onDragOver={(e) => e.preventDefault()}
                        style={{ minHeight: '200px', cursor: 'pointer' }}
                        onClick={() => document.getElementById('imageInput').click()}
                      >
                        {imagePreview ? (
                          <div>
                            <img
                              src={imagePreview}
                              alt="Preview"
                              className="img-fluid mb-3"
                              style={{ maxHeight: '200px' }}
                            />
                            <p className="text-muted">
                              {selectedImage?.name} ({(selectedImage?.size / 1024).toFixed(1)} KB)
                            </p>
                            <p className="text-muted">Click to change image</p>
                          </div>
                        ) : (
                          <div>
                            <i className="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                            <p className="text-muted">
                              Drag & drop an image here or click to select
                            </p>
                            <small className="text-muted">
                              Supports JPG, PNG, GIF, WebP
                            </small>
                          </div>
                        )}
                        <input
                          type="file"
                          id="imageInput"
                          accept="image/*"
                          onChange={handleImageSelect}
                          className="d-none"
                        />
                      </div>
                    </div>

                    {/* Enhanced Description Guide */}
                    <div className="mb-4">
                      <label htmlFor="imageCustomPrompt" className="form-label">
                        <strong>Describe Your Image</strong>
                        <span className="text-muted"> (For Better ALT Text)</span>
                      </label>
                      <textarea
                        className="form-control"
                        id="imageCustomPrompt"
                        rows="4"
                        placeholder="Describe what you see in the image:

• What is the main subject? (person, object, scene)
• What are they doing? (action, pose, expression)  
• What's the setting/background?
• Any text or important details?
• What is the purpose of this image on your website?

Example: 'A smiling woman in a blue dress standing in front of a modern office building, used as a profile photo for the About Us page'"
                        value={customPrompt}
                        onChange={(e) => setCustomPrompt(e.target.value)}
                      />
                      <small className="text-muted">
                        💡 <strong>Tip:</strong> The more detailed your description, the better and more accurate the ALT text will be!
                      </small>
                    </div>

                    {/* Description Helper Card */}
                    {imagePreview && (
                      <div className="mb-4">
                        <div className="card border-info">
                          <div className="card-header bg-info text-white">
                            <h6 className="mb-0">📝 Quick Description Guide</h6>
                          </div>
                          <div className="card-body">
                            <p className="small mb-2">Look at your image and describe:</p>
                            <div className="row">
                              <div className="col-md-6">
                                <ul className="small mb-0">
                                  <li><strong>Main subject:</strong> What/who is the focus?</li>
                                  <li><strong>Action/pose:</strong> What are they doing?</li>
                                  <li><strong>Setting:</strong> Where is this happening?</li>
                                </ul>
                              </div>
                              <div className="col-md-6">
                                <ul className="small mb-0">
                                  <li><strong>Mood/style:</strong> Professional, casual, artistic?</li>
                                  <li><strong>Purpose:</strong> How will this be used on your site?</li>
                                  <li><strong>Key details:</strong> Important text or elements?</li>
                                </ul>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    <button
                      className="btn btn-primary btn-lg w-100"
                      onClick={handleImageAnalysis}
                      disabled={loading || !selectedImage}
                    >
                      {loading ? (
                        <>
                          <span className="spinner-border spinner-border-sm me-2"></span>
                          Generating ALT Tags...
                        </>
                      ) : (
                        <>
                          <i className="fas fa-magic me-2"></i>
                          Generate ALT Tags
                        </>
                      )}
                    </button>
                  </div>
                )}

                {/* Loading State */}
                {loading && (
                  <div className="text-center mt-4">
                    <div className="spinner-border text-primary" role="status">
                      <span className="visually-hidden">Loading...</span>
                    </div>
                    <p className="mt-2 text-muted">
                      {activeTab === 'url' ? 'Analyzing website with browser automation...' : 'Processing image with enhanced AI analysis...'}
                    </p>
                  </div>
                )}

                {/* Error Display */}
                {error && (
                  <div className="alert alert-danger mt-4" role="alert">
                    <i className="fas fa-exclamation-triangle me-2"></i>
                    <strong>Error:</strong> {error}
                  </div>
                )}

                {/* Results Display */}
                {results && (
                  <div className="mt-4">
                    <div className="card">
                      <div className="card-header bg-success text-white">
                        <h5 className="mb-0">
                          <i className="fas fa-check-circle me-2"></i>
                          Analysis Results
                          {results.confidence_level && (
                            <span className={`badge ms-2 ${
                              results.confidence_level === 'high' ? 'bg-success' : 
                              results.confidence_level === 'medium' ? 'bg-warning' : 'bg-secondary'
                            }`}>
                              {results.confidence_level} confidence
                            </span>
                          )}
                        </h5>
                      </div>
                      <div className="card-body">
                        {/* Success message */}
                        {results.success && (
                          <div className="alert alert-success">
                            <i className="fas fa-check me-2"></i>
                            {results.message || 'Analysis completed successfully!'}
                          </div>
                        )}

                        {/* Main ALT Text Result */}
                        {results.alt_text && (
                          <div className="row">
                            <div className="col-md-8">
                              <h6><strong>🎯 Suggested ALT Text:</strong></h6>
                              <div className="alert alert-info">
                                <code style={{ fontSize: '1.1em', fontWeight: 'bold' }}>
                                  {results.alt_text}
                                </code>
                                <small className="d-block mt-2 text-muted">
                                  ({results.alt_text.length} characters)
                                </small>
                              </div>
                            </div>
                            <div className="col-md-4">
                              <h6><strong>📊 Quality Metrics:</strong></h6>
                              <ul className="list-unstyled">
                                {results.image_type && <li><strong>Type:</strong> {results.image_type}</li>}
                                {results.accessibility_score && (
                                  <li><strong>Score:</strong> {results.accessibility_score}/10</li>
                                )}
                                {results.context_quality && (
                                  <li><strong>Context:</strong> {results.context_quality}</li>
                                )}
                              </ul>
                            </div>
                          </div>
                        )}

                        {/* Technical Features (if available) */}
                        {results.technical_features && (
                          <div className="mt-3">
                            <h6><strong>🔧 Technical Features:</strong></h6>
                            <div className="row">
                              {results.technical_features.dimensions && (
                                <div className="col-md-3">
                                  <small><strong>Dimensions:</strong> {results.technical_features.dimensions}</small>
                                </div>
                              )}
                              {results.technical_features.aspect_ratio && (
                                <div className="col-md-3">
                                  <small><strong>Aspect Ratio:</strong> {results.technical_features.aspect_ratio}</small>
                                </div>
                              )}
                              {results.technical_features.brightness && (
                                <div className="col-md-3">
                                  <small><strong>Brightness:</strong> {results.technical_features.brightness}</small>
                                </div>
                              )}
                              {results.technical_features.color_mode && (
                                <div className="col-md-3">
                                  <small><strong>Color Mode:</strong> {results.technical_features.color_mode}</small>
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        {/* Enhanced Results for Image Analysis */}
                        {results.technical_insights && (
                          <div className="mt-3">
                            <h6><strong>🔍 Technical Analysis:</strong></h6>
                            <p className="text-muted">{results.technical_insights}</p>
                          </div>
                        )}

                        {/* File Information */}
                        {results.file_info && (
                          <div className="mt-3">
                            <h6><strong>📁 File Information:</strong></h6>
                            <div className="row">
                              <div className="col-md-4">
                                <small><strong>Size:</strong> {results.file_info.size_kb} KB</small>
                              </div>
                              <div className="col-md-4">
                                <small><strong>Format:</strong> {results.file_info.format.toUpperCase()}</small>
                              </div>
                              <div className="col-md-4">
                                <small><strong>Detected Type:</strong> {results.file_info.likely_type}</small>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Explanation */}
                        {results.explanation && (
                          <div className="mt-3">
                            <h6><strong>💡 Explanation:</strong></h6>
                            <p>{results.explanation}</p>
                          </div>
                        )}

                        {/* Improvements */}
                        {results.improvements && results.improvements.length > 0 && (
                          <div className="mt-3">
                            <h6><strong>🚀 Suggestions for Better Results:</strong></h6>
                            <ul>
                              {results.improvements.map((improvement, index) => (
                                <li key={index} className="mb-1">{improvement}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Visual Elements (if detected) */}
                        {results.visual_elements && results.visual_elements.length > 0 && (
                          <div className="mt-3">
                            <h6><strong>👁️ Detected Visual Elements:</strong></h6>
                            <div className="d-flex flex-wrap gap-1">
                              {results.visual_elements.map((element, index) => (
                                <span key={index} className="badge bg-secondary">{element}</span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Copy to Clipboard Button */}
                        {results.alt_text && (
                          <div className="mt-3 d-grid">
                            <button 
                              className="btn btn-outline-primary"
                              onClick={() => copyToClipboard(results.alt_text)}
                            >
                              <i className="fas fa-copy me-2"></i>
                              Copy ALT Text to Clipboard
                            </button>
                          </div>
                        )}

                        {/* URL Analysis Results */}
                        {results.analysis_result && (
                          <div className="mt-3">
                            <h6><strong>🌐 Website Analysis:</strong></h6>
                            <div className="alert alert-light">
                              <pre style={{ 
                                whiteSpace: 'pre-wrap', 
                                fontSize: '0.9em',
                                maxHeight: '400px',
                                overflow: 'auto'
                              }}>
                                {results.analysis_result}
                              </pre>
                            </div>
                          </div>
                        )}

                        {/* Raw results for debugging */}
                        <details className="mt-4">
                          <summary className="btn btn-sm btn-outline-secondary">
                            <i className="fas fa-code me-2"></i>
                            Show Technical Details (Debug)
                          </summary>
                          <pre className="bg-light p-3 rounded mt-2" style={{ 
                            fontSize: '0.8em', 
                            maxHeight: '300px', 
                            overflow: 'auto',
                            border: '1px solid #dee2e6'
                          }}>
                            {JSON.stringify(results, null, 2)}
                          </pre>
                        </details>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;