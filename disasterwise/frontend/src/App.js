import React, { useState } from "react";
import axios from "axios";

function App() {
  const [preImage, setPreImage] = useState(null);
  const [postImage, setPostImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    if (!preImage || !postImage) {
      alert("Please upload both pre-disaster and post-disaster images.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("pre_image", preImage);
    formData.append("post_image", postImage);

    try {
      console.log("Sending request to backend at http://127.0.0.1:5000/predict");
      const res = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      console.log("Response received:", res.data);
      setResult(res.data);
    } catch (err) {
      console.error("API connection error:", err);
      setError("Failed to connect to backend server. Make sure it is running on port 5000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
      maxWidth: "700px",
      margin: "40px auto",
      padding: "30px",
      border: "none",
      borderRadius: "20px",
      boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
      backgroundColor: "#ffffff",
      color: "#333"
    }}>
      <header style={{ textAlign: "center", marginBottom: "30px" }}>
        <h1 style={{ margin: "0", fontSize: "2.5rem", color: "#2c3e50" }}>🌍 DISASTERWISE</h1>
        <p style={{ color: "#7f8c8d", fontSize: "1.1rem" }}>Satellite Image Damage Analysis</p>
      </header>
      
      <main>
        <div style={{ 
          display: "grid", 
          gridTemplateColumns: "1fr 1fr", 
          gap: "20px", 
          marginBottom: "30px" 
        }}>
          <div style={{ 
            padding: "20px", 
            border: "2px dashed #bdc3c7", 
            borderRadius: "15px",
            textAlign: "center"
          }}>
            <p style={{ fontWeight: "600", marginBottom: "10px" }}>Pre-Disaster</p>
            <input 
              type="file" 
              accept="image/*"
              onChange={(e) => setPreImage(e.target.files[0])} 
              style={{ width: "100%" }}
            />
            {preImage && <p style={{ fontSize: "0.8rem", marginTop: "5px", color: "#27ae60" }}>✓ {preImage.name}</p>}
          </div>

          <div style={{ 
            padding: "20px", 
            border: "2px dashed #bdc3c7", 
            borderRadius: "15px",
            textAlign: "center"
          }}>
            <p style={{ fontWeight: "600", marginBottom: "10px" }}>Post-Disaster</p>
            <input 
              type="file" 
              accept="image/*"
              onChange={(e) => setPostImage(e.target.files[0])} 
              style={{ width: "100%" }}
            />
            {postImage && <p style={{ fontSize: "0.8rem", marginTop: "5px", color: "#27ae60" }}>✓ {postImage.name}</p>}
          </div>
        </div>

        <div style={{ textAlign: "center" }}>
          <button 
            onClick={handleSubmit} 
            disabled={loading}
            style={{
              padding: "15px 40px",
              fontSize: "1.1rem",
              fontWeight: "bold",
              backgroundColor: loading ? "#95a5a6" : "#3498db",
              color: "white",
              border: "none",
              borderRadius: "50px",
              cursor: loading ? "not-allowed" : "pointer",
              transition: "transform 0.2s, background-color 0.2s",
              boxShadow: "0 4px 15px rgba(52, 152, 219, 0.3)"
            }}
            onMouseOver={(e) => !loading && (e.currentTarget.style.backgroundColor = "#2980b9")}
            onMouseOut={(e) => !loading && (e.currentTarget.style.backgroundColor = "#3498db")}
          >
            {loading ? "Analyzing..." : "Analyze Damage"}
          </button>
        </div>

        {loading && (
          <div style={{ textAlign: "center", marginTop: "30px" }}>
            <div style={{ 
              display: "inline-block", 
              width: "40px", 
              height: "40px", 
              border: "4px solid #f3f3f3", 
              borderTop: "4px solid #3498db", 
              borderRadius: "50%", 
              animation: "spin 1s linear infinite" 
            }}></div>
            <p style={{ color: "#7f8c8d", marginTop: "10px" }}>Processing satellite imagery...</p>
            <style>{`
              @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            `}</style>
          </div>
        )}

        {error && (
          <div style={{ 
            marginTop: "30px", 
            padding: "15px", 
            backgroundColor: "#fdecea", 
            borderLeft: "5px solid #e74c3c", 
            borderRadius: "5px", 
            color: "#c0392b" 
          }}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <div style={{ 
            marginTop: "40px", 
            padding: "25px", 
            borderRadius: "15px", 
            backgroundColor: "#f8f9fa",
            border: "1px solid #e9ecef"
          }}>
            <h3 style={{ marginTop: "0", color: "#2c3e50", borderBottom: "2px solid #dee2e6", paddingBottom: "10px" }}>Analysis Results</h3>
            <div style={{ display: "flex", justifyContent: "space-around", textAlign: "center", margin: "20px 0" }}>
              <div>
                <p style={{ margin: "0", color: "#7f8c8d" }}>Damage</p>
                <p style={{ fontSize: "2rem", fontWeight: "bold", margin: "5px 0", color: "#2c3e50" }}>{result.damage_percentage}%</p>
              </div>
              <div>
                <p style={{ margin: "0", color: "#7f8c8d" }}>Alert Status</p>
                <p style={{ 
                  fontSize: "1.2rem", 
                  fontWeight: "bold", 
                  margin: "10px 0",
                  padding: "5px 20px",
                  borderRadius: "20px",
                  color: "white",
                  backgroundColor: result.alert_level === "RED" ? "#e74c3c" : result.alert_level === "YELLOW" ? "#f1c40f" : "#2ecc71"
                }}>
                  {result.alert_level}
                </p>
              </div>
            </div>
            <div style={{ padding: "15px", backgroundColor: "#fff", borderRadius: "10px", border: "1px solid #e9ecef" }}>
              <p style={{ margin: "0" }}><strong>Expert Summary:</strong> {result.message}</p>
            </div>
            <p style={{ fontSize: "0.8rem", color: "#bdc3c7", textAlign: "right", marginTop: "15px" }}>Report ID: {result.timestamp}</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
