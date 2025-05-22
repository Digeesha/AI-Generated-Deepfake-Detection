import { useState } from "react";

export default function App() {
  const [mode, setMode] = useState("video");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    setFile(droppedFile);
    setResult(null);
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append(mode, file);

    setLoading(true);

    const res = await fetch(`http://localhost:5000/predict${mode === "image" ? "-image" : ""}`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setResult(data.result);
    setLoading(false);
  };

  return (
    <div style={{
      height: "100vh",
      width: "100vw",
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      backgroundColor: "#f4f4f4"
    }}>
      <div style={{
        padding: "2rem",
        background: "white",
        borderRadius: "12px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
        maxWidth: "700px",
        width: "100%",
        textAlign: "center"
      }}>
        <h1 style={{ fontSize: "2.5rem", marginBottom: "1rem" }}>🎭 Deepfake Detector</h1>

        {/* Mode Toggle */}
        <div style={{ marginBottom: "1rem" }}>
          <label style={{ marginRight: "1rem" }}>
            <input type="radio" value="video" checked={mode === "video"} onChange={() => setMode("video")} />
            Video
          </label>
          <label>
            <input type="radio" value="image" checked={mode === "image"} onChange={() => setMode("image")} />
            Image
          </label>
        </div>

        {/* Drop Zone */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          style={{
            border: "2px dashed #ccc",
            padding: "2rem",
            borderRadius: "10px",
            backgroundColor: "#fafafa",
            marginBottom: "1rem"
          }}
        >
          {file ? <strong>{file.name}</strong> : (
            <p>Drag & drop a {mode} file here, or select below</p>
          )}
        </div>

        {/* File Input */}
        <input
          type="file"
          accept={mode === "video" ? "video/*" : "image/*"}
          onChange={handleFileChange}
        />
        <br /><br />

        {/* Upload Button */}
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          style={{
            padding: "0.6rem 1.5rem",
            fontSize: "1rem",
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer"
          }}
        >
          {loading ? "Analyzing..." : `Detect ${mode === "video" ? "Video" : "Image"}`}
        </button>

        {/* Loading State */}
        {loading && <p style={{ marginTop: "1rem" }}>Processing {mode}...</p>}

        {/* Result */}
        {result?.label && result?.score !== undefined && (
          <div style={{ marginTop: "2rem" }}>
            <h2>
              Result:{" "}
              <span style={{
                color: result.label === "FAKE" ? "red" : "green",
                fontWeight: "bold"
              }}>
                {result.label} - {(result.score * 100).toFixed(2)}%
              </span>
            </h2>
          </div>
        )}
      </div>
    </div>
  );
}
