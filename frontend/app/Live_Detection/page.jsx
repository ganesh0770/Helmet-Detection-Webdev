"use client"
import { useState, useRef, useEffect } from "react";

function Spinner() {
  return (
    <div className="w-12 h-12 border-4 border-blue-400 border-t-transparent rounded-full animate-spin mx-auto"></div>
  );
}

export default function UploadPage() {
   const [mode, setMode] = useState("upload");
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
    const canvasRef = useRef(null);
  const [boxes, setBoxes] = useState([]);
  const [person_no, setperson_no] = useState(null);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);


    useEffect(() => {
    if (mode === "camera") {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        })
        .catch(err => {
          console.error("Camera error:", err);
          setError("Unable to access camera.");
          setMode("upload");
        });
    } else {
      // stop camera if switching mode
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      }
    }
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      }
    };
  }, [mode]);


  const handleUploadChange = (e) => {
    const file = e.target.files[0];

    if (!file || !file.type.startsWith("video/")) {
      setError("Please upload a valid video file.");
      return;
    }

    setVideoFile(file);
    setVideoPreviewUrl(URL.createObjectURL(file));
    setError(null);
    setResult(null);
    setBoxes([]);
  };
  const handleDrop = (e) => {
    e.preventDefault();
    handleUploadChange({ target: { files: e.dataTransfer.files } });
  };
  const handleDetect = async () => {
    if (!videoFile) {
      setError("Please upload a video first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setBoxes([]);

    try {
      const formData = new FormData();
      formData.append("video", videoFile);

      const res = await fetch("http://127.0.0.1:8000/predict_video", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Video processing failed");
      }

      const data = await res.json();

      /*
        Expected response:
        {
          helmet: true/false,
          Total_helmet: number,
          boxes: [],
          result_video: "path or url"
        }
      */

      setperson_no(data.Total_helmet || 0);
      setResult(data.helmet ? "Helmet Detected" : "No Helmet Detected");
      setBoxes(data.boxes || []);

    } catch (err) {
      console.error(err);
      setError("Something went wrong while uploading video.");
    } finally {
      setLoading(false);
    }
  };

  
  




  return (
    <div className="min-h-screen bg-gray-950 text-white px-4 py-10 justify-center pt-30">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8">Upload video to Detect Helmet </h1>

        {/* Mode Switch */}
        <div className="flex justify-center mb-6 space-x-4">
          <button
            onClick={() => { setMode("camera"); setError(null); }}
            className={`px-4 py-2 rounded ${mode === "camera" ? "bg-blue-600" : "bg-gray-800 hover:bg-gray-700"}`}
          >
            Use Camera
          </button>
        </div>

        {/* Upload or Camera */}
        <div
          className="border-2 border-dashed border-gray-600 rounded-xl p-8 text-center hover:border-blue-500 transition mb-6"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          {mode === "upload" ? (
            <>
              <input
                type="file"
                accept="video/*"

                onChange={handleUploadChange}
                className="hidden"
                id="file-input"
              />
              <label htmlFor="file-input" className="cursor-pointer block">
                <div className="text-gray-300 text-lg mb-4">
                  {videoPreviewUrl ? "Change / Upload New Image" : "Click or Drag & Drop Video Here"}
                </div>
                <div className="text-gray-500 text-sm">(MP4)</div>
              </label>
            </>
          ) : (
            <div className="flex flex-col items-center">
              <video ref={videoRef} className="rounded-xl w-full max-h-80 mb-4 bg-black" />
              <button
                onClick={captureFromCamera}
                className="px-6 py-2 bg-green-600 hover:bg-green-700 rounded-lg"
              >
                Capture Photo
              </button>
            </div>
          )}
        </div>

        {/* Preview + Detection */}
        {videoPreviewUrl && (
          <video
            src={videoPreviewUrl}
            controls
            className="max-w-full rounded"
          />
        )}


        {/* Action Buttons + Result */}
        <div className="text-center">
          <button
            onClick={handleDetect}
            disabled={loading || !videoPreviewUrl}
            className={`px-8 py-3 rounded-xl text-lg font-semibold shadow-lg transition
              ${loading || !videoPreviewUrl ? "bg-gray-700 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"}`}
          >
            {loading ? <Spinner /> : "Detect Helmet"}
          </button>
        </div>

        {result && (
          <div className="mt-8 p-5 rounded-xl text-center text-2xl font-bold border border-gray-700 bg-gray-800/60">
            {result}  {person_no}
          </div>
        )}

        {error && (
          <div className="mt-4 p-4 bg-red-600 text-white rounded-md text-center">
            {error}
          </div>
        )}

        {/* {history.length > 0 && (
          <div className="mt-12">
            <h2 className="text-2xl font-semibold mb-4">History</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {history.map(item => (
                <div key={item.id} className="bg-gray-800 rounded-xl overflow-hidden shadow-lg">
                  <div className="relative">
                    <img src={item.url} alt="" className="w-full h-48 object-cover" />
                    {item.boxes.map((b, i) => (
                      <div
                        key={i}
                        className="absolute border-2 border-yellow-400"
                        style={{
                          left: `${b.x * 100}%`,
                          top: `${b.y * 100}%`,
                          width: `${b.width * 100}%`,
                          height: `${b.height * 100}%`,
                        }}
                      />
                    ))}
                  </div>
                  <div className="p-4 text-center">
                    <span className={`font-semibold ${item.result ? "text-green-400" : "text-red-400"}`}>
                      {item.result ? "Helmet" : "No Helmet"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )} */}
      </div>


      <canvas ref={canvasRef} className="hidden"></canvas>
    </div>
  );
}
