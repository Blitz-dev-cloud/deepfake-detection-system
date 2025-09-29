'use client';

import { useState } from 'react';

const MAX_SIZE_MB = 100;
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'video/mp4', 'video/webm'];

export default function DeepfakeDetector() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    setResult(null); // reset result on new file selection
    if (!file) return;
    if (!ALLOWED_TYPES.includes(file.type)) {
      alert('Unsupported file type. Please upload jpeg, png, mp4, or webm.');
      setFile(null);
      return;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      alert(`File too large. Maximum allowed size is ${MAX_SIZE_MB} MB.`);
      setFile(null);
      return;
    }
    setFile(file);
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a file.');
      return;
    }
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    // Select endpoint based on file type
    let apiEndpoint = '';
    if (file.type.startsWith('image')) {
      apiEndpoint = 'http://localhost:8000/predict_image';
    } else if (file.type.startsWith('video')) {
      apiEndpoint = 'http://localhost:8000/predict_video';
    } else {
      alert('Unsupported file type');
      setLoading(false);
      return;
    }

    try {
      const res = await fetch(apiEndpoint, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      setResult({ error: 'Failed to fetch from server' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-tr from-indigo-900 via-gray-900 to-black text-gray-100 font-sans flex flex-col justify-center items-center px-5 py-16">
      <header className="mb-12 text-center max-w-4xl">
        <h1 className="text-6xl font-extrabold tracking-tight mb-4 leading-tight drop-shadow-lg select-none">
          Deepfake Detector
        </h1>
        <p className="text-lg text-gray-400 max-w-xl mx-auto">
          A powerful AI tool developed by{' '}
          <span className="font-semibold text-indigo-300 cursor-default">
            Sounak & Trisha
          </span>{' '}
          to detect deepfakes with professional precision.
        </p>
      </header>

      <main className="bg-gray-800 bg-opacity-80 rounded-xl shadow-2xl p-12 w-full max-w-lg flex flex-col items-center space-y-8">
        <div className="w-full">
          <label
            htmlFor="file-upload"
            className="relative cursor-pointer rounded-md border-2 border-indigo-500 border-dashed p-6 flex flex-col items-center justify-center hover:bg-indigo-600 transition-colors"
          >
            <svg
              className="w-16 h-16 mb-3 text-indigo-400"
              fill="none"
              stroke="currentColor"
              strokeWidth={1.5}
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3 15a4 4 0 0 0 4 4h10a4 4 0 0 0 0-8H5a4 4 0 0 0-2 7Z"
              />
              <path strokeLinecap="round" strokeLinejoin="round" d="M7 9v4" />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M7 13l3 3m0 0l3-3m-3 3V9"
              />
            </svg>
            <p className="text-indigo-300 font-semibold">
              {file ? file.name : 'Click to select an image or video file'}
            </p>
            <input
              id="file-upload"
              type="file"
              accept={ALLOWED_TYPES.join(',')}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              onChange={handleChange}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !loading && file) {
                  e.preventDefault();
                  handleUpload();
                }
              }}
            />
          </label>
        </div>

        {/* File preview */}
        {file && (
          <div className="w-full flex justify-center mb-4">
            {file.type.startsWith('image') ? (
              <img
                src={URL.createObjectURL(file)}
                alt="preview"
                className="max-h-48 rounded-md"
              />
            ) : (
              <video
                src={URL.createObjectURL(file)}
                controls
                className="max-h-48 rounded-md"
              />
            )}
          </div>
        )}

        <button
          onClick={handleUpload}
          disabled={loading || !file}
          className="w-full py-3 bg-gradient-to-r from-indigo-500 to-purple-600 font-bold tracking-wide rounded-lg shadow-lg hover:scale-105 transform transition disabled:opacity-50"
        >
          {loading ? (
            <span className="flex items-center justify-center space-x-2">
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v8H4z"
                ></path>
              </svg>
              <span>Detecting...</span>
            </span>
          ) : (
            'Detect Deepfake'
          )}
        </button>

        {/* Result display */}
        {result && (
          <>
            {result.error ? (
              <p className="text-red-500 font-semibold mt-4">{result.error}</p>
            ) : (
              <section className="bg-gray-700 rounded-md p-6 w-full max-w-md text-center mt-4">
                <h3 className="text-2xl font-semibold mb-3 text-indigo-300">
                  Result
                </h3>
                <p className="mb-2 text-xl">
                  Prediction:{' '}
                  <span
                    className={`font-bold ${
                      result.prediction === 'FAKE'
                        ? 'text-red-500'
                        : 'text-green-400'
                    }`}
                  >
                    {result.prediction}
                  </span>
                </p>
                <p>Confidence: {(result.probability * 100).toFixed(2)}%</p>
                {/* Show additional frames analyzed if response has frames */}
                {'frames_processed' in result && (
                  <p>Frames Processed: {result.frames_processed}</p>
                )}
              </section>
            )}
          </>
        )}
      </main>

      <footer className="mt-16 text-gray-500 text-sm text-center">
        &copy; 2025 Deepfake Detector. AI project by Sounak & Trisha.
      </footer>
    </div>
  );
}
