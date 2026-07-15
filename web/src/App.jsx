import { useCallback, useEffect, useState } from "react";
import { analyze } from "./api.js";
import Dropzone from "./components/Dropzone.jsx";
import BeforeView from "./components/BeforeView.jsx";
import AfterView from "./components/AfterView.jsx";
import ScorePanel from "./components/ScorePanel.jsx";

const OVERLAY_DEFAULTS = { subjects: true, saliency: false, cropRect: true };

export default function App() {
  const [imageURL, setImageURL] = useState(null);
  const [result, setResult] = useState(null);
  const [aspect, setAspect] = useState(null);
  const [candidateIndex, setCandidateIndex] = useState(0);
  const [overlays, setOverlays] = useState(OVERLAY_DEFAULTS);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => () => imageURL && URL.revokeObjectURL(imageURL), [imageURL]);

  const onFile = useCallback(async (file) => {
    setBusy(true);
    setError(null);
    setResult(null);
    setImageURL((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(file);
    });
    try {
      const analysis = await analyze(file);
      setResult(analysis);
      setAspect(Object.keys(analysis.crops)[0]);
      setCandidateIndex(0);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }, []);

  const candidates = result && aspect ? result.crops[aspect] : [];
  const selected = candidates[candidateIndex] ?? null;
  const tilt = result?.horizon_tilt_deg;

  return (
    <div className="app">
      <header>
        <h1>ReCompose</h1>
        <p>Upload a photo — get it professionally composed, and see why.</p>
      </header>

      <Dropzone onFile={onFile} busy={busy} hasImage={Boolean(imageURL)} />
      {error && <div className="notice error">{error}</div>}

      {result && selected && (
        <>
          {result.constraints_relaxed && (
            <div className="notice">
              This photo resisted strict recomposition — constraints were relaxed.
            </div>
          )}
          {tilt != null && Math.abs(tilt) >= 0.5 && (
            <div className="notice">
              Horizon tilted {Math.abs(tilt).toFixed(1)}° — rotate{" "}
              {tilt > 0 ? "counter-clockwise" : "clockwise"} to level it.
            </div>
          )}

          <div className="controls">
            <div className="tabs" role="tablist" aria-label="Aspect ratio">
              {Object.keys(result.crops).map((key) => (
                <button
                  key={key}
                  role="tab"
                  aria-selected={key === aspect}
                  className={key === aspect ? "tab active" : "tab"}
                  onClick={() => {
                    setAspect(key);
                    setCandidateIndex(0);
                  }}
                >
                  {key}
                </button>
              ))}
            </div>
            <div className="toggles">
              {Object.keys(OVERLAY_DEFAULTS).map((key) => (
                <label key={key}>
                  <input
                    type="checkbox"
                    checked={overlays[key]}
                    onChange={() => setOverlays((o) => ({ ...o, [key]: !o[key] }))}
                  />
                  {key === "cropRect" ? "crop outline" : key}
                </label>
              ))}
            </div>
          </div>

          <div className="views">
            <BeforeView imageURL={imageURL} result={result} crop={selected} overlays={overlays} />
            <AfterView imageURL={imageURL} scored={selected} aspect={aspect} />
          </div>

          <div className="candidates" role="group" aria-label="Alternative crops">
            {candidates.map((candidate, index) => (
              <button
                key={index}
                className={index === candidateIndex ? "chip active" : "chip"}
                onClick={() => setCandidateIndex(index)}
              >
                #{index + 1} · {(candidate.score * 100).toFixed(0)}
              </button>
            ))}
          </div>

          <ScorePanel scored={selected} />
        </>
      )}
    </div>
  );
}
