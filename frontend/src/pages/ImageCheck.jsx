import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { analyzeImage, errorMessage } from "../api.js";
import CompareSlider from "../components/CompareSlider.jsx";
import Dropzone from "../components/Dropzone.jsx";
import { EmptyReport, ErrorNote, PageHeader, ReportActions, Switch, Working } from "../components/ui.jsx";
import { useKept } from "../lib/store.js";
import VerdictScale from "../components/VerdictScale.jsx";
import { imageVerdict } from "../lib/verdicts.js";

export default function ImageCheck() {
  const location = useLocation();
  const [file, setFile] = useKept("image.file", null);
  const [preview, setPreview] = useState("");
  const [explain, setExplain] = useKept("image.explain", true);
  const [result, setResult] = useKept("image.result", null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const report = useRef(null);

  useEffect(() => {
    if (!file) return setPreview("");
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function run(input = file) {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await analyzeImage(input, explain));
      if (window.innerWidth < 1024) report.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (location.state?.run && location.state?.file) {
      setFile(location.state.file);
      run(location.state.file);
      window.history.replaceState({}, "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div>
      <PageHeader title="Check an image">
        Find out whether an image is a real photo or was generated or edited by AI, and see which parts of it the model
        looked at.
      </PageHeader>

      <div className="print-full grid items-start gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <form
          className="no-print panel space-y-5 p-5 lg:sticky lg:top-8"
          onSubmit={(e) => {
            e.preventDefault();
            run();
          }}
        >
          <Dropzone
            accept="image/*"
            onFiles={(files) => {
              const f = files.find((x) => x.type.startsWith("image/"));
              if (f) {
                setFile(f);
                setResult(null);
                setError("");
              }
            }}
          >
            {preview ? (
              <div className="p-3">
                <img src={preview} alt="Selected image" className="mx-auto max-h-80 rounded-lg object-contain" />
                <p className="t-small mt-3 truncate text-center text-graphite">{file?.name}. Click or drop to replace.</p>
              </div>
            ) : (
              <div className="px-6 py-14 text-center">
                <p className="font-medium">Drop an image here</p>
                <p className="t-small mt-1 text-graphite">or click to choose a JPG, PNG or WEBP file, up to 15 MB</p>
              </div>
            )}
          </Dropzone>
          <Switch checked={explain} onChange={setExplain} label="Show where the model looked" hint="Grad-CAM heatmap." />
          <button className="btn w-full" disabled={!file || loading}>
            {loading ? "Checking…" : "Check this image"}
          </button>
        </form>

        <section ref={report} aria-label="Report" className="min-w-0 space-y-6 scroll-mt-28">
          {loading && <Working label="Checking the image" />}
          <ErrorNote message={error} />
          {!loading && !error && !result && (
            <EmptyReport
              title="Your report will appear here"
              points={[
                "Real photo or AI-generated, with the model's confidence",
                "A heatmap you can slide over the image to see where the model looked",
              ]}
            />
          )}
          {result && (
            <>
              <div className="sheet p-6">
                <VerdictScale
                  verdict={imageVerdict(result.label)}
                  fakeProbability={result.fake_probability}
                  confidence={result.confidence}
                  leftLabel="Real photo"
                  rightLabel="AI-generated"
                />
                <div className="mt-5 flex justify-end">
                  <ReportActions
                    onReset={() => {
                      setResult(null);
                      setFile(null);
                    }}
                  />
                </div>
              </div>
              {result.explanation && (
                <div className="sheet p-6">
                  <h2 className="t-h2">Where the model looked</h2>
                  <p className="t-small mt-1 mb-5 max-w-[62ch] text-graphite">
                    Drag across the image. Warm areas influenced the “{imageVerdict(result.explanation.explained_label).title.toLowerCase()}” decision most. The image is
                    cropped to the 224 × 224 square the model sees.
                  </p>
                  <div className="mx-auto max-w-md">
                    <CompareSlider before={result.explanation.original} after={result.explanation.overlay} label="Heatmap" />
                  </div>
                </div>
              )}
              <p className="t-small max-w-[62ch] text-graphite">
                Trained on CIFAKE, which uses Stable Diffusion images. Images from other generators, or heavily compressed
                photos, can be misjudged.
              </p>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
