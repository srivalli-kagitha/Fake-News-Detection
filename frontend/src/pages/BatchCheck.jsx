import { useState } from "react";
import { batchImages, batchText, downloadUrl, errorMessage } from "../api.js";
import Dropzone from "../components/Dropzone.jsx";
import { Chip, ErrorNote, LanguagePicker, PageHeader, Switch, Working } from "../components/ui.jsx";
import { useKept } from "../lib/store.js";
import { TONE_BAR, anyVerdict, pct } from "../lib/verdicts.js";

function Results({ result }) {
  const isText = result.kind === "text";
  const entries = Object.entries(result.counts);
  return (
    <section className="sheet mt-8 p-6" aria-label="Batch results">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="t-h2">
            {result.total} {isText ? "texts" : "images"} checked
          </h2>
          <div className="mt-4 flex h-3 w-72 max-w-full gap-[3px] overflow-hidden rounded-full" aria-hidden="true">
            {entries.map(([label, n]) => (
              <span key={label} className={TONE_BAR[anyVerdict(label).tone]} style={{ width: `${(n / result.total) * 100}%` }} />
            ))}
          </div>
          <p className="t-small mt-2 flex flex-wrap gap-x-4 text-graphite">
            {entries.map(([label, n]) => (
              <span key={label}>
                {n} {anyVerdict(label).title.toLowerCase()}
              </span>
            ))}
          </p>
        </div>
        <a href={downloadUrl(result.job_id)} className="btn">
          Download results (CSV)
        </a>
      </div>

      <div className="mt-6 max-h-[32rem] overflow-auto rounded-[10px] border border-rule-soft">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 bg-paper text-graphite">
            <tr>
              <th scope="col" className="px-4 py-2.5 font-medium">Row</th>
              <th scope="col" className="px-4 py-2.5 font-medium">{isText ? "Text" : "File"}</th>
              <th scope="col" className="px-4 py-2.5 font-medium">Result</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">Confidence</th>
              {result.results[0]?.claim_verdict !== undefined && (
                <th scope="col" className="px-4 py-2.5 font-medium">Claims</th>
              )}
            </tr>
          </thead>
          <tbody>
            {result.results.map((r) => {
              const v = r.prediction === "Error" ? { title: "Couldn't read", tone: "neutral" } : anyVerdict(r.prediction);
              return (
                <tr key={r.row} className="border-t border-rule-soft align-top">
                  <td className="px-4 py-3 text-graphite">{r.row}</td>
                  <td className="max-w-md px-4 py-3">
                    <span className={`line-clamp-2 ${isText ? "font-serif" : ""}`} title={isText ? r.text : r.filename}>
                      {isText ? r.text : r.filename}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <Chip tone={v.tone}>{v.title}</Chip>
                  </td>
                  <td className="px-4 py-3 text-right">{pct(r.confidence)}</td>
                  {r.claim_verdict !== undefined && (
                    <td className="px-4 py-3">
                      <Chip tone={anyVerdict(r.claim_verdict).tone}>{anyVerdict(r.claim_verdict).title}</Chip>
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function BatchCheck() {
  const [csv, setCsv] = useState(null);
  const [language, setLanguage] = useState("auto");
  const [verify, setVerify] = useState(false);
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useKept("batch.result", null);

  async function run(kind) {
    setLoading(kind);
    setError("");
    setResult(null);
    try {
      const data = kind === "text" ? await batchText(csv, language, verify) : await batchImages(images);
      setResult({ ...data, kind });
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading("");
    }
  }

  return (
    <div>
      <PageHeader title="Check many at once">
        Upload a spreadsheet of news texts or a set of images. Every item is checked and you can download the results as
        a CSV file.
      </PageHeader>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="panel flex flex-col p-5">
          <h2 className="t-h2">News texts</h2>
          <p className="t-small mt-1 text-graphite">
            A CSV file with a column called <span className="font-semibold text-ink">text</span>. An optional{" "}
            <span className="font-semibold text-ink">language</span> column can say en or te for each row.
          </p>
          <Dropzone accept=".csv,text/csv" onFiles={(f) => setCsv(f[0])} className="mt-4">
            <div className="px-5 py-8 text-center">
              <p className="font-medium">{csv ? csv.name : "Drop a CSV file here"}</p>
              <p className="t-small mt-1 text-graphite">{csv ? "Click or drop to replace" : "or click to choose one"}</p>
            </div>
          </Dropzone>
          <div className="mt-5 space-y-4">
            <div>
              <p className="mb-2 text-sm font-medium">Language</p>
              <LanguagePicker value={language} onChange={setLanguage} />
            </div>
            <Switch checked={verify} onChange={setVerify} label="Also check claims for each row" hint="Much slower, a few seconds per row." />
          </div>
          <button className="btn mt-6 self-start" disabled={!csv || !!loading} onClick={() => run("text")}>
            Check all texts
          </button>
        </div>

        <div className="panel flex flex-col p-5">
          <h2 className="t-h2">Images</h2>
          <p className="t-small mt-1 text-graphite">Several image files, or one ZIP file with images inside.</p>
          <Dropzone accept="image/*,.zip" multiple onFiles={setImages} className="mt-4">
            <div className="px-5 py-8 text-center">
              <p className="font-medium">
                {images.length ? `${images.length} ${images.length === 1 ? "file" : "files"} selected` : "Drop images or a ZIP file here"}
              </p>
              <p className="t-small mt-1 text-graphite">{images.length ? "Click or drop to replace" : "or click to choose"}</p>
            </div>
          </Dropzone>
          <button className="btn mt-6 self-start" disabled={!images.length || !!loading} onClick={() => run("images")}>
            Check all images
          </button>
        </div>
      </div>

      {loading && (
        <div className="mt-8">
          <Working label={loading === "text" ? "Checking every text" : "Checking every image"} />
        </div>
      )}
      <div className="mt-6">
        <ErrorNote message={error} />
      </div>
      {result && <Results result={result} />}
    </div>
  );
}
