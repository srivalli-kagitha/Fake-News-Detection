import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { clearHistory, deleteHistory, errorMessage, getHistory, getHistoryItem } from "../api.js";
import { setKept } from "../lib/store.js";
import { Chip, ErrorNote, PageHeader } from "../components/ui.jsx";
import { anyVerdict, pct } from "../lib/verdicts.js";

const KINDS = {
  text: "News text",
  image: "Image",
  claims: "Claims",
  batch_text: "Batch of texts",
  batch_image: "Batch of images",
};

const FILTERS = [
  ["all", "All"],
  ["text", "Text"],
  ["image", "Images"],
  ["claims", "Claims"],
  ["batch", "Batches"],
];

function when(iso) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

// Where each kind of check is shown, and how its saved report is put back on that page.
const OPEN = {
  text: (rec, r) => {
    setKept("text.input", rec.details.input);
    setKept("text.language", rec.language || "auto");
    setKept("text.result", { ...r, id: rec.id, checkedText: rec.details.input });
    return "/text";
  },
  claims: (rec, r) => {
    setKept("claims.input", rec.details.input);
    setKept("claims.language", rec.language || "auto");
    setKept("claims.result", { ...r, id: rec.id, checkedText: rec.details.input });
    return "/claims";
  },
  image: (rec, r) => {
    setKept("image.file", null, false);
    setKept("image.result", { ...r, id: rec.id });
    return "/image";
  },
  batch_text: (rec, r) => {
    setKept("batch.result", { ...r, kind: "text" });
    return "/batch";
  },
  batch_image: (rec, r) => {
    setKept("batch.result", { ...r, kind: "image" });
    return "/batch";
  },
};

export default function History() {
  const navigate = useNavigate();
  const [opening, setOpening] = useState(null);
  const [items, setItems] = useState(null);
  const [filter, setFilter] = useState("all");
  const [error, setError] = useState("");
  const [confirming, setConfirming] = useState(false);

  const load = () =>
    getHistory(200)
      .then(setItems)
      .catch((e) => setError(errorMessage(e)));
  useEffect(() => {
    load();
  }, []);

  async function open(it) {
    setOpening(it.id);
    setError("");
    try {
      const rec = await getHistoryItem(it.id);
      navigate(OPEN[rec.kind](rec, rec.details.result));
    } catch (e) {
      setError(errorMessage(e));
      setOpening(null);
    }
  }

  const shown = (items || []).filter((it) =>
    filter === "all" ? true : filter === "batch" ? it.kind.startsWith("batch") : it.kind === filter
  );

  return (
    <div>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <PageHeader title="History">Every check you've run on this computer. Click one to open its full report.</PageHeader>
        {items?.length > 0 &&
          (confirming ? (
            <div className="flex items-center gap-2">
              <span className="t-small text-graphite">Delete all {items.length} checks?</span>
              <button
                className="btn-quiet border-fake/40 text-fake"
                onClick={async () => {
                  await clearHistory();
                  setConfirming(false);
                  load();
                }}
              >
                Delete all
              </button>
              <button className="btn-quiet" onClick={() => setConfirming(false)}>
                Keep
              </button>
            </div>
          ) : (
            <button className="btn-quiet" onClick={() => setConfirming(true)}>
              Clear history
            </button>
          ))}
      </div>

      <ErrorNote message={error} />

      {items?.length === 0 && (
        <div className="inset max-w-xl px-6 py-8">
          <p className="t-h3">Nothing checked yet</p>
          <p className="mt-1 text-graphite">
            Results are saved here automatically.{" "}
            <Link to="/" className="link">
              Check your first article
            </Link>
            .
          </p>
        </div>
      )}

      {items?.length > 0 && (
        <>
          <div className="segmented mb-5" role="group" aria-label="Filter by type">
            {FILTERS.map(([v, label]) => (
              <button key={v} type="button" aria-pressed={filter === v} onClick={() => setFilter(v)}>
                {label}
              </button>
            ))}
          </div>
          <div className="panel overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-graphite">
                <tr>
                  <th scope="col" className="px-5 py-3 font-medium">When</th>
                  <th scope="col" className="px-5 py-3 font-medium">Type</th>
                  <th scope="col" className="px-5 py-3 font-medium">What was checked</th>
                  <th scope="col" className="px-5 py-3 font-medium">Result</th>
                  <th scope="col" className="px-5 py-3 text-right font-medium">Confidence</th>
                  <th scope="col" className="px-5 py-3"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {shown.map((it) => {
                  const batch = it.kind.startsWith("batch");
                  const v = anyVerdict(it.verdict);
                  const canOpen = it.details?.has_report && OPEN[it.kind];
                  return (
                    <tr
                      key={it.id}
                      className={`border-t border-rule-soft align-top ${canOpen ? "cursor-pointer hover:bg-rule-soft/40" : ""}`}
                      onClick={() => canOpen && open(it)}
                      title={canOpen ? "Open the full report" : "Saved before full reports were kept: only the summary is available"}
                    >
                      <td className="px-5 py-3.5 whitespace-nowrap text-graphite">{when(it.created_at)}</td>
                      <td className="px-5 py-3.5 whitespace-nowrap">{KINDS[it.kind] || it.kind}</td>
                      <td className="max-w-sm px-5 py-3.5">
                        <span className={`line-clamp-2 ${it.kind === "text" || it.kind === "claims" ? "font-serif" : ""}`} title={it.input_preview}>
                          {it.input_preview}
                        </span>
                      </td>
                      <td className="px-5 py-3.5">{batch ? <span className="text-graphite">{it.verdict}</span> : <Chip tone={v.tone}>{v.title}</Chip>}</td>
                      <td className="px-5 py-3.5 text-right">{pct(it.confidence)}</td>
                      <td className="px-5 py-3.5 text-right whitespace-nowrap">
                        {canOpen && (
                          <button
                            className="t-small mr-4 font-medium text-ink hover:underline"
                            onClick={(e) => {
                              e.stopPropagation();
                              open(it);
                            }}
                          >
                            {opening === it.id ? "Opening…" : "Open"}
                          </button>
                        )}
                        <button
                          className="t-small text-graphite hover:text-fake"
                          onClick={async (e) => {
                            e.stopPropagation();
                            await deleteHistory(it.id);
                            load();
                          }}
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {shown.length === 0 && <p className="px-5 py-6 text-graphite">No checks of this type yet.</p>}
          </div>
        </>
      )}
    </div>
  );
}
