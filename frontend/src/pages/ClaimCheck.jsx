import { useMemo, useRef, useState } from "react";
import { errorMessage, verifyClaims } from "../api.js";
import { ClaimsList, ClaimsSummary } from "../components/ClaimsReport.jsx";
import MarkedText, { tokensFromText } from "../components/MarkedText.jsx";
import { EmptyReport, ErrorNote, LanguagePicker, PageHeader, ReportActions, Working } from "../components/ui.jsx";
import { SAMPLES } from "../lib/samples.js";
import { useKept } from "../lib/store.js";

export default function ClaimCheck() {
  const [text, setText] = useKept("claims.input", "");
  const [language, setLanguage] = useKept("claims.language", "auto");
  const [result, setResult] = useKept("claims.result", null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeClaim, setActiveClaim] = useState(null);
  const report = useRef(null);

  async function run(e) {
    e?.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const data = await verifyClaims({ text, language });
      setResult({ ...data, checkedText: text });
      if (window.innerWidth < 1024) report.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  const tokens = useMemo(() => (result?.checkedText ? tokensFromText(result.checkedText) : null), [result]);

  return (
    <div>
      <PageHeader title="Verify claims">
        The article is split into factual claims. Each one is compared with evidence from the knowledge base and
        fact-checks, then marked supported, contradicted or no evidence found.
      </PageHeader>

      <div className="print-full grid items-start gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <form className="no-print panel space-y-5 p-5 lg:sticky lg:top-8" onSubmit={run}>
          <div>
            <label htmlFor="claims-text" className="t-h3 mb-2 block">
              News text
            </label>
            <textarea
              id="claims-text"
              className="field field-sheet min-h-64"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && text.trim().length >= 10) run();
              }}
              placeholder="Paste the article whose claims you want to check…"
            />
            <p className="t-small mt-2 text-graphite">
              <button type="button" className="link" onClick={() => setText(SAMPLES.claims)}>
                Use an example with two true claims and two hoaxes
              </button>
            </p>
          </div>
          <div>
            <p className="mb-2 text-sm font-medium">Language</p>
            <LanguagePicker value={language} onChange={setLanguage} />
          </div>
          <button className="btn w-full" disabled={loading || text.trim().length < 10}>
            {loading ? "Verifying…" : "Verify claims"}
          </button>
        </form>

        <section ref={report} aria-label="Report" className="min-w-0 space-y-6 scroll-mt-28">
          {loading && <Working label="Finding claims and checking evidence" />}
          <ErrorNote message={error} />
          {!loading && !error && !result && (
            <EmptyReport
              title="Your evidence report will appear here"
              points={[
                "An overall result and a credibility score",
                "Every claim with its status, underlined in the article",
                "The evidence used for each claim, with sources",
              ]}
            />
          )}
          {result && (
            <>
              <div className="sheet p-6">
                <ClaimsSummary summary={result.summary} />
                <div className="mt-5 flex justify-end">
                  <ReportActions
                    onReset={() => {
                      setResult(null);
                      setText("");
                    }}
                  />
                </div>
                <div className="mt-2">
                  <ClaimsList claims={result.claims} onActiveChange={setActiveClaim} />
                </div>
              </div>
              {tokens && (
                <div className="sheet p-6">
                  <h2 className="t-h2">The article</h2>
                  <p className="t-small mt-1 mb-5 text-graphite">Hover a claim above to find it in the text.</p>
                  <MarkedText tokens={tokens} claims={result.claims} activeClaim={activeClaim} />
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </div>
  );
}
