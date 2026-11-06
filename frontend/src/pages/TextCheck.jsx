import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { analyzeText, errorMessage, explainText } from "../api.js";
import { ClaimsList, ClaimsSummary } from "../components/ClaimsReport.jsx";
import MarkedText, { TopWords, tokensFromText } from "../components/MarkedText.jsx";
import { EmptyReport, ErrorNote, LanguagePicker, PageHeader, ReportActions, Switch, Working } from "../components/ui.jsx";
import VerdictScale from "../components/VerdictScale.jsx";
import { SAMPLES } from "../lib/samples.js";
import { useKept } from "../lib/store.js";
import { pct, summaryVerdict, textVerdict } from "../lib/verdicts.js";

// Inline note while LIME runs; the rest of the report is already readable.
function ExplainProgress() {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="examine mb-5 flex items-center justify-between gap-4 rounded-[10px] border border-rule-soft bg-paper px-4 py-3" role="status">
      <span className="t-small">
        <span className="font-semibold">Highlighting the words behind the verdict</span>
        <span className="text-graphite">. The model re-reads the article 200 times with different words hidden.</span>
      </span>
      <span className="t-small flex-none text-graphite">{seconds}s</span>
    </div>
  );
}

export default function TextCheck() {
  const location = useLocation();
  const [text, setText] = useKept("text.input", "");
  const [language, setLanguage] = useKept("text.language", "auto");
  const [explain, setExplain] = useKept("text.explain", true);
  const [verify, setVerify] = useKept("text.verify", true);
  const [result, setResult] = useKept("text.result", null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeClaim, setActiveClaim] = useState(null);
  const [explaining, setExplaining] = useState(false);
  const [explainError, setExplainError] = useState("");
  const report = useRef(null);

  async function run(input = text) {
    setLoading(true);
    setError("");
    setExplainError("");
    setResult(null);
    let data;
    try {
      // Step 1: verdict and claim evidence (fast). The word highlights need ~200 model runs, so they come after.
      data = await analyzeText({ text: input, language, explain: false, verify_claims: verify });
      setResult({ ...data, checkedText: input });
      if (window.innerWidth < 1024) report.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      setError(errorMessage(err));
      return;
    } finally {
      setLoading(false);
    }
    // Step 2: LIME highlights, filled into the report when ready.
    if (explain && data.classification) {
      setExplaining(true);
      try {
        const explanation = await explainText({ text: input, language: data.language });
        setResult((r) =>
          r && r.checkedText === input ? { ...r, classification: { ...r.classification, explanation } } : r
        );
      } catch (err) {
        setExplainError(errorMessage(err));
      } finally {
        setExplaining(false);
      }
    }
  }

  // Arriving from the Start page with text: check it straight away.
  useEffect(() => {
    if (location.state?.run && location.state?.text) {
      setText(location.state.text);
      run(location.state.text);
      window.history.replaceState({}, "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const c = result?.classification;
  const v = result?.verification;
  const overall = result?.overall;
  const tokens = useMemo(
    () =>
      c?.explanation?.tokens || (result?.checkedText && (v || explaining) ? tokensFromText(result.checkedText) : null),
    [c, v, result, explaining]
  );

  const reset = () => {
    setResult(null);
    setText("");
    setError("");
  };

  return (
    <div>
      <PageHeader title="Check news text">
        English or Telugu. The text model gives a verdict and highlights the words behind it. Claim checking tests each
        factual statement against evidence.
      </PageHeader>

      <div className="print-full grid items-start gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <form
          className="no-print panel space-y-5 p-5 lg:sticky lg:top-8"
          onSubmit={(e) => {
            e.preventDefault();
            run();
          }}
        >
          <div>
            <label htmlFor="news-text" className="t-h3 mb-2 block">
              News text
            </label>
            <textarea
              id="news-text"
              className="field field-sheet min-h-64"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && text.trim().length >= 10) run();
              }}
              placeholder="Paste a headline or the full article…"
            />
            <p className="t-small mt-2 text-graphite">
              Examples:{" "}
              <button type="button" className="link" onClick={() => setText(SAMPLES.enFake)}>
                rumour
              </button>
              ,{" "}
              <button type="button" className="link" onClick={() => setText(SAMPLES.enReal)}>
                news report
              </button>
              ,{" "}
              <button type="button" className="link" onClick={() => setText(SAMPLES.te)}>
                Telugu
              </button>
            </p>
          </div>
          <div>
            <p className="mb-2 text-sm font-medium">Language</p>
            <LanguagePicker value={language} onChange={setLanguage} />
          </div>
          <div className="space-y-3">
            <Switch checked={explain} onChange={setExplain} label="Highlight the words behind the verdict" hint="Shown a little after the verdict." />
            <Switch checked={verify} onChange={setVerify} label="Check claims against evidence" />
          </div>
          <button className="btn w-full" disabled={loading || text.trim().length < 10}>
            {loading ? "Checking…" : "Check this text"}
          </button>
        </form>

        <section ref={report} aria-label="Report" className="min-w-0 space-y-6 scroll-mt-28">
          {loading && <Working label="Checking the text" />}
          <ErrorNote message={error} />

          {!loading && !error && !result && (
            <EmptyReport
              title="Your report will appear here"
              points={[
                "A fake or real verdict with the model's confidence",
                "The article with influential words highlighted",
                "Each claim marked supported, contradicted or no evidence found, and underlined in the article",
              ]}
            />
          )}

          {result && (
            <>
              <div className="sheet p-6">
                {overall?.verdict === "Unverified" ? (
                  <div>
                    <p className="t-verdict text-graphite">Can't verify</p>
                    <p className="mt-3 max-w-prose">
                      We found no evidence that confirms or contradicts{" "}
                      {v.claims.length === 1 ? "the claim" : `any of the ${v.claims.length} claims`} in this text, so
                      TruthLens cannot say whether it is true or false.
                    </p>
                    {overall.style_fake_probability !== null && overall.style_fake_probability !== undefined && (
                      <p className="t-small mt-3 text-graphite">
                        Writing-style signal only: the text model rates the wording{" "}
                        {pct(overall.style_fake_probability)} like fake news. This is about how it is written, not
                        whether it is true.
                      </p>
                    )}
                  </div>
                ) : overall ? (
                  <VerdictScale
                    verdict={textVerdict(overall.verdict)}
                    fakeProbability={overall.fake_risk}
                    confidence={Math.max(overall.fake_risk, 1 - overall.fake_risk)}
                    note={
                      overall.based_on.length > 1
                        ? `Combines the text model (${textVerdict(c.label).title.toLowerCase()}, ${pct(c.confidence)}) with the claim evidence (${summaryVerdict(v.summary.verdict).title.toLowerCase()}).${v.method === "llm" ? ` Claims checked by an LLM (${v.model}) against live evidence.` : ""}`
                        : overall.based_on[0] === "classifier"
                          ? "Based on the text model only."
                          : "Based on the claim evidence only."
                    }
                  />
                ) : c ? (
                  <VerdictScale verdict={textVerdict(c.label)} fakeProbability={c.fake_probability} confidence={c.confidence} />
                ) : (
                  <div>
                    <p className="t-verdict text-graphite">Not enough evidence</p>
                    <p className="mt-2 text-graphite">
                      None of the claims could be confirmed or contradicted, and the text model is not available.
                    </p>
                  </div>
                )}
                <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
                  <p className="t-small text-graphite">Language: {result.language === "te" ? "Telugu" : "English"}</p>
                  <ReportActions onReset={reset} />
                </div>
              </div>

              {result.classifier_error && (
                <ErrorNote title="Text model not available" message={`${result.classifier_error} This report uses the claim evidence only.`} />
              )}

              {tokens && (
                <div className="sheet p-6">
                  <h2 className="t-h2">{c?.explanation || explaining ? "Why the model decided this" : "The article"}</h2>
                  <p className="t-small mt-1 mb-5 max-w-[62ch] text-graphite">
                    {c?.explanation || explaining
                      ? "Explained with LIME, which hides random groups of words and measures how the verdict changes."
                      : "Claims found in the text are underlined by what the evidence says."}
                    {v?.claims.length > 0 && " Hover a claim below to find it here."}
                  </p>
                  {explaining && <ExplainProgress />}
                  {explainError && <p className="t-small mb-4 text-fake">Couldn't highlight the words: {explainError}</p>}
                  <MarkedText tokens={tokens} claims={v?.claims || []} activeClaim={activeClaim} />
                  {c?.explanation?.top_words?.length > 0 && (
                    <div className="mt-8 border-t border-rule-soft pt-5">
                      <h3 className="t-h3 mb-3">Most influential words</h3>
                      <TopWords words={c.explanation.top_words} />
                    </div>
                  )}
                </div>
              )}

              {v && (
                <div className="sheet p-6">
                  <h2 className="t-h2 mb-5">Claims checked against evidence</h2>
                  <ClaimsSummary summary={v.summary} />
                  <div className="mt-4">
                    <ClaimsList claims={v.claims} onActiveChange={setActiveClaim} />
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </div>
  );
}
