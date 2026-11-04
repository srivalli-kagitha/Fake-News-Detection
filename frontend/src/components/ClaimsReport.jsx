import { Chip } from "./ui.jsx";
import { TONE_BAR, TONE_TEXT, claimVerdict, pct, summaryVerdict } from "../lib/verdicts.js";

const ORDER = ["Support", "Contradiction", "Unknown"];

// Overall result + one stacked bar showing how the claims split.
export function ClaimsSummary({ summary }) {
  const verdict = summaryVerdict(summary.verdict);
  const total = summary.total_claims || 1;
  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-1">
        <p className={`t-verdict ${TONE_TEXT[verdict.tone]}`}>{verdict.title}</p>
        <p className="pb-1 text-graphite">
          Credibility <span className="font-semibold text-ink">{pct(summary.credibility_score)}</span>
        </p>
      </div>
      <div className="mt-5 flex h-3 gap-[3px] overflow-hidden rounded-full" aria-hidden="true">
        {ORDER.map((k) =>
          summary.counts[k] ? (
            <span
              key={k}
              className={TONE_BAR[claimVerdict(k).tone]}
              style={{ width: `${(summary.counts[k] / total) * 100}%` }}
            />
          ) : null
        )}
      </div>
      <p className="t-small mt-2 flex flex-wrap gap-x-5 gap-y-1 text-graphite">
        <span>
          {summary.total_claims} {summary.total_claims === 1 ? "claim" : "claims"} checked
        </span>
        {ORDER.map((k) => (
          <span key={k} className="flex items-center gap-1.5">
            <span className={`h-2 w-2 rounded-full ${TONE_BAR[claimVerdict(k).tone]}`} />
            {summary.counts[k]} {claimVerdict(k).title.toLowerCase()}
          </span>
        ))}
      </p>
    </div>
  );
}

function NliBar({ nli }) {
  const parts = [
    ["entailment", "bg-real", "agrees"],
    ["neutral", "bg-mist", "unrelated"],
    ["contradiction", "bg-fake", "disagrees"],
  ];
  return (
    <div className="mt-2">
      <div className="flex h-1.5 overflow-hidden rounded-full bg-rule-soft">
        {parts.map(([k, cls]) => (
          <span key={k} className={cls} style={{ width: `${nli[k] * 100}%` }} />
        ))}
      </div>
      <p className="t-small mt-1 flex gap-4 text-graphite">
        {parts.map(([k, , label]) => (
          <span key={k}>
            {label} {pct(nli[k])}
          </span>
        ))}
      </p>
    </div>
  );
}

function Evidence({ e }) {
  return (
    <li className="border-t border-rule-soft py-3 first:border-t-0">
      <p className="font-serif text-[1.02rem] leading-relaxed">{e.text}</p>
      <p className="t-small mt-1 text-graphite">
        {e.url ? (
          <a href={e.url} target="_blank" rel="noreferrer" className="link">
            {e.source}
          </a>
        ) : (
          e.source
        )}
        {e.similarity !== null && e.similarity !== undefined && <>, {pct(e.similarity)} relevant</>}
      </p>
      {e.nli && <NliBar nli={e.nli} />}
    </li>
  );
}

// The evidence ledger: each claim with its status strip, label and expandable evidence.
export function ClaimsList({ claims, onActiveChange }) {
  const set = (i) => onActiveChange?.(i);
  return (
    <ol className="divide-y divide-rule-soft" onMouseLeave={() => set(null)}>
      {claims.map((c, i) => {
        const v = claimVerdict(c.label);
        return (
          <li
            key={i}
            className="relative py-5 pl-5"
            onMouseEnter={() => set(i)}
            onFocus={() => set(i)}
            onBlur={() => set(null)}
          >
            <span className={`absolute top-5 bottom-5 left-0 w-[3px] rounded-full ${TONE_BAR[v.tone]}`} aria-hidden="true" />
            <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
              <p className="max-w-[60ch] flex-1 font-serif text-[1.08rem] leading-relaxed">{c.claim}</p>
              <div className="flex items-center gap-2">
                <Chip tone={v.tone}>{v.title}</Chip>
                {c.label !== "Unknown" && <span className="t-small text-graphite">{pct(c.confidence)}</span>}
              </div>
            </div>
            {c.evidence.length > 0 ? (
              <details className="mt-2">
                <summary className="t-small inline-flex items-center gap-1 font-medium text-ink">
                  <svg className="chev h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                    <path d="M6 3l5 5-5 5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  Show {c.evidence.length} {c.evidence.length === 1 ? "piece" : "pieces"} of evidence
                </summary>
                <ul className="inset mt-3 px-4">
                  {c.evidence.map((e, j) => (
                    <Evidence key={j} e={e} />
                  ))}
                </ul>
              </details>
            ) : (
              <p className="t-small mt-1 text-graphite">{v.hint}</p>
            )}
          </li>
        );
      })}
    </ol>
  );
}
