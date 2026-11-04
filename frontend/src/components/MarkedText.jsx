import { useMemo } from "react";
import { claimVerdict } from "../lib/verdicts.js";

// Split like the backend does (whitespace kept), so token offsets line up with the original text.
export function tokensFromText(text) {
  return text.split(/(\s+)/).filter(Boolean).map((t) => ({ text: t, weight: 0 }));
}

// For each token, which claim (index) it belongs to, or -1.
function claimIndexPerToken(tokens, claims) {
  const full = tokens.map((t) => t.text).join("");
  const ranges = [];
  let searchFrom = 0;
  claims.forEach((c, i) => {
    let start = full.indexOf(c.claim, searchFrom);
    if (start < 0) start = full.indexOf(c.claim);
    if (start >= 0) {
      ranges.push([start, start + c.claim.length, i]);
      searchFrom = start + c.claim.length;
    }
  });
  let offset = 0;
  return tokens.map((t) => {
    const s = offset;
    const e = offset + t.text.length;
    offset = e;
    if (!t.text.trim()) return -2; // whitespace: decided by neighbours below
    const hit = ranges.find(([rs, re]) => s < re && e > rs);
    return hit ? hit[2] : -1;
  });
}

// The article as a marked-up proof.
//  - LIME word weights as highlighter strokes (red = towards fake, teal = towards real)
//  - extracted claims underlined by status; the hovered claim in the ledger lights up here
export default function MarkedText({ tokens, claims = [], activeClaim = null }) {
  const max = Math.max(...tokens.map((t) => Math.abs(t.weight)), 1e-6);
  const owner = useMemo(() => {
    const raw = claimIndexPerToken(tokens, claims);
    // whitespace inside one claim belongs to it, so the underline is continuous
    return raw.map((v, i) => (v !== -2 ? v : raw[i - 1] >= 0 && raw[i - 1] === raw[i + 1] ? raw[i - 1] : -1));
  }, [tokens, claims]);
  const hasWeights = tokens.some((t) => t.weight);
  let order = 0;

  return (
    <div>
      <p className="proof" lang="und">
        {tokens.map((t, i) => {
          const claimIdx = owner[i];
          const claim = claimIdx >= 0 ? claims[claimIdx] : null;
          const claimCls = claim
            ? `claim claim-${claimVerdict(claim.label).tone}${activeClaim === claimIdx ? " claim-active" : ""}`
            : "";
          if (!t.weight) {
            return claim ? (
              <span key={i} className={claimCls}>
                {t.text}
              </span>
            ) : (
              <span key={i}>{t.text}</span>
            );
          }
          const strength = Math.round(25 + 60 * (Math.abs(t.weight) / max));
          const delay = Math.min(order++ * 40, 900);
          const towards = t.weight > 0 ? "fake" : "real";
          return (
            <span
              key={i}
              className={`mark ${towards === "real" ? "real" : ""} ${claimCls}`}
              style={{ "--strength": `${strength}%`, "--delay": `${delay}ms` }}
              title={`Pushes towards ${towards} (weight ${t.weight > 0 ? "+" : ""}${t.weight.toFixed(3)})`}
            >
              {t.text}
            </span>
          );
        })}
      </p>

      <div className="t-small mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 text-graphite">
        {hasWeights && (
          <>
            <span className="flex items-center gap-2">
              <span className="mark inline-block h-3 w-6" style={{ "--strength": "70%" }} aria-hidden="true" />
              Makes it look fake
            </span>
            <span className="flex items-center gap-2">
              <span className="mark real inline-block h-3 w-6" style={{ "--strength": "70%" }} aria-hidden="true" />
              Makes it look real
            </span>
          </>
        )}
        {claims.length > 0 && (
          <>
            <span className="claim claim-real">Supported claim</span>
            <span className="claim claim-fake">Contradicted claim</span>
            <span className="claim claim-neutral">No evidence</span>
          </>
        )}
      </div>
      {hasWeights && (
        <p className="t-small mt-2 text-graphite">Stronger colour means more influence. Hover a word for its exact weight.</p>
      )}
    </div>
  );
}

export function TopWords({ words }) {
  if (!words?.length) return null;
  const top = words.slice(0, 8);
  const max = Math.max(...top.map((w) => Math.abs(w.weight)), 1e-6);
  return (
    <ol className="space-y-2">
      {top.map((w) => {
        const fake = w.weight > 0;
        return (
          <li key={w.word} className="grid grid-cols-[minmax(0,9rem)_1fr] items-center gap-3">
            <span className="truncate font-serif text-[1.02rem]" title={w.word}>
              {w.word}
            </span>
            <span className="flex items-center gap-2">
              <span
                className={`h-2 rounded-full ${fake ? "bg-fake" : "bg-real"}`}
                style={{ width: `${Math.max(6, (Math.abs(w.weight) / max) * 100)}%` }}
              />
              <span className={`t-small font-medium ${fake ? "text-fake" : "text-real"}`}>{fake ? "fake" : "real"}</span>
            </span>
          </li>
        );
      })}
    </ol>
  );
}
