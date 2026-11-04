import { TONE_TEXT, pct } from "../lib/verdicts.js";

const STRIP = {
  fake: "var(--color-fake)",
  real: "var(--color-real)",
  mixed: "var(--color-mixed)",
  neutral: "var(--color-mist)",
};

// Big verdict beside a status strip, over a Real <-> Fake instrument scale with ticks and a 50% threshold.
export default function VerdictScale({ verdict, fakeProbability, confidence, leftLabel = "Real", rightLabel = "Fake", note }) {
  const position = Math.min(98, Math.max(2, fakeProbability * 100));
  return (
    <div>
      <div className="verdict-strip" style={{ "--strip": STRIP[verdict.tone] }}>
        <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-1">
          <p className={`t-verdict ${TONE_TEXT[verdict.tone]}`}>{verdict.title}</p>
          <p className="pb-1 text-graphite">
            <span className="text-[1.35rem] font-semibold text-ink">{pct(confidence)}</span> confident
          </p>
        </div>
      </div>
      <div className="mt-6">
        <div
          className="scale-track"
          role="meter"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(fakeProbability * 100)}
          aria-label={`${rightLabel} probability`}
        >
          <span className="scale-marker" style={{ left: `${position}%` }} />
        </div>
        <div className="scale-ticks" aria-hidden="true">
          {[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map((t) => (
            <span key={t} className={t === 50 ? "mid" : ""} style={{ left: `${t}%`, opacity: t % 50 === 0 ? 1 : 0.55 }} />
          ))}
        </div>
        <div className="t-small mt-1 grid grid-cols-3 text-graphite">
          <span>{leftLabel}</span>
          <span className="text-center">
            {rightLabel} probability <span className="font-semibold text-ink">{pct(fakeProbability, 1)}</span>
          </span>
          <span className="text-right">{rightLabel}</span>
        </div>
      </div>
      {note && <p className="t-small mt-4 max-w-[62ch] text-graphite">{note}</p>}
    </div>
  );
}
