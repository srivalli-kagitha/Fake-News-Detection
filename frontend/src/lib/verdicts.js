// Plain-language names for everything the backend returns, plus a colour tone for each.

const TEXT = {
  Fake: { title: "Likely fake", tone: "fake" },
  Real: { title: "Likely real", tone: "real" },
  Unverified: { title: "Can't verify", tone: "neutral" },
};

const IMAGE = {
  "AI-Generated / Manipulated": { title: "Likely AI-generated", tone: "fake" },
  Real: { title: "Likely a real photo", tone: "real" },
};

const CLAIM = {
  Support: { title: "Supported", tone: "real", hint: "Evidence agrees with this claim." },
  Contradiction: { title: "Contradicted", tone: "fake", hint: "Evidence says the opposite." },
  Unknown: { title: "No evidence found", tone: "neutral", hint: "Nothing in the evidence base confirms or denies it." },
};

const SUMMARY = {
  "Likely Credible": { title: "Claims are supported", tone: "real" },
  "Likely Misleading": { title: "Claims are contradicted", tone: "fake" },
  "Mixed Evidence": { title: "Mixed evidence", tone: "mixed" },
  Unverified: { title: "Not enough evidence", tone: "neutral" },
};

export const textVerdict = (label) => TEXT[label] || { title: label, tone: "neutral" };
export const imageVerdict = (label) => IMAGE[label] || { title: label, tone: "neutral" };
export const claimVerdict = (label) => CLAIM[label] || { title: label, tone: "neutral" };
export const summaryVerdict = (label) => SUMMARY[label] || { title: label, tone: "neutral" };

export function anyVerdict(label) {
  return TEXT[label] || IMAGE[label] || CLAIM[label] || SUMMARY[label] || { title: label, tone: "neutral" };
}

export const TONE_TEXT = {
  fake: "text-fake",
  real: "text-real",
  mixed: "text-mixed",
  neutral: "text-graphite",
};

export const TONE_CHIP = {
  fake: "bg-fake-tint text-fake",
  real: "bg-real-tint text-real",
  mixed: "bg-mixed-tint text-mixed",
  neutral: "bg-rule-soft text-graphite",
};

export const TONE_BAR = {
  fake: "bg-fake",
  real: "bg-real",
  mixed: "bg-mixed",
  neutral: "bg-mist",
};

export const pct = (v, digits = 0) =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(digits)}%`;
