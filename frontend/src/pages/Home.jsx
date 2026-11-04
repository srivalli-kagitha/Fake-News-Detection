import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { getHistory } from "../api.js";
import Dropzone from "../components/Dropzone.jsx";
import { Chip } from "../components/ui.jsx";
import { SAMPLES } from "../lib/samples.js";
import { anyVerdict, pct } from "../lib/verdicts.js";

// Small working samples of each report part, so the page shows what you get instead of describing it.
function SpecimenMarks() {
  return (
    <p className="proof text-[1rem] leading-[1.9]">
      <span className="mark" style={{ "--strength": "75%" }}>BREAKING:</span> scientists{" "}
      <span className="mark" style={{ "--strength": "45%" }}>confirm</span> towers spread the virus.{" "}
      <span className="mark" style={{ "--strength": "65%" }}>Share</span> before it gets{" "}
      <span className="mark real" style={{ "--strength": "40%" }}>deleted</span>.
    </p>
  );
}

function SpecimenScale() {
  return (
    <div className="verdict-strip" style={{ "--strip": "var(--color-fake)" }}>
      <p className="font-semibold text-fake text-[1.35rem] leading-none tracking-tight">Likely fake</p>
      <div className="scale-track mt-4">
        <span className="scale-marker" style={{ left: "86%", height: "20px" }} />
      </div>
      <p className="t-small mt-2 flex justify-between text-graphite">
        <span>Real</span>
        <span>Fake</span>
      </p>
    </div>
  );
}

function SpecimenClaims() {
  return (
    <ul className="space-y-2.5">
      {[
        ["bg-real", "Chandrayaan-3 landed on 23 August 2023.", "real", "Supported"],
        ["bg-fake", "2000 rupee notes carry a GPS chip.", "fake", "Contradicted"],
      ].map(([bar, text, tone, label]) => (
        <li key={text} className="relative pl-3.5">
          <span className={`absolute top-0.5 bottom-0.5 left-0 w-[3px] rounded-full ${bar}`} />
          <p className="font-serif leading-snug">{text}</p>
          <span className={`t-small font-semibold ${tone === "real" ? "text-real" : "text-fake"}`}>{label}</span>
        </li>
      ))}
    </ul>
  );
}

const KIND = { text: "News text", image: "Image", claims: "Claims" };

function RecentChecks() {
  const [items, setItems] = useState([]);
  useEffect(() => {
    getHistory(12)
      .then((all) => setItems(all.filter((it) => !it.kind.startsWith("batch")).slice(0, 4)))
      .catch(() => setItems([]));
  }, []);
  if (!items.length) return null;
  return (
    <section className="mt-12 max-w-3xl">
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="t-h3">Your recent checks</h2>
        <Link to="/history" className="link t-small">
          See all history
        </Link>
      </div>
      <ul className="panel mt-3 divide-y divide-rule-soft">
        {items.map((it) => {
          const v = anyVerdict(it.verdict);
          return (
            <li key={it.id} className="flex items-center gap-4 px-5 py-3">
              <span className="t-small w-20 flex-none text-graphite">{KIND[it.kind]}</span>
              <span className={`min-w-0 flex-1 truncate ${it.kind === "image" ? "" : "font-serif"}`} title={it.input_preview}>
                {it.input_preview}
              </span>
              <Chip tone={v.tone}>{v.title}</Chip>
              <span className="t-small hidden w-10 text-right text-graphite sm:block">{pct(it.confidence)}</span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

export default function Home() {
  const [text, setText] = useState("");
  const navigate = useNavigate();
  const canCheck = text.trim().length >= 10;

  const checkText = (e) => {
    e?.preventDefault();
    if (canCheck) navigate("/text", { state: { text, run: true } });
  };
  const checkImage = (files) => {
    const file = files.find((f) => f.type.startsWith("image/"));
    if (file) navigate("/image", { state: { file, run: true } });
  };

  return (
    <div>
      <section className="max-w-3xl">
        <h1 className="t-display">Is this news real?</h1>
        <p className="mt-4 max-w-[60ch] text-[1.0625rem] text-graphite">
          Paste an article or a headline in English or Telugu, or drop in an image. You get a verdict, the words or image
          regions that led to it, and each claim checked against evidence.
        </p>

        <form onSubmit={checkText} className="mt-8">
          <label htmlFor="home-text" className="sr-only">
            News text
          </label>
          <textarea
            id="home-text"
            className="field field-sheet min-h-56 shadow-[var(--sheet-shadow)]"
            placeholder="Paste a news article or headline…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) checkText();
            }}
          />
          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-3">
            <button className="btn" disabled={!canCheck}>
              Check this text
            </button>
            <span className="t-small text-graphite">or Ctrl + Enter</span>
          </div>
          <p className="t-small mt-4 text-graphite">
            Not sure what to paste? Try{" "}
            <button type="button" className="link" onClick={() => setText(SAMPLES.enFake)}>
              a viral 5G rumour
            </button>
            ,{" "}
            <button type="button" className="link" onClick={() => setText(SAMPLES.enReal)}>
              a news report
            </button>{" "}
            or{" "}
            <button type="button" className="link" onClick={() => setText(SAMPLES.te)}>
              a Telugu article
            </button>
            .
          </p>
        </form>

        <Dropzone accept="image/*" onFiles={checkImage} className="mt-8">
          <div className="flex items-center gap-4 px-5 py-5">
            <svg viewBox="0 0 24 24" className="h-7 w-7 flex-none text-graphite" fill="none" aria-hidden="true">
              <path
                d="M4 5h16v14H4zM4 15l4.5-4.5 4 4L15 12l5 5M15.5 8.5h.01"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <div>
              <p className="font-medium">Checking an image instead?</p>
              <p className="t-small text-graphite">Drop it here or click to choose a JPG, PNG or WEBP file.</p>
            </div>
          </div>
        </Dropzone>
      </section>

      <RecentChecks />

      <section className="mt-16 grid max-w-5xl gap-6 md:grid-cols-3">
        <div className="sheet flex flex-col p-5">
          <SpecimenScale />
          <h2 className="t-h3 mt-6">A clear verdict</h2>
          <p className="mt-1 text-graphite">
            Fake or real for text, AI-generated or real for images, with how confident the model is.
          </p>
        </div>
        <div className="sheet flex flex-col p-5">
          <SpecimenMarks />
          <h2 className="t-h3 mt-auto pt-6">The reason behind it</h2>
          <p className="mt-1 text-graphite">
            Words that pushed the decision are highlighted. For images, a heatmap shows where the model looked.
          </p>
        </div>
        <div className="sheet flex flex-col p-5">
          <SpecimenClaims />
          <h2 className="t-h3 mt-auto pt-6">Claims tested against evidence</h2>
          <p className="mt-1 text-graphite">
            Each claim is marked supported, contradicted or no evidence found, with its sources.
          </p>
        </div>
      </section>

      <p className="t-small mt-10 text-graphite">
        Have a spreadsheet of headlines or a folder of images?{" "}
        <Link to="/batch" className="link">
          Check many at once
        </Link>
        .
      </p>
    </div>
  );
}
