import { useState } from "react";

// Drag (or use arrow keys) to wipe between the input image and its Grad-CAM heatmap.
export default function CompareSlider({ before, after, label }) {
  const [split, setSplit] = useState(50);
  return (
    <figure>
      <div className="compare">
        <img src={before} alt="Image that was checked" />
        <img src={after} alt="Grad-CAM heatmap" className="overlay" style={{ clipPath: `inset(0 0 0 ${split}%)` }} />
        <input
          type="range"
          min="0"
          max="100"
          value={split}
          onChange={(e) => setSplit(Number(e.target.value))}
          aria-label="Compare image and heatmap"
        />
        <span className="handle" style={{ left: `${split}%` }} />
      </div>
      <figcaption className="t-small mt-2 flex justify-between text-graphite">
        <span>Your image</span>
        <span>{label}</span>
      </figcaption>
    </figure>
  );
}
