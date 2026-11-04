import { useRef, useState } from "react";

export default function Dropzone({ accept, multiple = false, onFiles, children, className = "" }) {
  const [over, setOver] = useState(false);
  const input = useRef(null);
  return (
    <div
      className={`relative rounded-[12px] border-2 border-dashed transition-colors ${
        over ? "border-focus bg-[var(--drop-over)]" : "border-rule bg-surface hover:border-mist"
      } ${className}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (e.dataTransfer.files?.length) onFiles([...e.dataTransfer.files]);
      }}
    >
      <button
        type="button"
        className="absolute inset-0 h-full w-full rounded-[10px]"
        onClick={() => input.current?.click()}
        aria-label="Choose file"
      />
      <div className="pointer-events-none relative">{children}</div>
      <input
        ref={input}
        type="file"
        accept={accept}
        multiple={multiple}
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.length) onFiles([...e.target.files]);
          e.target.value = "";
        }}
      />
    </div>
  );
}
