import { useCallback, useRef, useState } from "react";

export default function Dropzone({ onFile, busy, hasImage }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = useCallback(
    (files) => {
      const file = files?.[0];
      if (file && file.type.startsWith("image/")) onFile(file);
    },
    [onFile]
  );

  return (
    <div
      className={`dropzone ${dragOver ? "dragover" : ""} ${hasImage ? "compact" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragOver(false);
        handleFiles(event.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => event.key === "Enter" && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(event) => handleFiles(event.target.files)}
      />
      {busy ? "Analyzing…" : hasImage ? "Drop another photo or click to replace" : "Drop a photo here, or click to choose"}
    </div>
  );
}
