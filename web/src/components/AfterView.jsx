import { useEffect, useRef, useState } from "react";

export default function AfterView({ imageURL, scored, aspect }) {
  const canvasRef = useRef(null);
  const [ready, setReady] = useState(false);
  const crop = scored;

  useEffect(() => {
    setReady(false);
    const image = new Image();
    image.onload = () => {
      const sx = crop.x * image.naturalWidth;
      const sy = crop.y * image.naturalHeight;
      const sw = crop.w * image.naturalWidth;
      const sh = crop.h * image.naturalHeight;
      const canvas = canvasRef.current;
      if (!canvas) return;
      // Full source resolution so the download is lossless.
      canvas.width = Math.round(sw);
      canvas.height = Math.round(sh);
      canvas.getContext("2d").drawImage(image, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
      setReady(true);
    };
    image.src = imageURL;
  }, [imageURL, crop.x, crop.y, crop.w, crop.h]);

  const download = () => {
    canvasRef.current?.toBlob((blob) => {
      if (!blob) return;
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `recompose-${aspect.replace(":", "x")}.png`;
      link.click();
      URL.revokeObjectURL(link.href);
    }, "image/png");
  };

  return (
    <figure className="view">
      <figcaption>
        Recomposed
        <button className="download" onClick={download} disabled={!ready}>
          Download
        </button>
      </figcaption>
      <div className="frame">
        <canvas ref={canvasRef} />
        <div className="thirds" aria-hidden="true">
          <i style={{ left: "33.333%" }} />
          <i style={{ left: "66.667%" }} />
          <i className="h" style={{ top: "33.333%" }} />
          <i className="h" style={{ top: "66.667%" }} />
        </div>
      </div>
    </figure>
  );
}
