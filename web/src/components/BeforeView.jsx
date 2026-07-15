const pct = (value) => `${value * 100}%`;

export default function BeforeView({ imageURL, result, crop, overlays }) {
  return (
    <figure className="view">
      <figcaption>Original</figcaption>
      <div className="frame">
        <img src={imageURL} alt="Uploaded original" />
        {overlays.saliency && (
          <img className="saliency" src={result.saliency_png} alt="" aria-hidden="true" />
        )}
        {overlays.subjects &&
          result.subjects.map((subject, index) => (
            <div
              key={index}
              className="subject-box"
              style={{
                left: pct(subject.x1),
                top: pct(subject.y1),
                width: pct(subject.x2 - subject.x1),
                height: pct(subject.y2 - subject.y1),
              }}
            >
              <span>
                {subject.label} {(subject.confidence * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        {overlays.cropRect && crop && (
          <div
            className="crop-rect"
            style={{
              left: pct(crop.x),
              top: pct(crop.y),
              width: pct(crop.w),
              height: pct(crop.h),
            }}
          />
        )}
      </div>
    </figure>
  );
}
