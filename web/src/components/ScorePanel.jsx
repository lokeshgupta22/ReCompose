const RULE_LABELS = {
  thirds: "Rule of thirds",
  retention: "Content kept",
  balance: "Visual balance",
  headroom: "Headroom",
  subject_size: "Subject size",
};

export default function ScorePanel({ scored }) {
  return (
    <section className="scores">
      <div className="overall">
        <span className="value">{(scored.score * 100).toFixed(0)}</span>
        <span className="label">composition score</span>
      </div>
      <div className="bars">
        {Object.entries(scored.rules).map(([rule, value]) => (
          <div className="bar-row" key={rule}>
            <span className="bar-label">{RULE_LABELS[rule] ?? rule}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${value * 100}%` }} />
            </div>
            <span className="bar-value">{(value * 100).toFixed(0)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
