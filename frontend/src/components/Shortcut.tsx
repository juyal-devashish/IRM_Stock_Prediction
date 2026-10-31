import { pct } from "../format";
import { SHORTCUT } from "../methods";

/** How often the shortcut feature agrees with the label in each environment. */
export function Shortcut({ benchmarkId }: { benchmarkId: string }) {
  const rows = SHORTCUT[benchmarkId];
  if (!rows) return null;
  return (
    <ul className="shortcut" aria-label="Shortcut agreement with the label, by environment">
      {rows.map((r) => (
        <li key={r.env} className={r.held_out ? "held-out" : undefined}>
          <span className="env">{r.env}</span>
          <span className="bar" aria-hidden="true">
            <span className="fill" style={{ width: `${r.agree * 100}%` }} />
            <span className="half" />
          </span>
          <span className="agree">{pct(r.agree, 0)}</span>
        </li>
      ))}
    </ul>
  );
}
