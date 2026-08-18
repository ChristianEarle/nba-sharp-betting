import { parseDriverChips } from "../lib/format";

interface Props {
  s: string | null | undefined;
  n: number;
}

/** Honest driver chips -- renders the payload's pre-resolved chip text verbatim. */
export function DriverChips({ s, n }: Props) {
  const chips = parseDriverChips(s, n);
  if (chips.length === 0) return null;
  return (
    <>
      {chips.map((c, i) => (
        <span className="drv" key={`${c.label}-${i}`}>
          {c.dir && (
            <span className={`dir ${c.dir}`} aria-hidden="true">
              {c.dir === "up" ? "▲" : "▼"}
            </span>
          )}
          {c.label}
        </span>
      ))}
    </>
  );
}
