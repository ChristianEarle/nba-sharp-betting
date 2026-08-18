export type ViewId = "board" | "positions" | "trust" | "method";

const VIEWS: { id: ViewId; label: string }[] = [
  { id: "board", label: "Board" },
  { id: "positions", label: "Positions" },
  { id: "trust", label: "Trust" },
  { id: "method", label: "Method" },
];

interface Props {
  current: ViewId;
  onChange: (id: ViewId) => void;
}

export function ViewNav({ current, onChange }: Props) {
  return (
    <nav className="viewnav" role="tablist" aria-label="Dashboard views">
      {VIEWS.map((v) => (
        <button
          key={v.id}
          className="vtab"
          role="tab"
          aria-selected={v.id === current}
          onClick={() => onChange(v.id)}
        >
          {v.label}
        </button>
      ))}
    </nav>
  );
}
