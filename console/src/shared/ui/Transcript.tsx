/** A heard transcript, word by word against what was said: misheard words struck through with what
 * was said beneath, words the facility list completed underlined in violet. */
import { align } from "../diff";

export function Transcript({ said, hyp, recovered = [], label }: { said: string; hyp: string; recovered?: string[]; label: string }) {
  const pieces = align(said, hyp);
  const rec = new Set(recovered.flatMap((p) => p.toLowerCase().split(/\s+/)));
  return (
    <p className="m-0 font-text text-lead" aria-label={label}>
      {pieces.map((p, n) => {
        const key = `${n}-${p.kind}`;
        if (p.kind === "same") {
          const word = (p.hyp ?? "").toLowerCase().replace(/[.,:;]/g, "");
          return (
            <span key={key} className={rec.has(word) ? "rounded-soft bg-recoveredWash px-1 text-recovered underline decoration-2 underline-offset-4" : ""}>
              {p.hyp}{" "}
            </span>
          );
        }
        if (p.kind === "ins")
          return (
            <span key={key} className="text-wrong">
              <span className="sr-only">inserted </span>
              <span className="line-through">{p.hyp}</span>{" "}
            </span>
          );
        if (p.kind === "del")
          return (
            <span key={key} className="rounded-soft bg-wrongWash px-1 text-wrong">
              <span className="sr-only">missing </span>[{p.ref}]{" "}
            </span>
          );
        return (
          <span key={key} className="inline-flex flex-col align-top leading-tight">
            <span className="text-wrong line-through">
              <span className="sr-only">heard </span>
              {p.hyp}
            </span>
            <span className="text-small font-ui text-muted">
              <span className="sr-only">said </span>
              {p.ref}
            </span>
          </span>
        );
      })}
    </p>
  );
}
