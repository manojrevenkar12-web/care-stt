/** Word alignment between what was said and what was heard, for colouring transcripts. */
export type Piece = { kind: "same" | "sub" | "del" | "ins"; ref?: string; hyp?: string };

const norm = (w: string) => w.toLowerCase().replace(/[.,;:!?„"()]/g, "");

export function align(ref: string, hyp: string): Piece[] {
  const a = ref.split(/\s+/).filter(Boolean);
  const b = hyp.split(/\s+/).filter(Boolean);
  const d: number[][] = Array.from({ length: a.length + 1 }, (_, i) => Array.from({ length: b.length + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)));
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++) {
      const row = d[i] as number[];
      const up = d[i - 1] as number[];
      row[j] = Math.min((up[j] as number) + 1, (row[j - 1] as number) + 1, (up[j - 1] as number) + (norm(a[i - 1] as string) === norm(b[j - 1] as string) ? 0 : 1));
    }
  const out: Piece[] = [];
  let i = a.length;
  let j = b.length;
  while (i > 0 || j > 0) {
    const here = (d[i] as number[])[j] as number;
    if (i > 0 && j > 0 && here === ((d[i - 1] as number[])[j - 1] as number) + (norm(a[i - 1] as string) === norm(b[j - 1] as string) ? 0 : 1)) {
      out.push({ kind: norm(a[i - 1] as string) === norm(b[j - 1] as string) ? "same" : "sub", ref: a[i - 1], hyp: b[j - 1] });
      i--;
      j--;
    } else if (i > 0 && here === ((d[i - 1] as number[])[j] as number) + 1) {
      out.push({ kind: "del", ref: a[i - 1] });
      i--;
    } else {
      out.push({ kind: "ins", hyp: b[j - 1] });
      j--;
    }
  }
  return out.reverse();
}
