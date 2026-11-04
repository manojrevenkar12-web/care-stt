import { motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import { useResource } from "../../shared/api";
import { EmptyState, Load } from "../../shared/ui/states";
import { Transcript } from "../../shared/ui/Transcript";
import { motion as m } from "../../theme/tokens";

type Score = {
  n: number;
  wer: number;
  term_recall: number;
  number_utt_acc: number;
  name_recall?: number;
  drug_recall_train?: number;
  drug_recall_heldout?: number;
  drug_recall_new?: number;
  fields_correct?: number;
  fields_wrong_silent?: number;
  rtf_mean: number;
};
type Metrics = Record<string, Record<string, Record<string, Score>>>;
type Gate = { flag_rate: number; recall: number; precision: number; silent_error_rate_without_gate: number; silent_error_rate_with_gate: number; test_notes: number };
type Utt = { id: string; voice: string; condition: string; ref: string; hyp: string; biased: string[] };

export const SYSTEMS: { id: string; label: string; note: string }[] = [
  { id: "stock", label: "Stock Whisper", note: "base, int8, greedy: what v1 shipped" },
  { id: "stock+lexicon-corrector", label: "Stock + v1 corrector", note: "post-correction against the global lexicon" },
  { id: "stock+facility-corrector", label: "Stock + facility corrector", note: "post-correction that also knows the facility list" },
  { id: "stock+bias", label: "Stock + biasing", note: "facility list in the decoder, beam 4" },
  { id: "finetuned", label: "Fine-tuned", note: "decoder trained on care German" },
  { id: "finetuned+bias", label: "Fine-tuned + biasing", note: "the release candidate" },
];

const pct = (x?: number) => (x === undefined || Number.isNaN(x) ? "–" : `${Math.round(x * 100)}%`);

export function ReleasesScreen() {
  const r = useResource<{ metrics: Metrics; gate: Gate | null }>("/v2/systems");
  const [cond, setCond] = useState("all");
  return (
    <section aria-labelledby="rel-title">
      <h1 id="rel-title" className="font-text text-display">
        Does the new model hear the care home?
      </h1>
      <p className="mt-2 max-w-prose text-lead text-muted">
        Every system decodes the same notes, spoken by two voices it never trained on. The facility notes name residents and drugs that appear nowhere in training: only the facility's own list can bring them in.
      </p>
      <Load r={r} what="the benchmark" rows={6}>
        {(d) => (
          <>
            <Examples />
            <div className="mt-12 flex flex-wrap items-end justify-between gap-4">
              <h2 className="m-0 text-title font-semibold">Facility notes</h2>
              <label className="flex items-center gap-2 text-small">
                Audio
                <select value={cond} onChange={(e) => setCond(e.target.value)} className="rounded-soft border border-line bg-sheet px-2 py-1">
                  <option value="all">All conditions</option>
                  <option value="clean">Clean</option>
                  <option value="babble_10">Ward babble, 10 dB</option>
                  <option value="telephone">Phone line</option>
                </select>
              </label>
            </div>
            <ScoreTable metrics={d.metrics} set="facility" cond={cond} cols={[
              ["Residents heard", (s) => pct(s.name_recall)],
              ["Plan drugs, seen in training", (s) => pct(s.drug_recall_train)],
              ["Plan drugs, held out", (s) => pct(s.drug_recall_heldout)],
              ["Plan drugs, never listed", (s) => pct(s.drug_recall_new)],
              ["Fields wrong, unflagged", (s) => pct(s.fields_wrong_silent)],
              ["Speed (× real time)", (s) => s.rtf_mean.toFixed(2)],
            ]} />
            <h2 className="mt-12 text-title font-semibold">Hand-written notes (v1 test)</h2>
            <ScoreTable metrics={d.metrics} set="v1" cond={cond} cols={[
              ["Word error rate", (s) => pct(s.wer)],
              ["Drug and nursing terms", (s) => pct(s.term_recall)],
              ["Notes with every number right", (s) => pct(s.number_utt_acc)],
              ["Fields wrong, unflagged", (s) => pct(s.fields_wrong_silent)],
            ]} />
            {d.gate ? <GateCard g={d.gate} /> : null}
          </>
        )}
      </Load>
    </section>
  );
}

function ScoreTable({ metrics, set, cond, cols }: { metrics: Metrics; set: string; cond: string; cols: [string, (s: Score) => string][] }) {
  const rows = SYSTEMS.filter((s) => metrics[s.id]?.[set]?.[cond]);
  if (!rows.length) return <EmptyState title="No results for this selection.">Run the benchmark for this test set and condition.</EmptyState>;
  return (
    <div className="mt-4 overflow-x-auto">
      <table className="w-full border-collapse text-left">
        <thead className="text-small text-muted">
          <tr>
            <th scope="col" className="border-b border-line py-2 pr-4 font-normal">System</th>
            {cols.map(([h]) => (
              <th key={h} scope="col" className="border-b border-line py-2 pr-4 text-right font-normal">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((s) => {
            const sc = metrics[s.id]?.[set]?.[cond] as Score;
            const release = s.id === "finetuned+bias";
            return (
              <tr key={s.id} className={`border-b border-line ${release ? "bg-sheet" : ""}`}>
                <th scope="row" className="py-2 pr-4 text-left font-normal">
                  <span className={release ? "font-semibold" : ""}>{s.label}</span>
                  <span className="block text-small text-muted">{s.note}</span>
                </th>
                {cols.map(([h, f]) => (
                  <td key={h} className="py-2 pr-4 text-right tabular-nums">{f(sc)}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** The signature: the same spoken notes, as the stock model and the release candidate heard them. */
function Examples() {
  const reduce = useReducedMotion();
  const a = useResource<{ items: Utt[] }>("/v2/utterances?decoder=stock&set=facility&condition=babble_10&limit=400");
  const b = useResource<{ items: Utt[] }>("/v2/utterances?decoder=finetuned%2Bbias&set=facility&condition=babble_10&limit=400");
  if (a.loading || b.loading || a.error || b.error || !a.data || !b.data) return null;
  const byKey = new Map(b.data.items.map((u) => [`${u.id}|${u.voice}`, u]));
  const pairs = a.data.items.map((u) => [u, byKey.get(`${u.id}|${u.voice}`)] as const).filter((p): p is readonly [Utt, Utt] => !!p[1]);
  const chosen = pairs.filter(([s, f]) => s.hyp !== f.hyp && f.biased.length > 0).slice(0, 3);
  if (!chosen.length) return null;
  return (
    <div className="mt-8 grid gap-6">
      {chosen.map(([s, f], i) => (
        <motion.figure
          key={s.id + s.voice}
          className="m-0 rounded-soft border border-line bg-sheet p-6"
          initial={reduce ? false : { opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * m.stagger * 4, duration: m.settle, ease: m.ease }}
        >
          <figcaption className="text-small text-muted">Said, with ward babble at 10 dB</figcaption>
          <p className="mt-1 font-text text-lead">{s.ref}</p>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div>
              <p className="text-small text-muted">Stock Whisper heard</p>
              <Transcript said={s.ref} hyp={s.hyp} label="Stock transcript" />
            </div>
            <div>
              <p className="text-small text-muted">Release candidate heard</p>
              <Transcript said={f.ref} hyp={f.hyp} recovered={f.biased} label="Release candidate transcript" />
            </div>
          </div>
        </motion.figure>
      ))}
    </div>
  );
}

function GateCard({ g }: { g: Gate }) {
  return (
    <section aria-labelledby="gate-title" className="mt-12 max-w-prose border-l-rule border-confirm bg-confirmWash p-4">
      <h2 id="gate-title" className="m-0 font-semibold">Read back before filing</h2>
      <p className="mt-1">
        The gate asks the nurse to confirm {pct(g.flag_rate)} of notes and catches {pct(g.recall)} of the notes that contain a chart error. Notes filed with an
        error drop from {pct(g.silent_error_rate_without_gate)} to {pct(g.silent_error_rate_with_gate)} of {g.test_notes} test notes.
      </p>
    </section>
  );
}
