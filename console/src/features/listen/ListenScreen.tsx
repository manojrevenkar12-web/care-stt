import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { useResource } from "../../shared/api";
import { EmptyState, Load } from "../../shared/ui/states";
import { Transcript } from "../../shared/ui/Transcript";
import { count } from "../../shared/words";
import { motion as m } from "../../theme/tokens";
import { SYSTEMS } from "../releases/ReleasesScreen";

type Utt = {
  id: string;
  set: string;
  voice: string;
  condition: string;
  ref: string;
  hyp: string;
  biased: string[];
  chart_error: boolean;
  terms_missed: string[];
  numbers_ref: string[];
  numbers_hyp: string[];
  resident: string | null;
  resident_heard: boolean;
  facility: string | null;
  risk: number | null;
  flagged: boolean;
  seconds: number;
  audio_seconds: number;
};

const CONDITIONS: [string, string][] = [["", "All audio"], ["clean", "Clean"], ["babble_10", "Ward babble"], ["telephone", "Phone line"]];

export function ListenScreen() {
  const [system, setSystem] = useState("finetuned+bias");
  const [set, setSet] = useState("facility");
  const [cond, setCond] = useState("");
  const [errorsOnly, setErrorsOnly] = useState(true);
  const [open, setOpen] = useState<string | null>(null);
  const q = new URLSearchParams({ decoder: system, set, errors_only: String(errorsOnly), limit: "60" });
  if (cond) q.set("condition", cond);
  const r = useResource<{ total: number; items: Utt[] }>(`/v2/utterances?${q}`);
  return (
    <section aria-labelledby="listen-title">
      <h1 id="listen-title" className="font-text text-display">
        Listen to what went wrong
      </h1>
      <p className="mt-2 max-w-prose text-muted">Notes with a chart error first: a lost resident name, a lost drug or nursing term, or a number heard wrong. Play the audio and see where the words went.</p>
      <div className="mt-6 flex flex-wrap items-center gap-4 text-small">
        <Select label="System" value={system} onChange={setSystem} options={SYSTEMS.map((s) => [s.id, s.label])} />
        <Select label="Notes" value={set} onChange={setSet} options={[["facility", "Facility notes"], ["v1", "Hand-written notes"]]} />
        <Select label="Audio" value={cond} onChange={setCond} options={CONDITIONS} />
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={errorsOnly} onChange={(e) => setErrorsOnly(e.target.checked)} />
          Only notes with a chart error
        </label>
      </div>
      <Load
        r={r}
        what="the notes"
        rows={6}
        isEmpty={(d) => !d.items.length}
        empty={<EmptyState title={errorsOnly ? "No chart errors in this selection." : "No notes in this selection."}>{errorsOnly ? "Untick the filter to see every note." : "Run the benchmark for this system."}</EmptyState>}
      >
        {(d) => (
          <>
            <p className="mt-4 text-small text-muted">{count(d.total, "note")}{d.total > d.items.length ? `, first ${d.items.length} shown` : ""}</p>
            <ul className="mt-2 flex list-none flex-col gap-3 p-0">
              {d.items.map((u) => {
                const key = `${u.id}-${u.voice}-${u.condition}`;
                const isOpen = open === key;
                return (
                  <li key={key} className="rounded-soft border border-line bg-sheet">
                    <button type="button" aria-expanded={isOpen} onClick={() => setOpen(isOpen ? null : key)} className="flex w-full flex-wrap items-baseline gap-x-4 gap-y-1 p-4 text-left">
                      <span className="font-text">{u.ref}</span>
                      <span className="ml-auto flex gap-2 text-small">
                        {u.chart_error ? <Tag tone="wrong">chart error</Tag> : <Tag tone="right">chart right</Tag>}
                        {u.flagged ? <Tag tone="confirm">read back</Tag> : null}
                      </span>
                    </button>
                    <AnimatePresence initial={false}>
                      {isOpen && (
                        <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: m.settle, ease: m.ease }} className="overflow-hidden">
                          <Detail u={u} />
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </li>
                );
              })}
            </ul>
          </>
        )}
      </Load>
    </section>
  );
}

function Detail({ u }: { u: Utt }) {
  return (
    <div className="border-t border-line p-4">
      <p className="text-small text-muted">Heard</p>
      <Transcript said={u.ref} hyp={u.hyp} recovered={u.biased} label="Heard transcript" />
      <audio className="mt-4 w-full" controls preload="none" src={`/v2/audio/${encodeURIComponent(u.id)}?voice=${encodeURIComponent(u.voice)}&condition=${encodeURIComponent(u.condition)}`}>
        <track kind="captions" />
      </audio>
      <dl className="mt-4 grid gap-x-8 gap-y-2 text-small sm:grid-cols-2">
        {u.resident ? (
          <Row k="Resident">
            {u.resident} {u.resident_heard ? "heard" : "lost"}
          </Row>
        ) : null}
        <Row k="Terms lost">{u.terms_missed.length ? u.terms_missed.join(", ") : "none"}</Row>
        <Row k="Numbers">
          said {u.numbers_ref.join(" ") || "none"}, heard {u.numbers_hyp.join(" ") || "none"}
        </Row>
        <Row k="Completed from the facility list">{u.biased.length ? u.biased.join(", ") : "nothing"}</Row>
        <Row k="Gate">{u.risk === null ? "not trained" : `risk ${u.risk.toFixed(2)}, ${u.flagged ? "read back to the nurse" : "filed"}`}</Row>
        <Row k="Audio">
          {u.voice.replace("de_DE-", "")}, {u.condition.replace("_", " ")}, {u.audio_seconds.toFixed(1)} s decoded in {u.seconds.toFixed(2)} s
        </Row>
      </dl>
    </div>
  );
}

function Row({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-muted">{k}</dt>
      <dd className="m-0">{children}</dd>
    </div>
  );
}

function Tag({ tone, children }: { tone: "wrong" | "right" | "confirm"; children: React.ReactNode }) {
  const c = { wrong: "bg-wrongWash text-wrong", right: "bg-rightWash text-right", confirm: "bg-confirmWash text-confirm" }[tone];
  return <span className={`rounded-soft px-2 font-semibold ${c}`}>{children}</span>;
}

function Select({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: [string, string][] }) {
  return (
    <label className="flex items-center gap-2">
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value)} className="rounded-soft border border-line bg-sheet px-2 py-1">
        {options.map(([v, l]) => (
          <option key={v} value={v}>
            {l}
          </option>
        ))}
      </select>
    </label>
  );
}
