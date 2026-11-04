import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { ApiError, api } from "../../shared/api";
import { motion as m } from "../../theme/tokens";

type Answer = {
  transcript: string;
  completed_from_facility_list: string[];
  risk: number | null;
  decision: "file" | "confirm";
  vitals: Record<string, number>;
  doses: { drug: string | null; amount: number; unit: string }[];
  review_flags: { field: string; reason: string }[];
  timing: { decode_s: number; audio_s: number; rtf: number };
};

const FACILITIES = ["Haus Lindenhof", "Seniorenresidenz Am Park", "Pflegezentrum Nordufer", "Haus Sonnenhang"];

export function TryScreen() {
  const [file, setFile] = useState<File | null>(null);
  const [facility, setFacility] = useState(FACILITIES[0] as string);
  const [state, setState] = useState<{ busy: boolean; error: string | null; answer: Answer | null }>({ busy: false, error: null, answer: null });
  const send = async () => {
    if (!file) return;
    setState({ busy: true, error: null, answer: null });
    const body = new FormData();
    body.append("audio", file);
    body.append("facility", facility);
    try {
      setState({ busy: false, error: null, answer: await api<Answer>("/v2/transcribe", { method: "POST", body }) });
    } catch (e) {
      setState({ busy: false, error: e instanceof ApiError ? e.message : String(e), answer: null });
    }
  };
  const a = state.answer;
  return (
    <section aria-labelledby="try-title">
      <h1 id="try-title" className="font-text text-display">
        Dictate a note
      </h1>
      <p className="mt-2 max-w-prose text-muted">Upload a WAV or FLAC recording and pick the care home. The model decodes it with that home's residents and medication plan, then the gate decides whether it can be filed or must be read back.</p>
      <div className="mt-6 flex flex-wrap items-end gap-4">
        <label className="flex flex-col gap-1 text-small">
          Recording
          <input type="file" accept=".wav,.flac,audio/wav,audio/flac" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </label>
        <label className="flex flex-col gap-1 text-small">
          Care home
          <select value={facility} onChange={(e) => setFacility(e.target.value)} className="rounded-soft border border-line bg-sheet px-2 py-1">
            {FACILITIES.map((f) => (
              <option key={f}>{f}</option>
            ))}
          </select>
        </label>
        <button type="button" disabled={!file || state.busy} onClick={send} className="rounded-soft bg-ink px-4 py-2 font-semibold text-paper disabled:opacity-60">
          {state.busy ? "Transcribing…" : "Transcribe"}
        </button>
      </div>
      {!a && !state.error && !state.busy ? <p className="mt-8 max-w-prose rounded-soft border border-dashed border-line p-6 text-muted">No note yet. Choose a recording to see the transcript, the fields it fills and the gate's decision.</p> : null}
      {state.busy ? (
        <output className="mt-8 block h-12 w-2/3 rounded-soft bg-idle motion-safe:animate-pulse" aria-label="Transcribing" />
      ) : null}
      {state.error ? (
        <div role="alert" className="mt-8 max-w-prose border-l-rule border-wrong bg-wrongWash p-4">
          <p className="font-semibold">The note could not be transcribed.</p>
          <p className="mt-1 text-muted">{state.error}</p>
          <button type="button" onClick={send} className="mt-3 rounded-soft border border-ink px-3 py-1 font-semibold">
            Try again
          </button>
        </div>
      ) : null}
      <AnimatePresence>
        {a ? (
          <motion.article initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: m.settle, ease: m.ease }} className="mt-8 max-w-prose rounded-soft border border-line bg-sheet p-6">
            <p className={`m-0 inline-block rounded-soft px-2 font-semibold ${a.decision === "file" ? "bg-rightWash text-right" : "bg-confirmWash text-confirm"}`}>
              {a.decision === "file" ? "Filed to the chart" : "Read back to the nurse first"}
            </p>
            <p className="mt-3 font-text text-lead">{a.transcript}</p>
            <p className="mt-2 text-small text-muted">
              Completed from the facility list: {a.completed_from_facility_list.join(", ") || "nothing"}. Risk {a.risk === null ? "not available" : a.risk.toFixed(2)}. {a.timing.audio_s} s of audio in {a.timing.decode_s} s.
            </p>
            {a.doses.length || Object.keys(a.vitals).length ? (
              <ul className="mt-3 list-disc pl-6">
                {Object.entries(a.vitals).map(([k, v]) => (
                  <li key={k}>
                    {k}: {v}
                  </li>
                ))}
                {a.doses.map((d) => (
                  <li key={`${d.drug}-${d.amount}`}>
                    {d.drug ?? "unknown drug"} {d.amount} {d.unit}
                  </li>
                ))}
              </ul>
            ) : null}
          </motion.article>
        ) : null}
      </AnimatePresence>
    </section>
  );
}
