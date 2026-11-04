/** Every colour role, size, duration and type step the console uses; components never carry their own.
 *
 * The console is where speech-model releases for care documentation are judged, so its look borrows
 * from a care record rather than a dashboard: a calm green-grey paper, blue-black ink, and four
 * colours that each mean one thing about a word: heard right, misheard, recovered from the
 * facility list, and waiting for a nurse to confirm. Transcripts are set in a serif because they
 * are what a person said; everything the system says is in Lexend, made for reading ease. */
export const color = {
  paper: "var(--paper)",
  sheet: "var(--sheet)",
  ink: "var(--ink)",
  muted: "var(--muted)",
  line: "var(--line)",
  right: "var(--right)",
  rightWash: "var(--right-wash)",
  wrong: "var(--wrong)",
  wrongWash: "var(--wrong-wash)",
  recovered: "var(--recovered)",
  recoveredWash: "var(--recovered-wash)",
  confirm: "var(--confirm)",
  confirmWash: "var(--confirm-wash)",
  focus: "var(--focus)",
  idle: "var(--idle)",
} as const;

export const font = {
  ui: ['"Lexend"', "system-ui", "sans-serif"],
  text: ['"Literata"', "Georgia", "serif"],
} as const;

/** Modular scale 1.2 from a 16 px body; [size, line-height]. */
export const type = {
  small: ["0.833rem", "1.3rem"],
  body: ["1rem", "1.55rem"],
  lead: ["1.2rem", "1.8rem"],
  title: ["1.44rem", "1.95rem"],
  display: ["2.074rem", "2.55rem"],
} as const;

export const space = { 0: "0", 1: "0.25rem", 2: "0.5rem", 3: "0.75rem", 4: "1rem", 6: "1.5rem", 8: "2rem", 12: "3rem" } as const;
export const radius = { none: "0", soft: "0.375rem", pill: "999px" } as const;
export const measure = { prose: "66ch", rail: "18rem", page: "78rem" } as const;
export const line = { hair: "1px", rule: "3px" } as const;
export const bar = { height: "0.625rem" } as const;

export const motion = {
  quick: 0.14,
  settle: 0.26,
  stagger: 0.035,
  ease: [0.2, 0, 0, 1] as [number, number, number, number],
} as const;
