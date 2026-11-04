import type { Config } from "tailwindcss";
import { bar, color, font, line, measure, radius, space, type } from "./src/theme/tokens";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    colors: { ...color, transparent: "transparent", current: "currentColor" },
    fontFamily: { ui: [...font.ui], text: [...font.text] },
    fontSize: Object.fromEntries(Object.entries(type).map(([k, [size, lineHeight]]) => [k, [size, { lineHeight }]])),
    spacing: { ...space, bar: bar.height, px: line.hair },
    borderRadius: radius,
    borderWidth: { DEFAULT: line.hair, 0: "0", rule: line.rule },
    extend: { maxWidth: measure, width: { rail: measure.rail } },
  },
  plugins: [],
} satisfies Config;
