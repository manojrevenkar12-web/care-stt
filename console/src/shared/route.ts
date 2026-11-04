/** Hash routes: #/releases, #/listen, #/try. */
import { useEffect, useState } from "react";

export type Screen = "releases" | "listen" | "try";
const SCREENS: Screen[] = ["releases", "listen", "try"];

export function parse(hash: string): Screen {
  const s = hash.replace(/^#\/?/, "").split("/")[0];
  return SCREENS.find((x) => x === s) ?? "releases";
}

export const href = (s: Screen) => `#/${s}`;

export function useScreen(): Screen {
  const [s, set] = useState(() => parse(window.location.hash));
  useEffect(() => {
    const on = () => set(parse(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return s;
}
