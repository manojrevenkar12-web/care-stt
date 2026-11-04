import { ListenScreen } from "./features/listen/ListenScreen";
import { ReleasesScreen } from "./features/releases/ReleasesScreen";
import { TryScreen } from "./features/try/TryScreen";
import { type Screen, href, useScreen } from "./shared/route";

const NAV: [Screen, string][] = [
  ["releases", "Release"],
  ["listen", "Listen"],
  ["try", "Dictate"],
];

export function App() {
  const screen = useScreen();
  return (
    <div className="min-h-screen">
      <a href="#main" className="sr-only focus:not-sr-only">
        Skip to content
      </a>
      <header className="border-b border-line bg-sheet">
        <div className="mx-auto flex max-w-page flex-wrap items-center gap-x-8 gap-y-3 px-4 py-3">
          <p className="m-0 font-text text-lead font-semibold">care-stt</p>
          <nav aria-label="Main">
            <ul className="m-0 flex list-none gap-6 p-0">
              {NAV.map(([s, label]) => (
                <li key={s}>
                  <a href={href(s)} aria-current={screen === s ? "page" : undefined} className={`border-b-rule pb-1 ${screen === s ? "border-ink font-semibold" : "border-transparent text-muted hover:text-ink"}`}>
                    {label}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
          <p className="m-0 ml-auto text-small text-muted">Speech models for care documentation, judged on held-out voices</p>
        </div>
      </header>
      <main id="main" className="mx-auto max-w-page px-4 py-8">
        {screen === "releases" && <ReleasesScreen />}
        {screen === "listen" && <ListenScreen />}
        {screen === "try" && <TryScreen />}
      </main>
    </div>
  );
}
