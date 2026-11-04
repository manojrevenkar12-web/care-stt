/** The three states every screen has besides its content: loading, failed, and nothing yet. */
import type { ReactNode } from "react";
import type { ApiError, Resource } from "../api";

const WIDTHS = ["w-full", "w-11/12", "w-4/5", "w-2/3"] as const;

export function Skeleton({ rows = 4, label }: { rows?: number; label: string }) {
  return (
    <output aria-live="polite" aria-label={label} className="flex flex-col gap-3 py-4">
      {Array.from({ length: rows }, (_, n) => ({ n, width: WIDTHS[n % WIDTHS.length] })).map((row) => (
        <span
          key={row.n}
          className={`block h-6 rounded-soft bg-idle motion-safe:animate-pulse ${row.width}`}
        />
      ))}
      <span className="sr-only">{label}</span>
    </output>
  );
}

export function ErrorState({ error, retry, what }: { error: ApiError; retry: () => void; what: string }) {
  return (
    <div role="alert" className="my-4 max-w-prose border-l-rule border-fault bg-faultWash p-4">
      <p className="font-bold">Could not load {what}.</p>
      <p className="mt-1 text-muted">{error.message}</p>
      <button
        type="button"
        onClick={retry}
        className="mt-3 rounded-soft border border-ink px-3 py-1 font-bold hover:bg-sheet"
      >
        Try again
      </button>
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="my-4 max-w-prose border border-dashed border-line p-6">
      <p className="font-bold">{title}</p>
      <div className="mt-1 text-muted">{children}</div>
    </div>
  );
}

/** Render a resource: skeleton while loading, an error with retry, the empty state, or the content. */
export function Load<T>({
  r,
  what,
  empty,
  isEmpty,
  rows,
  children,
}: {
  r: Resource<T>;
  what: string;
  empty?: ReactNode;
  isEmpty?: (d: T) => boolean;
  rows?: number;
  children: (d: T) => ReactNode;
}) {
  if (r.loading && !r.data) return <Skeleton rows={rows} label={`Loading ${what}`} />;
  if (r.error) return <ErrorState error={r.error} retry={r.retry} what={what} />;
  if (!r.data) return null;
  if (isEmpty?.(r.data) && empty) return <>{empty}</>;
  return <>{children(r.data)}</>;
}
