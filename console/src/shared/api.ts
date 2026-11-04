/** The API, and the one hook every screen loads with: loading, error (with retry) and data. */
import { useCallback, useEffect, useState } from "react";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(path, { ...init, headers: { accept: "application/json", ...(init?.headers ?? {}) } });
  } catch {
    throw new ApiError(0, "The console cannot reach the API. Start it with: uvicorn care_stt.review_api:app --port 8000");
  }
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new ApiError(r.status, (body as { detail?: string }).detail ?? `The API answered ${r.status}.`);
  return body as T;
}

export type Resource<T> = { data: T | null; error: ApiError | null; loading: boolean; retry: () => void };

export function useResource<T>(path: string | null): Resource<T> {
  const [state, setState] = useState<{ data: T | null; error: ApiError | null; loading: boolean }>({
    data: null,
    error: null,
    loading: !!path,
  });
  const [n, setN] = useState(0);
  // biome-ignore lint/correctness/useExhaustiveDependencies: n is the retry trigger
  useEffect(() => {
    if (!path) return;
    let live = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    api<T>(path)
      .then((data) => live && setState({ data, error: null, loading: false }))
      .catch((error: ApiError) => live && setState({ data: null, error, loading: false }));
    return () => {
      live = false;
    };
  }, [path, n]);
  const retry = useCallback(() => setN((x) => x + 1), []);
  return { ...state, retry };
}
