import { useEffect, useRef, useState } from "react";
import type { LiveSnapshot, Results } from "./types";

export function useResults() {
  const [data, setData] = useState<Results | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const url = `${import.meta.env.BASE_URL}data/results.json`;
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`Could not load ${url} (${r.status})`);
        return r.json() as Promise<Results>;
      })
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, []);

  return { data, error };
}

export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(640);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.round(entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

export function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const stamped = document.documentElement.dataset.theme;
    if (stamped === "light" || stamped === "dark") return stamped;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch {
      /* storage unavailable: theme still applies for this visit */
    }
  };
  return [theme, toggle] as const;
}

/** Weekly walk-forward history; null when the file is absent (section hidden). */
export function useLive() {
  const [history, setHistory] = useState<LiveSnapshot[] | null>(null);
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/live.json`)
      .then((r) => (r.ok ? (r.json() as Promise<LiveSnapshot[]>) : null))
      .then((h) => setHistory(h && h.length ? h : null))
      .catch(() => setHistory(null));
  }, []);
  return history;
}
