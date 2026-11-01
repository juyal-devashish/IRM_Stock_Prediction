import { useCallback, useEffect, useRef, useState } from "react";
import type { Figure } from "../types";

/** One figure at a time. Swipe or scroll (scroll-snap), buttons, arrow keys. No autoplay. */
export function Carousel({ figures }: { figures: Figure[] }) {
  const track = useRef<HTMLDivElement>(null);
  const [index, setIndex] = useState(0);
  const base = import.meta.env.BASE_URL;

  // Keep the counter in sync with swipes and scrolling, not just button presses.
  useEffect(() => {
    const el = track.current;
    if (!el) return;
    const onScroll = () => {
      const i = Math.round(el.scrollLeft / el.clientWidth);
      setIndex(Math.max(0, Math.min(figures.length - 1, i)));
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [figures.length]);

  const go = useCallback(
    (i: number) => {
      const el = track.current;
      if (!el) return;
      const next = (i + figures.length) % figures.length;
      const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      el.scrollTo({ left: next * el.clientWidth, behavior: smooth ? "smooth" : "auto" });
      setIndex(next);
    },
    [figures.length],
  );

  return (
    <div
      className="carousel"
      role="region"
      aria-roledescription="carousel"
      aria-label="Figures from the report"
      onKeyDown={(e) => {
        if (e.key === "ArrowRight") {
          e.preventDefault();
          go(index + 1);
        } else if (e.key === "ArrowLeft") {
          e.preventDefault();
          go(index - 1);
        }
      }}
    >
      <div className="carousel-track" ref={track} tabIndex={0} aria-label="Use the arrow keys to change figure">
        {figures.map((f, i) => (
          <figure
            key={f.file}
            className="carousel-slide"
            role="group"
            aria-roledescription="slide"
            aria-label={`${i + 1} of ${figures.length}`}
            aria-hidden={i !== index}
          >
            <a href={`${base}${f.file}`} tabIndex={i === index ? 0 : -1}>
              <img src={`${base}${f.file}`} alt={f.caption} loading="eager" />
            </a>
          </figure>
        ))}
      </div>

      <div className="carousel-bar">
        <button type="button" className="chip small" onClick={() => go(index - 1)} aria-label="Previous figure">
          Previous
        </button>
        <p className="carousel-caption" aria-live="polite">
          <span className="carousel-count">
            {index + 1} / {figures.length}
          </span>
          {figures[index]?.caption}
        </p>
        <button type="button" className="chip small" onClick={() => go(index + 1)} aria-label="Next figure">
          Next
        </button>
      </div>

      <div className="carousel-dots" role="group" aria-label="Choose a figure">
        {figures.map((f, i) => (
          <button
            key={f.file}
            type="button"
            className="dot"
            aria-label={`Figure ${i + 1}: ${f.caption}`}
            aria-current={i === index ? "true" : undefined}
            onClick={() => go(i)}
          />
        ))}
      </div>
    </div>
  );
}
