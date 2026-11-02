import katex from "katex";
import { useMemo } from "react";

/** KaTeX-rendered math. Output includes MathML, so screen readers can read it. */
export function Tex({ children, display = false }: { children: string; display?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(children, { displayMode: display, throwOnError: true, output: "htmlAndMathml" }),
    [children, display],
  );
  return display ? (
    <div className="tex-display" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span className="tex-inline" dangerouslySetInnerHTML={{ __html: html }} />
  );
}
