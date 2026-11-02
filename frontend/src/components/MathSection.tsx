import { METHOD_INFO } from "../methods";
import { Tex } from "./Tex";

/** The setup and each method's objective, matching src/irm_stock/{data,methods}. */
export function MathSection() {
  return (
    <>
      <h3 className="spaced">The setup</h3>
      <div className="math-block">
        <p>
          Each environment <Tex>e</Tex> gives pairs <Tex>{"(x, y)"}</Tex> with <Tex>{"y \\in \\{0, 1\\}"}</Tex>. For
          stocks, <Tex>{"r_t = p_t / p_{t-1} - 1"}</Tex> is the daily return and
        </p>
        <Tex display>{"x_t = \\big(r_{t-10}, \\dots, r_{t-1},\\; s_t\\big), \\qquad y_t = \\mathbf{1}[\\,r_{t+1} > 0\\,]"}</Tex>
        <p>
          The shortcut <Tex>{"s_t"}</Tex> copies the label, flipped at random so that it agrees with probability{" "}
          <Tex>{"a_e"}</Tex>:
        </p>
        <Tex display>
          {"\\Pr(s_t = y_t) = a_e, \\qquad a_e \\in \\{0.8,\\ 0.9\\} \\text{ in training}, \\quad a_e = 0.1 \\text{ at test}"}
        </Tex>
        <p>
          A model <Tex>f</Tex> outputs a logit. Its risk in an environment is the average logistic loss, with{" "}
          <Tex>{"\\sigma(z) = 1/(1 + e^{-z})"}</Tex>:
        </p>
        <Tex display>
          {"R^e(f) = \\mathbb{E}_{(x,y)\\sim e}\\Big[-y \\log \\sigma\\big(f(x)\\big) - (1-y)\\log\\big(1-\\sigma(f(x))\\big)\\Big]"}
        </Tex>
      </div>

      <h3 className="spaced">How each method is trained</h3>
      <dl className="methods methods-math">
        <div>
          <dt>
            <span className="swatch" style={{ background: "var(--m-erm)" }} />
            {METHOD_INFO.erm.label}
          </dt>
          <dd>
            {METHOD_INFO.erm.blurb} The training environments are pooled, each weighted by its size{" "}
            <Tex>{"n_e"}</Tex>:
            <Tex display>{"\\min_f\\; \\sum_{e \\in \\mathcal{E}_{\\text{tr}}} \\frac{n_e}{n}\\, R^e(f)"}</Tex>
          </dd>
        </div>
        <div>
          <dt>
            <span className="swatch" style={{ background: "var(--m-irm)" }} />
            {METHOD_INFO.irm.label} (IRMv1)
          </dt>
          <dd>
            {METHOD_INFO.irm.blurb} The penalty is the squared gradient of each environment's risk with respect to a
            fixed scalar multiplier <Tex>w = 1</Tex> on the output. It is zero only when <Tex>f</Tex> is already optimal
            in that environment:
            <Tex display>
              {"\\min_f\\; \\frac{1}{|\\mathcal{E}_{\\text{tr}}|} \\sum_{e} \\Big[ R^e(f) + \\lambda \\big( \\nabla_{w \\mid w=1} R^e(w \\cdot f) \\big)^2 \\Big]"}
            </Tex>
            For the logistic loss the gradient term is
            <Tex display>{"\\nabla_{w \\mid w=1} R^e(w \\cdot f) = \\mathbb{E}_e\\big[f(x)\\,\\big(\\sigma(f(x)) - y\\big)\\big]"}</Tex>
            <Tex>{"\\lambda = 1"}</Tex> for the first 100 steps, then <Tex>{"\\lambda = 10^4"}</Tex>, with the
            gradient divided by <Tex>{"\\lambda"}</Tex> so the step size stays comparable.
          </dd>
        </div>
        <div>
          <dt>
            <span className="swatch" style={{ background: "var(--m-irm_games)" }} />
            {METHOD_INFO.irm_games.label}
          </dt>
          <dd>
            {METHOD_INFO.irm_games.blurb} Each environment has its own model <Tex>{"f_e"}</Tex>, and the prediction is
            their average. In turn, each player takes a step on its own environment's risk of the ensemble, holding the
            others fixed:
            <Tex display>
              {"\\bar f = \\frac{1}{|\\mathcal{E}_{\\text{tr}}|} \\sum_{k} f_k, \\qquad \\min_{f_e}\\; R^e\\big(\\bar f\\big) \\text{ for each } e"}
            </Tex>
            Shown in the tables only, not the charts.
          </dd>
        </div>
      </dl>
    </>
  );
}
