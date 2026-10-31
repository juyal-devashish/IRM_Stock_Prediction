"""Training methods: ERM, IRMv1, IRM-Games (MLP classifiers and linear regressors)."""

from irm_stock.methods.mlp_methods import train_erm, train_irm, train_irm_games

#: Canonical method keys, in the order they are run and plotted.
METHODS = ("erm", "irm", "irm_games")

#: Human-readable labels for plots and summaries.
METHOD_LABELS = {"erm": "ERM", "irm": "IRM", "irm_games": "IRM-Games"}

_ALIASES = {"irm_game": "irm_games", "irmgames": "irm_games", "irm-games": "irm_games"}


def normalize_method(name: str) -> str:
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in METHODS:
        raise ValueError(f"unknown method {name!r}; expected one of {', '.join(METHODS)}")
    return key


__all__ = [
    "METHODS",
    "METHOD_LABELS",
    "normalize_method",
    "train_erm",
    "train_irm",
    "train_irm_games",
]
