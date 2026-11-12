"""Check class retention and resolve less aggressive ENN settings."""

import logging
from collections import Counter

from sklearn.base import clone


class SamplingClassLoss(ValueError):
    """Sampling removed a class from a training partition."""


def check_sampling_classes(pipeline, X, y, *, context):
    steps = getattr(pipeline, "named_steps", {})
    sampler = steps.get("sampler")
    if sampler is None:
        sampler = steps.get("oversampler")
    if sampler is None:
        return

    _, sampled_y = clone(sampler).fit_resample(X, y)
    before = Counter(y)
    after = Counter(sampled_y)
    if set(before) != set(after):
        retained = {label: after.get(label, 0) for label in before}
        raise SamplingClassLoss(
            f"{context}: {sampler!r}; "
            f"class counts before={dict(before)}, after={retained}"
        )


def resolve_enn_neighbors(pipeline, X, y, cv_splits, *, context):
    """Choose one valid neighbour count for all training partitions."""
    parameters = pipeline.get_params(deep=True)
    sampler = getattr(pipeline, "named_steps", {}).get("sampler")
    # Restrict fallback to ENN and the ENN component of SMOTEENN.
    sampler_name = type(sampler).__name__
    key = {"EditedNearestNeighbours": "sampler__n_neighbors",
           "SMOTEENN": "sampler__enn__n_neighbors"}.get(sampler_name)
    if key is None or key not in parameters:
        return {}

    requested = parameters[key]
    last_failure = None
    for neighbors in range(requested, 0, -1):
        candidate = clone(pipeline)
        candidate.set_params(**{key: neighbors})
        try:
            for fold, (train_indices, _) in enumerate(cv_splits, 1):
                check_sampling_classes(
                    candidate, X.iloc[train_indices], y[train_indices],
                    context=f"{context}, fold {fold}",
                )
            check_sampling_classes(
                candidate, X, y, context=f"{context}, final fit",
            )
        except SamplingClassLoss as error:
            last_failure = error
            logging.warning("ENN candidate rejected: %s", error)
            continue

        pipeline.set_params(**{key: neighbors})
        if neighbors != requested:
            logging.warning(
                "ENN fallback | %s | requested=%d | applied=%d",
                context, requested, neighbors,
            )
        return {key: neighbors}

    raise SamplingClassLoss(
        f"{context}: no ENN neighbour count from {requested} to 1 "
        "preserved every class. Sampling was not disabled."
    ) from last_failure
