import numpy as np
import pandas as pd


PROPERTY_COLUMNS = (
    "Density_g_cm3",
    "RefractiveIndex",
    "Tg_C",
    "Tsoft_C"
)


def validate_bounds(bounds, allowed_oxides):
    """Проверка химических ограничений."""

    if not isinstance(bounds, dict) or not bounds:
        raise ValueError("Не задан химический состав.")

    allowed = set(allowed_oxides)
    cleaned = {}

    for oxide, limits in bounds.items():

        if oxide not in allowed:
            raise ValueError(
                f"Неподдерживаемый оксид: {oxide}"
            )

        if not isinstance(limits, (list, tuple)):
            raise ValueError(
                f"{oxide}: нужен диапазон [min, max]"
            )

        if len(limits) != 2:
            raise ValueError(
                f"{oxide}: нужны две границы"
            )

        low, high = map(float, limits)

        if not (
            np.isfinite(low)
            and np.isfinite(high)
            and 0 <= low <= high <= 100
        ):
            raise ValueError(
                f"{oxide}: некорректный диапазон"
            )

        cleaned[oxide] = (low, high)

    minimum = sum(
        low for low, high in cleaned.values()
    )

    maximum = sum(
        high for low, high in cleaned.values()
    )

    if minimum > 100 + 1e-8:
        raise ValueError(
            "Сумма минимумов превышает 100 мол.%"
        )

    if maximum < 100 - 1e-8:
        raise ValueError(
            "Сумма максимумов меньше 100 мол.%"
        )

    return cleaned


def sample_composition(bounds, rng):
    """Создание одного математически допустимого состава."""

    names = list(bounds)

    lower = np.array(
        [bounds[name][0] for name in names],
        dtype=float
    )

    upper = np.array(
        [bounds[name][1] for name in names],
        dtype=float
    )

    values = lower.copy()
    remaining = 100.0 - lower.sum()

    order = rng.permutation(len(names))

    for position, index in enumerate(order):

        rest = order[position + 1:]

        other_capacity = (
            upper[rest] - lower[rest]
        ).sum()

        min_add = max(
            0.0,
            remaining - other_capacity
        )

        max_add = min(
            upper[index] - values[index],
            remaining
        )

        if min_add > max_add + 1e-7:
            raise RuntimeError(
                "Невозможно распределить концентрации"
            )

        if position == len(order) - 1:
            addition = remaining

        elif max_add - min_add <= 1e-10:
            addition = min_add

        else:
            addition = rng.uniform(
                min_add, max_add
            )

        values[index] += addition
        remaining -= addition

    if not np.isclose(
        values.sum(), 100.0, atol=1e-6
    ):
        raise RuntimeError(
            "Нарушена сумма концентраций"
        )

    if (
        (values < lower - 1e-7).any()
        or (values > upper + 1e-7).any()
    ):
        raise RuntimeError(
            "Нарушены границы концентраций"
        )

    return dict(zip(names, values))


def generate_compositions(
    families,
    allowed_oxides,
    count=700,
    random_seed=42
):
    """Генерация составов из заданных семейств."""

    if int(count) != count or count < 1:
        raise ValueError(
            "Количество должно быть положительным целым"
        )

    count = int(count)

    if not families:
        raise ValueError(
            "Не выбраны химические семейства"
        )

    checked = {
        name: validate_bounds(bounds, allowed_oxides)
        for name, bounds in families.items()
    }

    rng = np.random.default_rng(random_seed)

    names = list(checked)
    ordered_oxides = list(allowed_oxides)

    rows = []
    seen = set()
    attempts = 0

    max_attempts = max(10000, count * 100)

    while len(rows) < count:

        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Не удалось создать заданное "
                "число уникальных составов"
            )

        family = names[
            len(rows) % len(names)
        ]

        composition = sample_composition(
            checked[family], rng
        )

        full = {
            oxide: float(
                composition.get(oxide, 0.0)
            )
            for oxide in ordered_oxides
        }

        key = tuple(
            round(full[oxide], 6)
            for oxide in ordered_oxides
        )

        if key in seen:
            continue

        seen.add(key)

        rows.append({
            "Candidate_ID": len(rows) + 1,
            "Glass_family": family,
            **full
        })

    result = pd.DataFrame(rows)

    if not np.allclose(
        result[ordered_oxides].sum(axis=1),
        100.0,
        atol=1e-6
    ):
        raise RuntimeError(
            "Ошибка баланса составов"
        )

    return result


def screen_predictions(
    data,
    criteria,
    exclude_inconsistent=True
):
    """Отбор стекол по заданным свойствам."""

    if not criteria:
        raise ValueError(
            "Выбери хотя бы одну характеристику"
        )

    result = data.copy()

    mask = pd.Series(
        True,
        index=result.index
    )

    result["Target_score"] = 0.0

    for column, limits in criteria.items():

        if column not in PROPERTY_COLUMNS:
            raise ValueError(
                f"Неизвестное свойство: {column}"
            )

        if column not in result.columns:
            raise ValueError(
                f"Нет рассчитанного свойства: {column}"
            )

        low, high = limits

        if low is not None:
            low = float(low)

        if high is not None:
            high = float(high)

        if low is None and high is None:
            raise ValueError(
                f"{column}: границы не заданы"
            )

        if (
            low is not None
            and not np.isfinite(low)
        ) or (
            high is not None
            and not np.isfinite(high)
        ):
            raise ValueError(
                f"{column}: граница не является числом"
            )

        if (
            low is not None
            and high is not None
            and low > high
        ):
            raise ValueError(
                f"{column}: минимум больше максимума"
            )

        if low is not None:
            mask &= result[column] >= low

        if high is not None:
            mask &= result[column] <= high

        # Оценка близости к центру диапазона.
        # Это эвристика ранжирования, а не
        # оценка экспериментальной надёжности.

        if low is not None and high is not None:

            midpoint = (low + high) / 2.0

            scale = max(
                (high - low) / 2.0,
                1e-8
            )

            result["Target_score"] += (
                (result[column] - midpoint).abs()
                / scale
            )

    result["Temperature_consistent"] = (
        result["Tsoft_C"] >= result["Tg_C"]
    )

    if exclude_inconsistent:
        mask &= result["Temperature_consistent"]

    result["Pass_user_filters"] = mask

    selected = result.loc[mask].copy()

    selected = selected.sort_values(
        ["Target_score", "Candidate_ID"]
    )

    return result, selected
