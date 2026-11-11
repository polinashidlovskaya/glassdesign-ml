"""GlassDesign ML — user-facing descriptions and validated UI counts.

Presentation only. Do not change frozen ML predictions or scientific screening tiers.
"""
from __future__ import annotations
import math
import pandas as pd

PROPERTY_RU = {
    'Density293K': 'Плотность',
    'RefractiveIndex': 'Показатель преломления',
    'Tg': 'Температура стеклования (Tg)',
    'TdilatometricSoftening': 'Температура размягчения (Td)',
}


def validate_custom_oxides(bounds):
    """Readable prevalidation; underlying step391 remains authoritative."""
    if len(bounds) < 2:
        return 'Выбери как минимум два оксида для собственного состава.'
    for oxide, (lo, hi) in bounds.items():
        if not all(math.isfinite(float(v)) for v in (lo, hi)) or not (0 <= lo <= hi <= 100):
            return f'{oxide}: проверь границы доли (от 0 до 100 мол.%).'
    minimum = sum(float(x[0]) for x in bounds.values())
    maximum = sum(float(x[1]) for x in bounds.values())
    if minimum > 100 + 1e-8 or maximum < 100 - 1e-8:
        return ('С этими диапазонами невозможно получить состав суммой 100 мол.%. '
                'Уменьши минимумы или увеличь максимумы концентраций.')
    return None


def tier_letter(raw):
    s = str(raw).strip().upper()
    return s[0] if s and s[0] in 'ABCD' else None


def grade_title(tier):
    return {
        'A': 'A · Меньше выявленных ограничений',
        'B': 'B · Требуется дополнительная проверка',
        'C': 'C · Осторожно с прогнозом температуры',
        'D': 'D · Не хватает данных для оценки',
    }.get(tier, 'Проверка надёжности ещё не выполнялась')


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def experimental_checks(row=None):
    """Plain-language, non-diagnostic actions based on actual numeric fields.

    Tier is an existing step399/385 heuristic; this does not reclassify.
    """
    checks = []
    if row is not None:
        delta_gn = number(row.get('GN_Td_minus_Tg_C'))
        delta_gd = number(row.get('GD_Td_minus_Tg_C'))
        if (delta_gn is not None and delta_gn < 5) or (delta_gd is not None and delta_gd < 5):
            checks.append('Уточнить температуры: по одной из моделей размягчение наступает почти при Tg. Измерить Tg методом ДСК, а Td — дилатометрией.')
        local_jump = number(row.get('Max_local_Tg_jump_C'))
        if local_jump is not None and local_jump > 5:
            checks.append(f'Проверить устойчивость Tg: при изменении доли оксидов на 0,05 мол.% расчёт изменяется до {local_jump:.1f} °C. Желательно измерить Tg и для соседних составов.')
        gap_tg = number(row.get('Tg_model_disagreement_C'))
        if gap_tg is None:
            gd, gn = number(row.get('Tg_C')), number(row.get('GN_Tg_C'))
            if gd is not None and gn is not None:
                gap_tg = abs(gd-gn)
        if gap_tg is not None and gap_tg > 10:
            checks.append(f'Уточнить Tg экспериментально: две модели расходятся примерно на {gap_tg:.1f} °C.')
        near_td = number(row.get('Nearest_Tsoft_halfL1_mol'))
        if near_td is None or near_td > 5:
            # Only if the relevant diagnostic field is genuinely present.
            if 'Nearest_Tsoft_halfL1_mol' in row:
                checks.append('Найти дополнительные экспериментальные данные по Td: в обучающей/справочной базе нет близкого измерения в пределах выбранного порога.')
        if 'Screening_tier' in row and tier_letter(row['Screening_tier']) == 'D':
            checks.append('Проверить недостающие экспериментальные сведения, прежде чем выбирать состав для изготовления.')
    checks.extend([
        'Проверить, образуется ли однородное стекло при выбранном режиме получения (например, по внешнему виду и рентгенофазовому анализу).',
        'Измерить Tg методом ДСК и температуру размягчения Td дилатометрией; расчётные температуры не равны температуре плавления или изготовления.',
    ])
    # Keep order and avoid exact duplicates; no more than 5 user-facing actions.
    return list(dict.fromkeys(checks))[:5]


def literature_statistics(eligible, matches):
    """Different denominators: candidates, match rows, unique experimental IDs, sources."""
    n_candidates = len(eligible)
    if matches is None or matches.empty:
        return {'candidates': n_candidates, 'matched_candidates': 0, 'match_rows': 0,
                'unique_measurements': 0, 'unique_documents': 0, 'with_DOI_documents': 0}
    m = matches.copy()
    if 'Run_Unique_ID' in eligible and 'Run_Unique_ID' in m:
        valid_ids = set(eligible.Run_Unique_ID.astype(str))
        m = m.loc[m.Run_Unique_ID.astype(str).isin(valid_ids)]
    def nunique(col):
        return int(m[col].dropna().astype(str).nunique()) if col in m else 0
    if 'KOD_verified' in m:
        documented = m.loc[m.KOD_verified.notna()]
        if 'Primary_RefID' in documented:
            documented = documented.loc[documented.Primary_RefID.notna()]
            unique_docs = len(documented[['KOD_verified', 'Primary_RefID']].drop_duplicates())
        else:
            unique_docs = int(documented.KOD_verified.nunique())
    else:
        unique_docs = nunique('Source_URL')
    doi_docs = 0
    if 'Source_DOI' in m:
        doi = m.loc[m.Source_DOI.notna() & m.Source_DOI.astype(str).str.strip().ne('')]
        if 'KOD_verified' in doi:
            doi_docs = int(doi.KOD_verified.nunique())
        else:
            doi_docs = int(doi.Source_DOI.nunique())
    return {'candidates': n_candidates, 'matched_candidates': nunique('Run_Unique_ID'),
            'match_rows':len(m), 'unique_measurements':nunique('SciGlass_ID'),
            'unique_documents':int(unique_docs), 'with_DOI_documents':doi_docs}


def property_coverage(matches, eligible):
    """Return counts per property, independent of match-row total."""
    n = len(eligible)
    if matches is None or matches.empty:
        return {name:(0,n) for name in PROPERTY_RU}
    valid=set(eligible.Run_Unique_ID.astype(str))
    return {
        p:(int(matches.loc[matches.Property.eq(p) & matches.Run_Unique_ID.astype(str).isin(valid),
                            'Run_Unique_ID'].astype(str).nunique()),n)
        for p in PROPERTY_RU
    }
