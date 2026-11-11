"""GlassDesign ML 409 — goal-centered top 15 and referenced literature and genuine prediction progress, backend 391/399/398.

Presentation-only update. No model training, no GA changes, no rewriting old runs.
"""
from __future__ import annotations

import io
import os
import zipfile
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from glass391_engine import PROPS, discover_root, load_catalog, run
from glass399_integrated import (
    latest_verified_run, load_verified_run, report_directory,
    compute_scientific_report,
    load_cached_science,
)
from glass403_ranking import recommend_top_diverse
from glass405_published_table import group_measured_glasses, render_grouped_table
from glass408_source_access import resolve_source
from glass409_doi_enrichment import enrich, load_enriched
from glass403_literature import (compute_literature_403, load_cached_literature_403,
                                  distance_description)
from glass401_public_logic import (
    PROPERTY_RU, validate_custom_oxides, tier_letter, grade_title,
    experimental_checks, literature_statistics, property_coverage,
)

st.set_page_config(page_title='GlassDesign ML', page_icon='🧪', layout='wide',
                   initial_sidebar_state='expanded')

LABELS = {
    'Density_g_cm3': ('Плотность, г/см³', '3.2', '5.3'),
    'RefractiveIndex': ('Показатель преломления', '1.62', '1.90'),
    'Tg_C': ('Температура стеклования Tg, °C', '400', '650'),
    'Tsoft_C': ('Температура размягчения Td, °C', '480', '795'),
}
# Fixed internal research settings; not user-facing controls.
_SEARCH_BUDGET = 120
_SEARCH_FAMILIES = 4
_SEARCH_SEED = 71


def fmt(v, places=2):
    try:
        return f'{float(v):.{places}f}'.replace('.', ',') if pd.notna(v) else '—'
    except (ValueError, TypeError):
        return '—'


def parse(v):
    s = str(v).strip().replace(',', '.')
    if not s:
        return None
    n = float(s)
    if not np.isfinite(n):
        raise ValueError('Введи обычное число, например 4,5')
    return n


def formula(row, oxides):
    return ' · '.join(f'{o} {fmt(row[o])} %' for o in oxides
                    if o in row and pd.notna(row[o]) and float(row[o]) >= .05)



def readable_analogue_composition(value):
    """Форматируем полный состав экспериментальной записи SciGlass в мол.%.

    Не восстанавливаем состав из DOI или из состава кандидата.
    При отсутствии исходных данных честно показываем это пользователю.
    """
    if pd.isna(value) or not str(value).strip():
        return 'Состав не указан в SciGlass'
    subscripts=str.maketrans('0123456789','₀₁₂₃₄₅₆₇₈₉')
    result=[]
    for item in str(value).split(';'):
        item=item.strip()
        if not item:
            continue
        match=re.fullmatch(r'([^\s;]+)\s+([+-]?\d+(?:\.\d+)?)',item)
        if not match:
            result.append(item)
            continue
        oxide, concentration=match.groups()
        amount=('%0.2f' % float(concentration)).rstrip('0').rstrip('.').replace('.',',')
        result.append(oxide.translate(subscripts)+' '+amount+'%')
    return ' · '.join(result) if result else 'Состав не указан в SciGlass'

def compact_source(rec):
    title = rec.get('Source_title')
    if pd.notna(title) and str(title).strip():
        return str(title).strip()
    author = rec.get('SciGlass_author', '')
    year = rec.get('SciGlass_year', '')
    return f'{author}, {str(year).split(".")[0]}'.strip(', ')





# Возвращено из реального старого приложения app_glassdesign_381_progress.py.
# Показываем число выполненных предсказаний GlassNet, а не таймер или выдуманный процент.
class _ProgressGlassNet391:
    def __init__(self, inner, meter, total):
        self.inner = inner
        self.meter = meter
        self.total = max(1, int(total))

    @property
    def rows(self):
        return self.inner.rows

    @property
    def calls(self):
        return self.inner.calls

    def __call__(self, frame):
        result = self.inner(frame)
        done = int(self.inner.rows)
        fraction = min(1., done / self.total)
        pct = min(88, 12 + int(76 * fraction))
        self.meter.progress(
            pct, text=f'Подбор стекол: проверено {done} из {self.total} составов'
        )
        return result

    def attach_littleton(self, frame):
        # Обязательный метод версии 391: оставляем аудит Littleton Tsoft неизменным.
        return self.inner.attach_littleton(frame)


def _finish_candidate_reports(verified, meter):
    # Вторичные отчёты вычисляются только по той же таблице, которая прошла фильтр 391.
    out = report_directory(ROOT, verified['run_folder'])
    fingerprint = verified['input_fingerprint']
    failures = []
    if verified['tables']['both_pass_recipes'].empty:
        meter.progress(100, text='Подбор завершён. Рекомендованных составов не найдено.')
        return failures

    existing_science = load_cached_science(out, fingerprint)
    if existing_science is None:
        meter.progress(93, text='Проверяем, насколько устойчивы прогнозы…')
        try:
            compute_scientific_report(verified, ROOT, out)
        except Exception:
            failures.append('Оценку надёжности некоторых прогнозов выполнить не удалось.')
    meter.progress(96, text='Проверяем источники измерений SciGlass…')
    existing_lit = load_cached_literature_403(out, fingerprint)
    if existing_lit is None:
        try:
            def update_lit(done, total):
                pct=min(99, 96 + int(3*done/max(1,total)))
                meter.progress(pct, text=f'Литература: проверено {done} из {total} составов')
            compute_literature_403(verified, ROOT, out, progress=update_lit,
                                   top_k=3, near_threshold=5.)
        except Exception as exc:
            # User sees readable status; developer traceback goes to Colab log.
            import traceback
            print('ERROR 403 literature:', traceback.format_exc())
            failures.append('Библиографию сейчас не удалось загрузить. Можно повторить поиск.')
    meter.progress(100, text='Готово: составы и научные источники загружены.')
    return failures

st.title('🧪 GlassDesign ML')
st.markdown('**Подбор составов оксидных стёкол по заданным свойствам**')
st.caption('Программа предлагает составы для дальнейшей экспериментальной проверки.')

ROOT = discover_root()
if ROOT is None:
    st.error('Не удалось подключить проект на Google Drive. Открой сайт из установочного блокнота Colab.')
    st.stop()
try:
    _, OXIDES, ALL_FAMILIES = load_catalog(ROOT)
    # Offer only families actually supported by the unchanged generation backend.
    from glass_benchmark_v21 import wide_family_filter
    FAMILIES, _ = wide_family_filter(ALL_FAMILIES, OXIDES, max_components=12)
except Exception as error:
    st.error('Не удалось загрузить список химических систем: ' + str(error))
    st.stop()

# The mode selector is deliberately OUTSIDE st.form.
# Streamlit forms delay reruns, making conditional family/oxide controls invisible.
with st.sidebar:
    st.header('Подбор состава')
    st.subheader('1. Химическая система')
    mode_ru = st.radio('Как задать состав?',
        ('Пусть программа подберёт систему', 'Выбрать готовые семейства',
         'Собственный набор оксидов'),
        key='ui401_mode',
    )
    mode = {
        'Пусть программа подберёт систему': 'auto',
        'Выбрать готовые семейства': 'families',
        'Собственный набор оксидов': 'manual',
    }[mode_ru]
    chosen_families, manual_bounds, excluded = [], {}, []
    if mode == 'families':
        chosen_families = st.multiselect(
            'Какие семейства стекол рассмотреть?', sorted(FAMILIES),
            placeholder='Выбери одно или несколько семейств', key='ui401_families')
        st.caption('Поиск будет проводиться только в выбранных семействах.')
    elif mode == 'manual':
        defaults = [x for x in ('SiO2', 'B2O3', 'ZnO') if x in OXIDES]
        chosen_oxides = st.multiselect(
            'Какие оксиды будут входить в состав?', OXIDES,
            default=defaults,
            placeholder='Выбери компоненты смеси', key='ui401_oxides')
        initial = {'SiO2': (40., 75.), 'B2O3': (10., 45.), 'ZnO': (0., 40.),
                   'TeO2': (60., 90.), 'Na2O': (0., 20.)}
        for oxide in chosen_oxides:
            st.markdown(f'**{oxide}, мол.%**')
            a, b = st.columns(2)
            v1, v2 = initial.get(oxide, (0., 100.))
            lo = a.number_input('От', min_value=0., max_value=100.,
                                value=float(v1), step=1., key='ui401_min_'+oxide)
            hi = b.number_input('До', min_value=0., max_value=100.,
                                value=float(v2), step=1., key='ui401_max_'+oxide)
            manual_bounds[oxide] = (lo, hi)
        if manual_bounds:
            problem = validate_custom_oxides(manual_bounds)
            if problem:
                st.caption('Проверь выбранные диапазоны концентраций.')
    else:
        excluded = st.multiselect(
            'Какие оксиды нельзя использовать? (необязательно)',
            OXIDES, placeholder='Можно оставить пустым', key='ui401_excluded')

    st.subheader('2. Нужные свойства')
    chosen_props = st.multiselect(
        'Какие свойства важны?', list(PROPS), default=list(PROPS),
        format_func=lambda prop: LABELS[prop][0], key='ui401_properties')
    raw_bounds = {}
    for prop in chosen_props:
        label, default_lo, default_hi = LABELS[prop]
        st.markdown(f'**{label}**')
        a, b = st.columns(2)
        low = a.text_input('От', value=default_lo, key='ui401_lower_'+prop)
        high = b.text_input('До', value=default_hi, key='ui401_upper_'+prop)
        raw_bounds[prop] = (low, high)
    st.caption('Можно оставить одну границу пустой.')

    generate_clicked = st.button('Подобрать составы', type='primary',
                                 use_container_width=True, key='ui401_generate')
    restore_clicked = st.button('Показать последний расчёт',
                                use_container_width=True, key='ui401_restore')

if restore_clicked:
    meter = st.progress(0, text='Открываем сохранённые составы…')
    try:
        old = latest_verified_run(ROOT)
        meter.progress(90, text='Сохранённый расчёт открыт.')
        warnings = _finish_candidate_reports(old, meter)
        st.session_state['ui401_current_run'] = str(old['run_folder'])
        st.session_state['ui402_report_notes'] = warnings
        st.rerun()
    except Exception as error:
        meter.empty()
        st.error('Не удалось открыть сохранённый результат: ' + str(error))

if generate_clicked:
    try:
        if not chosen_props:
            raise ValueError('Выбери хотя бы одно свойство.')
        if mode == 'families' and not chosen_families:
            raise ValueError('Выбери хотя бы одно семейство стекол.')
        if mode == 'manual':
            message = validate_custom_oxides(manual_bounds)
            if message:
                raise ValueError(message)
        criteria = {p: (parse(lo), parse(hi)) for p, (lo, hi) in raw_bounds.items()}
        from glass391_engine import load_backends, select_families
        from glass391_glassnet import FrozenGlassNetAligned391
        engine_catalog = load_backends(Path(__file__).resolve().parent)
        requested_families = select_families(
            FAMILIES, mode, chosen_families, manual_bounds, excluded, OXIDES
        )
        expected = 8 * len(requested_families) + _SEARCH_BUDGET
        meter = st.progress(0, text='Проверяем параметры подбора…')

        def gd_factory(root, features, oxides):
            meter.progress(2, text='Загружаем модели свойств стекол…')
            predictor = engine_catalog['glass_benchmark_v21'].FrozenPredictor(
                root, features, oxides
            )
            meter.progress(6, text='Модели GlassDesign ML готовы; загружаем GlassNet…')
            return predictor

        def gn_factory(oxides):
            meter.progress(8, text='Готовим независимый расчёт GlassNet…')
            predictor = FrozenGlassNetAligned391(oxides)
            meter.progress(12, text=f'Модели загружены. Проверено 0 из {expected} составов')
            return _ProgressGlassNet391(predictor, meter, expected)

        fresh = run(criteria, root=ROOT, mode=mode,
                    selected_names=chosen_families, manual_bounds=manual_bounds,
                    excluded_oxides=excluded, budget=_SEARCH_BUDGET,
                    max_families=_SEARCH_FAMILIES, seed=_SEARCH_SEED,
                    gd_factory=gd_factory, gn_factory=gn_factory)
        meter.progress(90, text='Подбор завершён. Проверяем отобранные составы…')
        loaded = load_verified_run(fresh['folder'])
        meter.progress(92, text='Результаты проверены и сохранены.')
        warnings = _finish_candidate_reports(loaded, meter)
        st.session_state['ui401_current_run'] = str(loaded['run_folder'])
        st.session_state['ui402_report_notes'] = warnings
        st.rerun()
    except Exception as error:
        st.error('Не получилось подобрать составы: ' + str(error))

folder = st.session_state.get('ui401_current_run')
if not folder:
    st.caption('Выбери химическую систему и свойства слева, затем нажми «Подобрать составы».')
    st.stop()

try:
    RESULT = load_verified_run(folder)
except Exception as error:
    st.error('Не удалось открыть результаты: ' + str(error))
    st.stop()

T = RESULT['tables']
GOOD, BAD = T['both_pass_recipes'], T['excluded_from_recommendations']
PICKED, ALL = T['selected_representatives'], T['all_new_recipes']
REPORT = report_directory(ROOT, RESULT['run_folder'])
FINGERPRINT = RESULT['input_fingerprint']
SCIENCE = load_cached_science(REPORT, FINGERPRINT)
LITERATURE = load_cached_literature_403(REPORT, FINGERPRINT)

# DOI enrichment for any NEW confirmed SciGlass reference, not just the fixed
# sources of the first 29 tellurite glass candidates. A reusable catalogue on
# Drive avoids searching the same paper at every new generator run.
# No network requests are made when this exact run's enriched CSV is cached.
DOI_REPORT = ROOT/'GlassDesignML_Upgrade'/'SciGlass_DOI_Enrichment_step409'/str(FINGERPRINT)[:20]
if LITERATURE is not None and not LITERATURE.empty:
    enriched = load_enriched(DOI_REPORT, FINGERPRINT)
    if enriched is not None:
        LITERATURE = enriched
    else:
        st.caption('Уточняем ссылки на публикации: проверяем названия, авторов и годы…')
        doi_meter = st.progress(0, text='Проверяем DOI научных источников…')
        def doi_update(done, total, status):
            doi_meter.progress(min(100, int(100*done/max(total,1))),
                text=f'Проверено {done} из {total} литературных источников')
        try:
            doi_cache = ROOT/'GlassDesignML_Upgrade'/'SciGlass_DOI_Cache_step409'/'source_doi_cache_409.json'
            enriched, doi_audit, doi_summary = enrich(LITERATURE,
                cache_path=doi_cache,out_dir=DOI_REPORT,input_fingerprint=FINGERPRINT,
                use_openalex=True,openalex_key=os.environ.get("OPENALEX_API_KEY",""),
                progress=doi_update, sleep_sec=.25,
                max_new_queries=120)
            if len(enriched)!=len(LITERATURE):
                raise AssertionError('Нарушена целостность литературных данных')
            LITERATURE = enriched
            doi_meter.empty()
        except Exception as err:
            doi_meter.empty()
            st.caption('DOI не удалось дополнить автоматически. Библиография и исходные измерения сохранены.')
            print('STEP409 DOI enrichment error:', repr(err))

st.divider()
st.header('Результаты подбора')
a, b, c = st.columns(3)
a.metric('Рассчитано составов', len(ALL))
b.metric('Отвечают заданным условиям', len(GOOD))
c.metric('Не прошли проверку', len(BAD))
st.write(f'Из {len(ALL)} рассчитанных составов **{len(GOOD)}** соответствуют заданным свойствам '
         'по двум моделям. Это кандидаты для лабораторных исследований, а не подтверждённые рецептуры.')

# Keep one short explanation. Reliability tiers are heuristic, not success probabilities.
st.markdown('**Оценка прогнозов:** **A** — меньше замечаний; '
            '**B** — нужны дополнительные измерения; '
            '**C** — особенно важно проверить Tg; '
            '**D** — для полноценной оценки не хватает данных.')
st.caption('Tg — температура стеклования; Td — дилатометрическое размягчение. '
           'Ни одна оценка не гарантирует успешное получение стекла.')
for note in st.session_state.get('ui402_report_notes', []):
    st.caption(note)

if not GOOD.empty:
    TOP, OTHERS = recommend_top_diverse(GOOD, SCIENCE, RESULT['criteria'], OXIDES, n=15,
                                          literature=LITERATURE)
    # Only the display selection is modified; original step391 shortlist remains unchanged.
    REPORT.mkdir(parents=True,exist_ok=True)
    TOP.to_csv(REPORT/'user_top15_step403.csv',index=False,encoding='utf-8-sig')
else:
    TOP, OTHERS = GOOD.copy(),GOOD.copy()


def grade_for(runid):
    if SCIENCE is None or SCIENCE.empty:
        return None
    c = SCIENCE.loc[SCIENCE.Run_Unique_ID.astype(str).eq(str(runid))]
    return c.iloc[0] if len(c) else None


def analogues_for(runid):
    if LITERATURE is None:
        return None
    return LITERATURE.loc[LITERATURE.Run_Unique_ID.astype(str).eq(str(runid))]


def show_candidate(row):
    runid = str(row.Run_Unique_ID)
    with st.container(border=True):
        st.markdown('**Состав № ' + runid + '**  ·  ' + str(row.Glass_family))
        st.write('**Компоненты:** ' + formula(row, OXIDES))
        a, b, c, d = st.columns(4)
        a.metric('Плотность, г/см³', fmt(row.Density_g_cm3, 3))
        b.metric('Преломление n', fmt(row.RefractiveIndex, 3))
        c.metric('Tg, °C', fmt(row.Tg_C, 1))
        d.metric('Td, °C', fmt(row.Tsoft_C, 1))

        info = grade_for(runid)
        if info is not None:
            st.write('**Оценка надёжности:** ' + grade_title(tier_letter(info.Screening_tier)))
        else:
            st.caption('Оценка по дополнительным проверкам пока недоступна.')
        with st.expander('Что проверить перед экспериментом'):
            checks = experimental_checks(info if info is not None else row)
            for item in checks:
                st.markdown('• ' + item)

        subset = analogues_for(runid)
        if subset is None:
            st.caption('Сведения об опубликованных измерениях ещё загружаются.')
        elif subset.empty:
            st.caption('Для этого состава не удалось установить проверяемый научный источник SciGlass.')
        else:
            # Все источники уже показаны возле соответствующих измерений ниже.
            # Отдельная карточка «Из научной литературы» здесь дублировала их.
            with st.expander('Экспериментальные стекла и публикации'):
                if not subset.Is_close_at_5mol.astype(bool).any():
                    st.caption('Близких измеренных стекол (до 5 мол.% отличия) не найдено. '
                               'Ниже — более удалённые опубликованные составы, '
                               'которые не подтверждают точность прогноза.')
                # Одна строка = один полный состав SciGlass, а не одно свойство.
                # Значения из разных работ не усредняются. [1], [2] связывают
                # каждое измерение с публикацией в той же строке.
                measured_glasses = group_measured_glasses(subset)
                st.caption(f'Найдено экспериментальных составов: {len(measured_glasses)}. '
                           'Прочерк — измерение для состава не найдено. '
                           'Если значений несколько, они перечислены отдельно; '
                           'номера в квадратных скобках указывают на публикации справа.')
                st.markdown(render_grouped_table(measured_glasses),
                            unsafe_allow_html=True)


if GOOD.empty:
    st.warning('Не найдено составов, соответствующих всем выбранным свойствам. Попробуй расширить диапазоны.')
else:
    st.subheader(f'{len(TOP)} вариантов, которые стоит проверить в первую очередь')
    st.caption('Сначала — составы ближе к серединам заданных диапазонов, '
               'с учётом химического разнообразия и замечаний научной проверки. '
               'Это не гарантия более успешного получения стекла.')
    for _,row in TOP.iterrows():
        show_candidate(row)
    if len(OTHERS):
        st.subheader('Другие подходящие варианты')
        if st.toggle(f'Показать остальные {len(OTHERS)} составов', value=False,
                     key='ui403_show_remaining'):
            for _,row in OTHERS.iterrows():
                show_candidate(row)
    st.download_button('Скачать основные 15 составов',
        TOP.to_csv(index=False).encode('utf-8-sig'),
        file_name='GlassDesignML_main15.csv', mime='text/csv')
    st.download_button('Скачать все подходящие по прогнозу составы',
        GOOD.to_csv(index=False).encode('utf-8-sig'),
        file_name='GlassDesignML_candidates.csv', mime='text/csv')


with st.expander(f'Составы, не прошедшие проверку ({len(BAD)})'):
    st.write('Эти варианты не соответствуют заданным условиям по обеим моделям. '
             'Это не означает, что такие стекла невозможно получить.')
    if not BAD.empty:
        simple = BAD[['Run_Unique_ID', 'Exclusion_reason']].copy().rename(
            columns={'Run_Unique_ID': 'Состав №', 'Exclusion_reason': 'Почему не предложен'})
        simple['Почему не предложен'] = simple['Почему не предложен'].astype(str).replace({
            'GlassNet: Td ниже Tg': 'Одна модель предсказывает размягчение ниже температуры стеклования',
            'GlassDesign: Td ниже Tg': 'Расчёт температур стеклования и размягчения несогласован',
        })
        st.dataframe(simple, hide_index=True, use_container_width=True)
        st.download_button('Скачать исключённые составы',
                           BAD.to_csv(index=False).encode('utf-8-sig'),
                           file_name='GlassDesignML_excluded.csv',mime='text/csv')

# Preserve full scientific provenance in export; developer-only metadata stays in archive.
with zipfile.ZipFile((buffer := io.BytesIO()), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for filename in ('manifest.json','all_new_recipes.csv','both_pass_recipes.csv',
                     'excluded_from_recommendations.csv','selected_representatives.csv'):
        source = Path(folder) / filename
        if source.is_file():
            archive.write(source, arcname='calculations/'+filename)
    for filename in ('scientific_screen_step399.csv','scientific_shortlist_step399.csv',
                     'science_meta_step399.json','sciglass_analogues_step399.csv',
                     'sciglass_sources_step399.csv','literature_coverage_step399.csv',
                     'literature_meta_step399.json', 'sciglass_literature_step403.csv',
                     'literature_meta_step403.json', 'user_top15_step403.csv'):
        source = REPORT / filename
        if source.is_file():
            archive.write(source, arcname='sources_and_checks/'+filename)
    for filename in ('sciglass_literature_doi_step409.csv','source_doi_audit_step409.csv',
                     'source_doi_review_step409.csv','doi_meta_step409.json'):
        source = DOI_REPORT / filename
        if source.is_file():
            archive.write(source,arcname='sources_and_checks/'+filename)
    archive.writestr('README.txt',
        'Расчётные составы и данные о близких экспериментальных аналогах. '
        'Научные оценки не являются гарантией стеклообразования. '
        'При отсутствии дополнительной проверки её отчёта в архиве нет.\n')
st.download_button('Скачать полный отчёт',buffer.getvalue(),
                   file_name='GlassDesignML_full_report.zip',mime='application/zip')
