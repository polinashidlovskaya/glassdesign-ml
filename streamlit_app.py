"""GlassDesign ML — usable scientist-facing Streamlit release candidate, step 380.

No model training; no automatic overwriting of app.py or original joblib files.
"""
from __future__ import annotations
import os
import tempfile
import copy
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from glass380_engine import (PROPS,load_catalog,discover_root,open_saved,run)
from glass380_science import check_all

# Model files are large. Load each frozen model once per web process, then
# clone only counters on every run. Do not mutate shared model parameters.
@st.cache_resource(show_spinner=False)
def _cached_gd(root_text):
    from glass_benchmark_v21 import FrozenPredictor
    root=Path(root_text)
    features,oxides,_=load_catalog(root)
    return FrozenPredictor(root,features,oxides)

@st.cache_resource(show_spinner=False)
def _cached_gn(oxide_names):
    from GlassDesignML_step364_dual_feedback import FrozenGlassNet
    return FrozenGlassNet(list(oxide_names))

@st.cache_resource(show_spinner=False)
def _search_lock():
    return threading.RLock()

def make_gd(root, features, oxides):
    predictor=copy.copy(_cached_gd(str(root)))
    predictor.rows=0
    predictor.calls=0
    return predictor

def make_gn(oxides):
    predictor=copy.copy(_cached_gn(tuple(oxides)))
    predictor.rows=0
    predictor.calls=0
    return predictor


class _ProgressGlassNet:
    """Only reports genuine completed GlassNet evaluations; GA is unchanged."""
    def __init__(self, inner, meter, total):
        self.inner = inner
        self.meter = meter
        self.total = max(int(total), 1)

    @property
    def rows(self):
        return self.inner.rows

    @property
    def calls(self):
        return self.inner.calls

    def __call__(self, frame):
        output = self.inner(frame)
        checked = int(self.inner.rows)
        pct = min(90, 12 + round(78 * min(checked, self.total) / self.total))
        self.meter.progress(
            pct, text=f'Проверено {checked} из {self.total} составов (GlassDesign ML + GlassNet)'
        )
        return output

st.set_page_config(page_title='GlassDesign ML',page_icon='🧪',layout='wide',
                   initial_sidebar_state='expanded')
LABELS={'Density_g_cm3':('Плотность, г/см³','3.2','5.3'),
        'RefractiveIndex':('Показатель преломления','1.62','1.90'),
        'Tg_C':('Температура стеклования Tg, °C','400','650'),
        'Tsoft_C':('Температура размягчения Tsoft, °C','480','795')}

def parse(x):
    s=str(x).strip().replace(',','.')
    if not s:return None
    n=float(s)
    if not np.isfinite(n):raise ValueError('Невозможно использовать бесконечность или NaN')
    return n

def formatted(x,dec=2):
    try:return f'{float(x):.{dec}f}'.replace('.',',') if pd.notna(x) else '—'
    except (TypeError,ValueError):return '—'

def composition(row,oxides):
    return ' · '.join(f'{ox} {formatted(row[ox],2)}%' for ox in oxides
                    if ox in row and pd.notna(row[ox]) and float(row[ox])>=.05)

def stat(value):
    if value is True or str(value).lower()=='true':return 'Найден близкий аналог'
    if value is False or str(value).lower()=='false':return 'Не найден в пределах 5 мол.%'
    return 'Не проверено'

st.title('🧪 GlassDesign ML')
st.markdown('**Обратное проектирование составов оксидных стёкол**')
st.caption('Предлагаемые составы и свойства получены расчётными моделями. '
           'Прогноз не подтверждает стеклообразование и технологические свойства экспериментально.')
ROOT=discover_root()
if ROOT is None:
    st.error('Не найдена конфигурация приложения. Обратитесь к разработчику.')
    st.stop()
try:
    _,oxides,raw_catalog=load_catalog(ROOT)
    manifest=__import__('json').loads((ROOT/'GlassDesignML/model_manifest.json').read_text(encoding='utf-8'))
    missing=[x['filename'] for x in manifest.values()
             if not (ROOT/'GlassDesignML'/'models'/x['filename']).is_file()]
    if missing:
        st.error('Для расчётов не загружены обученные модели: '+', '.join(missing))
        st.stop()
except Exception as e:
    st.error('Не удалось прочитать химический каталог: '+str(e))
    st.stop()

with st.sidebar:
    st.header('Подбор состава')
    st.subheader('1. Химическая система')
    mode_ru=st.radio('Как задать состав?',
                     ('Пусть программа подберёт систему','Выбрать готовые семейства',
                      'Собственный набор оксидов'))
    mode={'Пусть программа подберёт систему':'auto',
          'Выбрать готовые семейства':'families',
          'Собственный набор оксидов':'manual'}[mode_ru]
    selected_names=[];manual_bounds={};excluded=[]
    if mode=='families':
        selected_names=st.multiselect('Семейства',sorted(raw_catalog),
                     default=[x for x in ['Боратные','Боросиликатные','Германатные']
                              if x in raw_catalog])
    elif mode=='manual':
        chosen=st.multiselect('Оксиды',oxides,
                 default=[x for x in ('SiO2','B2O3','ZnO') if x in oxides])
        defaults={'SiO2':(40.,75.),'B2O3':(10.,45.),'ZnO':(0.,40.)}
        for oxide in chosen:
            a,b=st.columns(2)
            lo,hi=defaults.get(oxide,(0.,100.))
            lower=a.number_input(oxide+' от, мол.%',0.,100.,float(lo),step=1.,key='l'+oxide)
            upper=b.number_input(oxide+' до, мол.%',0.,100.,float(hi),step=1.,key='u'+oxide)
            manual_bounds[oxide]=(lower,upper)
    else:
        excluded=st.multiselect('Исключить нежелательные оксиды (необязательно)',oxides)
    st.subheader('2. Требуемые свойства')
    chosen_props=st.multiselect('Свойства',list(PROPS),default=list(PROPS),
                                format_func=lambda key: LABELS[key][0])
    with st.form('input',clear_on_submit=False):
        raw_bounds={}
        for key in chosen_props:
            label,default_lo,default_hi=LABELS[key]
            st.markdown('**'+label+'**')
            a,b=st.columns(2)
            low=a.text_input('От',value=default_lo,key='range_lo_'+key)
            high=b.text_input('До',value=default_hi,key='range_hi_'+key)
            raw_bounds[key]=(low,high)
        st.caption('Оставь одну из границ пустой, если она не нужна.')
        submitted=st.form_submit_button('Сгенерировать составы',type='primary',
                                        use_container_width=True)

if submitted:
    try:
        criteria={p:(parse(v[0]),parse(v[1])) for p,v in raw_bounds.items()}
        meter=st.progress(0, text='Проверяем параметры поиска…')
        # Previously tested step 381 progress protocol; inputs resolve *before* model initialization.
        from glass380_engine import load_backends, select_families
        backend=load_backends(Path(__file__).resolve().parent)
        feat,all_oxides,catalog=load_catalog(ROOT)
        valid,_=backend['glass_benchmark_v21'].wide_family_filter(catalog,all_oxides,max_components=12)
        available=select_families(valid,mode,selected_names,manual_bounds,excluded,all_oxides)
        total=8*len(available)+120
        def tracked_gd(root,features,oxides):
            meter.progress(2,text='Открываем четыре обученные модели GlassDesign ML…')
            predictor=make_gd(root,features,oxides)
            meter.progress(6,text='Модели GlassDesign ML загружены')
            return predictor
        def tracked_gn(oxides):
            meter.progress(7,text='Подготавливаем GlassNet (первый запуск может быть медленнее)…')
            predictor=make_gn(oxides)
            meter.progress(12,text=f'Обе модели готовы. Проверено 0 из {total} составов')
            return _ProgressGlassNet(predictor,meter,total)
        session_dir=Path(tempfile.mkdtemp(prefix='glassdesign_public_'))
        with _search_lock():
            result=run(criteria,root=ROOT,mode=mode,selected_names=selected_names,
                       manual_bounds=manual_bounds,excluded_oxides=excluded,seed=71,
                       out_dir=session_dir/'result',
                       gd_factory=tracked_gd,gn_factory=tracked_gn)
        meter.progress(100,text='Готово! Прогнозы сохранены, результаты доступны ниже.')
        st.session_state['glass380_folder']=str(result['folder'])
        st.session_state.pop('glass380_science',None)
        st.session_state.pop('glass380_science_issues',None)
        st.success('Подбор завершён. Результаты можно скачать в CSV.')
    except Exception as exc:
        st.error('Не удалось выполнить поиск: '+str(exc))

folder=st.session_state.get('glass380_folder')
if folder:
    try:result=open_saved(folder)
    except Exception as exc:
        st.error('Не удалось открыть сохранённые результаты: '+str(exc))
        st.stop()
    tables=result['tables']; summary=result['summary']; allfound=tables['both_pass_recipes']
    picked=tables['selected_representatives']
    st.divider();st.header('Результаты проектирования')
    a,b,c,d=st.columns(4)
    a.metric('Проверено новых составов',summary['new'])
    b.metric('Подходят по обеим моделям',summary['both_pass'])
    c.metric('Подходящих семейств',summary['successful_families'])
    d.metric('Отобрано представителей',summary['selected_representatives'])
    st.caption('Подходят по GlassDesign: '+str(summary['gd_pass'])+
               ' · по GlassNet: '+str(summary['glassnet_pass'])+
               ' · общий пилот: '+str(summary['pilot'])+
               ' · ID новых составов: '+str(summary['unique_id_range'][0])+
               '–'+str(summary['unique_id_range'][1]))
    criteria=result['manifest']['criteria']
    text='; '.join(f"{LABELS[p][0]}: {formatted(lo) if lo is not None else '—'} — "
                   f"{formatted(hi) if hi is not None else '—'}" for p,(lo,hi) in criteria.items())
    st.caption('Заданные требования: '+text+'. Во всех прогнозах проверяется Tsoft ≥ Tg.')
    science_targets=allfound if not allfound.empty else tables['nearest_30'].head(7)
    science=st.session_state.get('glass380_science')
    issues=st.session_state.get('glass380_science_issues',[])
    st.caption('Для быстрого подбора составов VITRIFY и поиск экспериментальных аналогов '
               'запускаются отдельно. Эти оценки не подтверждают свойства стекла экспериментально.')
    if st.button('Проверить стеклообразование (VITRIFY) и найти экспериментальные аналоги'):
        with st.spinner('Проверяем составы с помощью VITRIFY и ищем аналоги…'):
            science,issues=check_all(result,ROOT,oxides,force=True,candidates=science_targets)
            st.session_state['glass380_science']=science
            st.session_state['glass380_science_issues']=issues
    if science is not None:
        checked_vitrify=int(pd.to_numeric(science['VITRIFY_score'],errors='coerce').notna().sum())
        st.caption(f'VITRIFY: расчёт выполнен для {checked_vitrify} из {len(science)} составов.')
        if issues:
            with st.expander('Ограничения дополнительной проверки'):
                for issue in issues: st.write(str(issue))
        st.caption('Расстояние до экспериментального состава — половина суммы абсолютных '
                   'различий концентраций оксидов (мол.%). Это не ошибка прогноза свойств. '
                   'Порог 5 мол.% по всем выбранным свойствам проверяется отдельно.')
    if not allfound.empty:
        st.subheader('Отобранные составы')
        cols=['Run_Unique_ID','Glass_family']+list(PROPS)
        st.dataframe(picked[cols],hide_index=True,use_container_width=True)
        st.caption('Отбор проводится по близости к серединам интервалов и химическому разнообразию. '
                   'Это не рейтинг вероятности успешного синтеза.')
        def show_recipe(r):
            with st.container(border=True):
                st.markdown('**№ '+str(r.Run_Unique_ID)+' · '+str(r.Glass_family)+'**')
                st.write('**Состав, мол.%:** '+composition(r,oxides))
                ca,cb,cc,cd=st.columns(4)
                ca.metric('ρ, г/см³',formatted(r.Density_g_cm3,3))
                cb.metric('n',formatted(r.RefractiveIndex,4))
                cc.metric('Tg, °C',formatted(r.Tg_C,1))
                cd.metric('Tsoft, °C',formatted(r.Tsoft_C,1))
                details=['Tsoft ≥ Tg: '+('да' if r.Tsoft_C>=r.Tg_C else 'нет')]
                if science is not None:
                    found=science.loc[science.Run_Unique_ID==r.Run_Unique_ID]
                    if not found.empty:
                        evidence=found.iloc[0]
                        score=pd.to_numeric(evidence.get('VITRIFY_score'),errors='coerce')
                        if pd.notna(score):
                            details.append('VITRIFY: '+formatted(score,3)+
                                           (' (прогноз стеклообразования ≥ 0,5)' if score>=.5
                                            else ' (прогноз стеклообразования < 0,5)'))
                        nearest=pd.to_numeric(
                            evidence.get('Nearest_any_halfL1_mol'),errors='coerce')
                        source=evidence.get('Nearest_any_source')
                        if pd.notna(nearest):
                            label='Ближайший экспериментальный состав: '+formatted(nearest,2)+' мол.%'
                            if pd.notna(source):
                                label+=' (база: '+str(source)+')'
                            details.append(label)
                        all_dist=pd.to_numeric(
                            evidence.get('Nearest_max_halfL1_mol'),errors='coerce')
                        if pd.notna(all_dist):
                            details.append('Максимальное из минимальных расстояний по '
                                           'выбранным свойствам: '+formatted(all_dist,2)+' мол.%')
                st.caption(' · '.join(details))
        for _,r in picked.head(10).iterrows():
            show_recipe(r)
        if len(picked)>10:
            with st.expander('Показать ещё '+str(len(picked)-10)+' отобранных составов'):
                for _,r in picked.iloc[10:].iterrows():
                    show_recipe(r)
        rest=allfound.loc[~allfound.Run_Unique_ID.isin(picked.Run_Unique_ID)]
        if len(rest) and st.checkbox('Показать карточки остальных подходящих составов ('+str(len(rest))+')'):
            for _,r in rest.iterrows(): show_recipe(r)
        with st.expander('Все '+str(len(allfound))+' подходящих составов'):
            visible_cols=['Run_Unique_ID','Glass_family','Density_g_cm3','RefractiveIndex',
                          'Tg_C','Tsoft_C','GN_Density_g_cm3','GN_RefractiveIndex',
                          'GN_Tg_C','GN_Tsoft_C']
            full_table=allfound.copy()
            if science is not None:
                full_table=full_table.merge(science.drop(columns=['Candidate_ID']),on='Run_Unique_ID',
                                            how='left',validate='one_to_one')
                visible_cols+=['Near_analogue_within_5_mol',
                               'Nearest_any_halfL1_mol','Nearest_any_source',
                               'Nearest_max_halfL1_mol','VITRIFY_score','Tsoft_ge_Tg']
            st.dataframe(full_table[visible_cols],hide_index=True,use_container_width=True)
        download=allfound.copy()
        if science is not None:
            download=download.merge(science.drop(columns=['Candidate_ID']),on='Run_Unique_ID',
                                    how='left',validate='one_to_one')
        st.download_button('Скачать все '+str(len(download))+' подходящих составов (CSV)',
                           download.to_csv(index=False).encode('utf-8-sig'),
                           file_name='GlassDesignML_all_feasible_381.csv',mime='text/csv')
    else:
        st.warning('Составов, удовлетворяющих всем условиям по обоим прогнозам, не найдено. '
                   'Ниже ближайшие кандидаты — они не соответствуют всем требованиям.')
        near=tables['nearest_30'].head(10).copy()
        cols=['Run_Unique_ID','Glass_family','Max_normalized_violation']+list(PROPS)
        if science is not None and not science.empty:
            near=near.merge(science.drop(columns=['Candidate_ID']),on='Run_Unique_ID',how='left',validate='one_to_one')
            cols+=['Near_analogue_within_5_mol','Nearest_max_halfL1_mol','VITRIFY_score','Tsoft_ge_Tg']
        st.dataframe(near[cols],hide_index=True,use_container_width=True)
        with st.expander('Карточки семи ближайших вариантов (не соответствуют всем требованиям)'):
            for _,r in near.iterrows(): show_recipe(r)
        st.download_button('Скачать ближайшие варианты (CSV)',
                           tables['nearest_30'].to_csv(index=False).encode('utf-8-sig'),
                           file_name='GlassDesignML_nearest_381.csv',mime='text/csv')
    with st.expander('Подробности расчёта'):
        st.write('Выбранные химические семейства: '+', '.join(summary['selected_families']))
        st.caption('Результаты сохраняются временно и доступны для скачивания через браузер.')
        st.caption('Оценка двух моделей используется прямо в оптимизации; их согласие не является независимой валидацией. '
                   'Фильтрация не доказывает стеклообразование, фазовую стабильность или технологическую пригодность.')
        if issues:st.write('Дополнительные проверки: '+('; '.join(issues)))
else:
    st.info('Настрой параметры в панели слева и нажми «Сгенерировать составы» .')