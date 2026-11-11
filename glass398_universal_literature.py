"""GlassDesign ML step 398: property-specific universal SciGlass literature lookup.

Research aid, NOT independent validation. No changes to frozen prediction models.
Requires GlassPy for live SciGlass loading and requests for archive/Crossref download.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import re
import time
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STEP398_VERSION = "398.1"
PROPERTIES = {
    'Density293K': ('Плотность', 'г/см³', 'Density_g_cm3'),
    'RefractiveIndex': ('Показатель преломления', 'безразм.', 'RefractiveIndex'),
    'Tg': ('Температура стеклования Tg', '°C', 'Tg_C'),
    'TdilatometricSoftening': ('Дилатометрическое размягчение Td', '°C', 'TdilatometricSoftening_C'),
}
ARCHIVE_URL = 'https://raw.githubusercontent.com/drcassar/SciGlass/master/sciglass/select/'
OXIDE_PATTERN = re.compile(r'^(?:(?:[A-Z][a-z]?\d*)+)O\d*$')


def clean_text(v):
    if v is None or pd.isna(v):
        return ''
    s = str(v).strip()
    return '' if s.casefold() in ('nan','nat','none','<na>') else s


def canonical_int(v):
    s = clean_text(v)
    s = re.sub(r'\.0+$', '', s)
    return s if s.isdigit() else ''


def canon_year(v):
    s = canonical_int(v)
    return s if len(s) == 4 else ''


def surname(v):
    s = unicodedata.normalize('NFKD', clean_text(v)).casefold()
    s = re.sub(r'["\'`’]', '', s)
    s = re.sub(r'[^a-zа-яё0-9 ]', ' ', s)
    return s.split()[0] if s.split() else ''


def primary_author_from_reference(v):
    s = clean_text(v)
    match = re.search(r"'([^']+)'", s)
    return match.group(1) if match else s.split(' and ')[0].split(',')[0]


def normalize_doi(v):
    s = clean_text(v)
    m = re.search(r'10\.\d{4,9}/[^\s<>";,|]+', s, re.I)
    return m.group(0).rstrip('.,);]') if m else ''


def norm_title(s):
    s = unicodedata.normalize('NFKD', clean_text(s)).casefold()
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9а-яё]+', ' ', s)).strip()


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_recipes(path):
    path = Path(path)
    df = pd.read_csv(path, encoding='utf-8-sig')
    if len(df) == 0:
        raise ValueError('Входная таблица пустая')
    if 'Candidate_ID' not in df:
        df.insert(0, 'Candidate_ID', np.arange(1, len(df)+1))
    if 'Input_Row' in df:
        raise ValueError('Колонка Input_Row зарезервирована для внутренних ID')
    df.insert(0, 'Input_Row', np.arange(1, len(df)+1))
    return df


def discover_components(recipes, sciglass_components):
    """All *SciGlass-recognized* compound columns + extra oxide formula columns from input.

    Do not silently drop user oxides absent from SciGlass: these count as zero in references.
    User input must be in mol%, not wt%.
    """
    glass_cols = {str(c) for c in sciglass_components}
    input_cols = []
    for col in recipes.columns:
        if col in glass_cols or OXIDE_PATTERN.fullmatch(str(col)):
            input_cols.append(col)
    if not input_cols:
        raise ValueError('Не обнаружены химические компоненты в таблице составов')
    inputs = recipes[input_cols].apply(pd.to_numeric, errors='coerce')
    if inputs.isna().any().any() or (inputs < -1e-6).any().any():
        raise ValueError('В столбцах компонентов найдены отрицательные или нечисловые концентрации')
    sums = inputs.sum(axis=1).to_numpy(dtype=float)
    if np.allclose(sums, 100., rtol=0, atol=0.06):
        factor = 1.
    elif np.allclose(sums, 1., rtol=0, atol=0.0006):
        factor = 100.
    else:
        bad = recipes.loc[~np.isclose(sums, 100, atol=.06, rtol=0), ['Input_Row','Candidate_ID']].head(5)
        raise ValueError('Суммы компонентов должны равняться 100 мол.% (или 1 как доля). ' +
                         f'Диапазон сумм={np.min(sums):.4f}…{np.max(sums):.4f}; строки: '+str(bad.to_dict('records')))
    all_cols = list(dict.fromkeys(list(sciglass_components) + [c for c in input_cols if c not in glass_cols]))
    full = inputs.mul(factor).reindex(columns=all_cols, fill_value=0).astype('float32')
    return all_cols, full, [c for c in input_cols if c not in glass_cols], factor


def make_measurement_arrays(compounds, properties, metadata, recipes):
    """Normalize SciGlass compositions once; preserve all components, no family prefilter."""
    if not compounds.index.equals(properties.index):
        raise ValueError('Индексы состава и свойств SciGlass расходятся')
    if compounds.index.has_duplicates:
        raise ValueError('SciGlass содержит повторяющиеся идентификаторы строк')
    numeric = compounds.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    orig_sums = numeric.sum(axis=1).to_numpy(dtype=float)
    positive = orig_sums[orig_sums > 0]
    if not len(positive):
        raise ValueError('Не удалось определить единицы SciGlass')
    med = float(np.median(positive))
    if .7 < med < 1.3:
        scale = 100.
        min_total, max_total = .95, 1.05
    elif 70. < med < 130.:
        scale = 1.
        min_total, max_total = 95., 105.
    else:
        raise ValueError(f'Нетипичная сумма компонентов SciGlass ({med:.3f}); единицы неизвестны')
    valid = ((orig_sums >= min_total) & (orig_sums <= max_total) &
             np.isfinite(orig_sums) & (numeric.to_numpy() >= -1e-7).all(axis=1))
    numeric = numeric.iloc[np.flatnonzero(valid)]
    sums = numeric.sum(axis=1).to_numpy(dtype='float32')
    normalized = numeric.mul((100. / sums).astype('float32'), axis=0)
    allcols, rdata, not_in_sciglass, input_factor = discover_components(recipes, numeric.columns)
    X = normalized.reindex(columns=allcols, fill_value=0).to_numpy(dtype='float32')
    R = rdata.to_numpy(dtype='float32')
    p = properties.reindex(normalized.index)
    Y = {k: pd.to_numeric(p[k], errors='coerce').to_numpy('float64') for k in PROPERTIES}
    meta = metadata.reindex(normalized.index)
    auth = (meta['Author'].fillna('').astype(str).tolist() if 'Author' in meta else ['']*len(X))
    years = (meta['Year'].fillna('').astype(str).tolist() if 'Year' in meta else ['']*len(X))
    sids = [canonical_int(v) for v in normalized.index]
    if len(set(sids)) != len(sids) or any(not s for s in sids):
        raise ValueError('Потеря/повтор SciGlass_ID после подготовки измерений')
    return {
        'X': X, 'R':R, 'Y':Y,'auth':auth,'years':years,'ids':sids,
        'oxide_columns':allcols,'recipe_compounds':rdata,
        'source_not_recognized':not_in_sciglass,
        'source_rows':len(X), 'source_rows_total':len(compounds),
        'sciglass_compound_columns':list(compounds.columns),
        'source_original_median_sum':med,'source_scale':scale,
        'recipe_scale':input_factor,
        'property_rows':{k:int(np.isfinite(v).sum()) for k,v in Y.items()},
    }



def save_measurement_index(arrays, path):
    """Persist a small *source-only* index for later, unrelated user compositions."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    native=arrays['sciglass_compound_columns']
    positions=[arrays['oxide_columns'].index(k) for k in native]
    compact_X=arrays['X'][:,positions]
    values={
        'index_version':np.asarray(STEP398_VERSION),
        'X':compact_X,'oxide_columns':np.asarray(native,dtype=str),
        'ids':np.asarray(arrays['ids'],dtype=str),
        'auth':np.asarray(arrays['auth'],dtype=str),
        'years':np.asarray(arrays['years'],dtype=str),
        'source_rows_total':np.asarray(arrays['source_rows_total']),
        'source_original_median_sum':np.asarray(arrays['source_original_median_sum']),
        'source_scale':np.asarray(arrays['source_scale']),
    }
    for key,y in arrays['Y'].items():
        values['Y_'+key]=y
    part=path.with_suffix('.tmp.npz')
    np.savez_compressed(part,**values)
    part.replace(path)
    return path


def load_measurement_index(path, recipes):
    """Load without GlassPy; append new user-only oxide columns as zero in reference compositions."""
    path=Path(path)
    with np.load(path,allow_pickle=False) as z:
        version=str(z['index_version'].item())
        if version!=STEP398_VERSION:
            raise ValueError(f'Индекс {version} устарел для библиографического модуля {STEP398_VERSION}')
        native=list(map(str,z['oxide_columns']))
        X=z['X'].astype('float32')
        allcols,rdata,absent,recipe_scale=discover_components(recipes,native)
        if allcols!=native:
            # New user-only compounds are zero in all archived SciGlass rows.
            X=np.pad(X,((0,0),(0,len(allcols)-len(native))),mode='constant')
        ids=list(map(str,z['ids']))
        auth=list(map(str,z['auth']))
        years=list(map(str,z['years']))
        y={p:z['Y_'+p].astype(float) for p in PROPERTIES}
        source_rows_total=int(z['source_rows_total'].item())
        med=float(z['source_original_median_sum'].item())
        scale=float(z['source_scale'].item())
    if len(ids)!=len(X) or any(len(v)!=len(X) for v in y.values()):
        raise ValueError('Индекс повреждён: длины записей не совпадают')
    return {'X':X,'R':rdata.to_numpy('float32'),'Y':y,'auth':auth,'years':years,'ids':ids,
            'oxide_columns':allcols,'sciglass_compound_columns':native,'recipe_compounds':rdata,
            'source_not_recognized':absent,'source_rows':len(X),'source_rows_total':source_rows_total,
            'source_original_median_sum':med,'source_scale':scale,'recipe_scale':recipe_scale,
            'property_rows':{p:int(np.isfinite(a).sum()) for p,a in y.items()}}

def composition_label(row, names, max_oxides=20):
    ordered = sorted(((str(k),float(v)) for k,v in zip(names,row) if float(v)>0.005),
                     key=lambda t:(-t[1],t[0]))
    text = '; '.join(f'{k} {v:.3f}' for k,v in ordered[:max_oxides])
    return text + (f'; … (+{len(ordered)-max_oxides} компонентов)' if len(ordered)>max_oxides else '')


def search_nearest(recipes, arrays, max_half_l1=5., top_k=3):
    """Exact full-composition half-L1 with rigorously safe prefilter on major oxide.

    Since compositions sum to 100%, half-L1 >= |difference in any one oxide|.
    Thus filtering rows with |major difference|<=radius cannot remove a true match.
    """
    if not (max_half_l1 > 0 and max_half_l1 <= 100):
        raise ValueError('Некорректный порог расстояния')
    if top_k < 1 or top_k > 50:
        raise ValueError('TOP_K должен быть от 1 до 50')
    X,R,Y = arrays['X'], arrays['R'], arrays['Y']
    names = arrays['oxide_columns']
    eligible = {key: np.flatnonzero(np.isfinite(y)) for key,y in Y.items()}
    allmatches=[]
    # Use positional pandas rows, not itertuples() attribute names: pandas may
    # rename namedtuple fields for input CSVs with duplicate/invalid columns.
    # Row identity is provided by load_recipes(), not a DataFrame index.
    if len(recipes) != len(R):
        raise ValueError(f'Число рецептур {len(recipes)} не совпадает с индексом кандидатов {len(R)}')
    for required_col in ('Input_Row', 'Candidate_ID'):
        if required_col not in recipes.columns:
            raise ValueError(f'Отсутствует служебная колонка {required_col}; используйте load_recipes(INPUT)')
    if recipes['Input_Row'].isna().any() or recipes['Input_Row'].duplicated().any():
        raise ValueError('Input_Row должен содержать уникальные непустые идентификаторы строк')
    for i in range(len(recipes)):
        recipe = recipes.iloc[i]
        rid = int(recipe['Input_Row'])
        cid = str(recipe['Candidate_ID'])
        runid = clean_text(recipe.get('Run_Unique_ID', ''))
        v = R[i]
        if not np.isclose(v.sum(),100.,atol=.065):
            raise AssertionError('Нарушена сумма концентраций состава')
        major = int(np.argmax(v))
        label = composition_label(v,names)
        for p,(ru,unit,predname) in PROPERTIES.items():
            candidate_indices=eligible[p]
            if not len(candidate_indices):
                continue
            broad=(np.abs(X[candidate_indices,major]-v[major])<=max_half_l1+1e-4)
            idx=candidate_indices[broad]
            if not len(idx):
                continue
            distances=np.abs(X[idx]-v).sum(axis=1,dtype=np.float64)*.5
            good=distances<=max_half_l1+1e-4
            if not good.any():
                continue
            idx=idx[good]
            distances=distances[good]
            # Stable sort by numerical distance; initial rows of SciGlass are deterministic.
            keep=np.argsort(distances,kind='stable')[:top_k]
            # The i-th composition in R corresponds to the i-th input row,
            # even if the user DataFrame has a nonconsecutive index.
            observed = pd.to_numeric(recipe.get(predname),errors='coerce') if predname in recipes else np.nan
            if p=='TdilatometricSoftening' and not np.isfinite(observed) and 'Tsoft_C' in recipes:
                observed = pd.to_numeric(recipe['Tsoft_C'],errors='coerce')
            for rank,j in enumerate(keep,1):
                ix=int(idx[j])
                value=float(Y[p][ix])
                if p in ('Tg','TdilatometricSoftening'):
                    value-=273.15
                sid=arrays['ids'][ix]
                allmatches.append({
                    'Input_Row':rid,'Candidate_ID':cid,'Run_Unique_ID':runid,
                    'Property':p,'Property_RU':ru,'Rank':rank,
                    'Distance_halfL1_mol':round(float(distances[j]),5),
                    'Measured_value':round(value,6),'Unit':unit,
                    'Model_prediction_value':float(observed) if np.isfinite(observed) else np.nan,
                    'SciGlass_ID':sid, 'KOD_from_ID':sid[:-8] if len(sid)>8 else '',
                    'SciGlass_author':arrays['auth'][ix],
                    'SciGlass_year':canon_year(arrays['years'][ix]),
                    'Candidate_composition_mol':label,
                    'Analogue_composition_mol':composition_label(X[ix],names),
                    'Analogue_relation':'Близкий экспериментальный состав; не независимая валидация ML',
                })
    fields=['Input_Row','Candidate_ID','Run_Unique_ID','Property','Property_RU','Rank',
            'Distance_halfL1_mol','Measured_value','Unit','Model_prediction_value',
            'SciGlass_ID','KOD_from_ID','SciGlass_author','SciGlass_year',
            'Candidate_composition_mol','Analogue_composition_mol','Analogue_relation']
    frame=pd.DataFrame(allmatches,columns=fields)
    if not frame.empty:
        frame.sort_values(['Input_Row','Property','Rank'],inplace=True,kind='stable')
        if frame.duplicated(['Input_Row','Property','Rank']).any():
            raise AssertionError('Дубли рангов')
    return frame


def _decode_archive_zip(blob, table):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        members=[m for m in z.namelist() if m.lower().endswith('.csv')]
        if not members:
            raise ValueError(f'{table}: нет CSV в архиве')
        raw=z.read(max(members, key=lambda m:z.getinfo(m).file_size))
    for enc in ('utf-8-sig','cp1251','latin1'):
        try:
            text=raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    try:sep=csv.Sniffer().sniff(text[:6000],delimiters=',;\t|').delimiter
    except csv.Error:sep=','
    result=pd.read_csv(io.StringIO(text),sep=sep,dtype=str,low_memory=False)
    result.columns=result.columns.str.replace('\ufeff','',regex=False).str.strip()
    return result


def load_archive(cache_dir, force_download=False):
    import requests
    cache_dir=Path(cache_dir)
    cache_dir.mkdir(parents=True,exist_ok=True)
    tables={}; info={}
    for table in ('Reference','Kod2Ref','ARHIV'):
        path=cache_dir/(table+'.csv.zip')
        if force_download or not path.is_file():
            url=ARCHIVE_URL+table+'.csv.zip'
            resp=requests.get(url,timeout=120,headers={'User-Agent':'GlassDesignML-UniversalLiterature/1.0'})
            resp.raise_for_status()
            # Check the ZIP is well formed before overwriting cached file.
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                if z.testzip() is not None:
                    raise ValueError('ZIP повреждён: '+table)
            part=path.with_suffix('.tmp')
            part.write_bytes(resp.content)
            part.replace(path)
        tables[table]=_decode_archive_zip(path.read_bytes(),table)
        info[table]={'sha256':hash_file(path),'rows':len(tables[table]),
                     'download_url':ARCHIVE_URL+table+'.csv.zip'}
        print(f'SciGlass {table}: {len(tables[table]):,} строк; sha256 {info[table]["sha256"][:12]}…')
    ref,rel,arc=(tables[k] for k in ('Reference','Kod2Ref','ARHIV'))
    required=[('Reference',ref,{'Refer_ID','Authors','YEAR','DOI'}),
              ('Kod2Ref',rel,{'Kod','RefID'}),
              ('ARHIV',arc,{'KOD','Refer_Id','AUTHOR','YEAR'})]
    for label,frame,cols in required:
        if not cols.issubset(frame.columns):
            raise ValueError(f'Неверная схема SciGlass {label}: {cols-set(frame.columns)}')
    return tables,info


def build_archive_lookup(tables):
    """Only relation KOD -> Kod2Ref.RefID -> Reference.Refer_ID is permissible."""
    ar=tables['ARHIV'].copy()
    rel=tables['Kod2Ref'].copy()
    ref=tables['Reference'].copy()
    ar['KEY']=ar['KOD'].map(canonical_int)
    rel['KEY']=rel['Kod'].map(canonical_int)
    rel['REFKEY']=rel['RefID'].map(canonical_int)
    ref['KEY']=ref['Refer_ID'].map(canonical_int)
    for label,frm in (('ARHIV',ar),('Reference',ref)):
        keys=frm.loc[frm.KEY.ne(''),'KEY']
        if keys.duplicated().any():
            raise ValueError(f'{label} содержит дубли первичных ключей')
    ar_map=ar.set_index('KEY').to_dict('index')
    ref_map=ref.set_index('KEY').to_dict('index')
    rel_map={k:set(v.REFKEY) for k,v in rel.groupby('KEY')}
    return ar_map,ref_map,rel_map


def check_source_link_for_id(sid, author, year, lookup, reviewed=None):
    """Verify record ID prefix, author/year, primary ARHIV ref, archive reference.

    A DOI from reviewed catalog is imported only if KOD and Primary_RefID match.
    """
    ar,refs,rel=lookup
    sid=canonical_int(sid)
    kod=sid[:-8] if len(sid)>8 else ''
    table=ar.get(kod,{})
    table_url=f'https://sciglass.uni-jena.de/database?table_id={kod}' if kod and table else ''
    pid=canonical_int(table.get('Refer_Id'))
    rr=refs.get(pid,{}) if pid and pid in rel.get(kod,set()) else {}
    a=clean_text(table.get('AUTHOR'))
    y=canon_year(table.get('YEAR'))
    refauth=primary_author_from_reference(rr.get('Authors')) if rr else ''
    refyear=canon_year(rr.get('YEAR')) if rr else ''
    metadata_author_ok=(bool(surname(author)) and surname(author)==surname(a))
    metadata_year_ok=bool(canon_year(year) and canon_year(year)==y)
    primary_author_ok=bool(surname(a) and surname(a)==surname(refauth))
    primary_year_ok=bool(y and y==refyear)
    confirmed=bool(table and pid and rr and metadata_author_ok and metadata_year_ok
                   and primary_author_ok and primary_year_ok)
    citation=''
    for col in ('FullDescr','Descr','ShortDescr'):
        if rr and clean_text(rr.get(col)):
            citation=clean_text(rr[col]);break
    title=clean_text(rr.get('Title')) if rr else ''
    doi=normalize_doi(rr.get('DOI')) if confirmed else ''
    url=('https://doi.org/'+doi) if doi else table_url
    confidence='DOI указан в оригинальной Reference SciGlass' if doi else 'DOI отсутствует в оригинальной Reference SciGlass'
    url_type='DOI' if doi else 'sciglass_bibliography'
    match_status='Библиография подтверждена по архиву SciGlass' if confirmed else 'Библиография не прошла полную проверку'
    review = reviewed.get(kod) if reviewed else None
    # Never reuse reviewed DOI if KOD shares a number with an unrelated Reference entry.
    if confirmed and review and canonical_int(review.get('Primary_RefID'))==pid and \
            canon_year(review.get('Main_Year'))==y and surname(review.get('Main_Author'))==surname(a):
        proposed_doi=normalize_doi(review.get('DOI_step397'))
        proposed_link=clean_text(review.get('Source_URL_step397'))
        if doi and proposed_doi and doi.casefold()!=proposed_doi.casefold():
            match_status='Конфликт DOI оригинального SciGlass и каталога 397 — проверить вручную'
            url_type='DOI_conflict'
            url=table_url
            doi=''
            confidence='Конфликт DOI SciGlass и каталога 397 — ручная проверка обязательна'
        else:
            if proposed_doi:
                doi=proposed_doi
            if doi:
                url='https://doi.org/'+doi
                url_type='DOI'
            elif proposed_link.startswith('https://'):
                url=proposed_link
                url_type=clean_text(review.get('Source_Type_step397')) or 'sciglass_bibliography'
            confidence=clean_text(review.get('DOI_Confidence_step397')) or confidence
    if not confirmed:
        # Do not turn unverified bibliography into direct asserted journal link.
        doi='';url=table_url;url_type='sciglass_table_unverified';confidence='ID/автор/год/RefID не полностью подтверждены'
    return {'SciGlass_ID':sid, 'KOD_verified':kod if table else '',
            'Primary_RefID':pid,'Primary_author':refauth,'Primary_year':refyear,
            'Source_bibliography':citation if confirmed else '',
            'Source_title':title if confirmed else '',
            'Source_DOI':doi,'Source_URL':url,
            'Source_link_type':url_type,'Source_DOI_confidence':confidence,
            'Source_status':match_status,'Source_archive_confirmed':confirmed,
            'SciGlass_table_URL':table_url,'Additional_references_count':max(0,len(rel.get(kod,set()))-1)}


def read_reviewed_catalog(path):
    path=Path(path)
    if not path.is_file():
        print('Нет каталога шага 397: DOI возьмём из SciGlass или Crossref; ссылки на таблицы останутся')
        return {}
    df=pd.read_csv(path,encoding='utf-8-sig',dtype=str).fillna('')
    if df['KOD_verified'].duplicated().any():
        raise ValueError('В каталоге шага 397 неоднозначный KOD')
    print('Каталог ранее проверенных ссылок:',len(df),'таблиц')
    return {canonical_int(r['KOD_verified']):r.to_dict() for _,r in df.iterrows()}


def attach_sources(matches, lookup, reviewed=None):
    if matches.empty:
        return matches.copy(),pd.DataFrame()
    unique=matches[['SciGlass_ID','SciGlass_author','SciGlass_year']].drop_duplicates('SciGlass_ID')
    links=[check_source_link_for_id(r.SciGlass_ID,r.SciGlass_author,r.SciGlass_year,lookup,reviewed)
           for r in unique.itertuples(index=False)]
    source_table=pd.DataFrame(links)
    joined=matches.merge(source_table,on='SciGlass_ID',validate='many_to_one',how='left')
    if len(joined)!=len(matches):
        raise AssertionError('Потеря/размножение строк после присоединения библиографии')
    return joined,source_table


def crossref_query_source(title,author,year,session,timeout=14):
    from difflib import SequenceMatcher
    s=norm_title(title)
    if not s or len(s)<15 or not surname(author) or not canon_year(year):
        return {'status':'Нет достаточной библиографии для поиска'}
    resp=session.get('https://api.crossref.org/works',params={'query.title':title,'rows':5,'select':'DOI,title,author,published,issued,created'},timeout=timeout)
    resp.raise_for_status()
    items=resp.json().get('message',{}).get('items',[])
    found=[]
    for item in items:
        mt=clean_text((item.get('title') or [''])[0])
        sim=SequenceMatcher(None,s,norm_title(mt)).ratio()
        a=(item.get('author') or [{}])[0].get('family','')
        match_a=surname(author)==surname(a)
        yy=''
        for field in ('published','issued','created'):
            dp=item.get(field,{}).get('date-parts',[])
            if dp and dp[0] and dp[0][0]:
                yy=str(dp[0][0]);break
        okyear=bool(yy.isdigit() and abs(int(yy)-int(year))<=1)
        found.append({'doi':normalize_doi(item.get('DOI')),'title':mt,'similarity':round(sim,4),
                      'author_ok':match_a,'year_ok':okyear,'year':yy})
    found.sort(key=lambda v:(v['author_ok'] and v['year_ok'],v['similarity']),reverse=True)
    if not found:return {'status':'Crossref не нашёл совпадений','candidates':[]}
    best=found[0]
    if best['similarity']>=.96 and best['author_ok'] and best['year_ok'] and best['doi']:
        # A very close match is provisional, never manually verified.
        return {'status':'Предварительное автоматическое соответствие Crossref','doi':best['doi'],
                'similarity':best['similarity'],'candidates':found}
    return {'status':'Crossref требует ручной проверки','candidates':found}


def enrich_with_crossref(sources,cache_json,max_requests=25,enable=True):
    """Optional, bounded public DOI lookups for new KOD only. Cached for repeat runs."""
    cache_json=Path(cache_json)
    try: cache=json.loads(cache_json.read_text(encoding='utf-8')) if cache_json.is_file() else {}
    except Exception:cache={}
    if sources.empty or not enable:
        return sources.copy(),0,0
    import requests
    df=sources.copy()
    unique=df.drop_duplicates('KOD_verified')
    candidates=unique[(unique.Source_archive_confirmed==True)&unique.Source_DOI.fillna('').eq('')&
                      unique.Source_title.fillna('').str.len().ge(15)&unique.Primary_year.fillna('').ne('')]
    sess=requests.Session()
    sess.headers.update({'User-Agent':'GlassDesignML-UniversalSources/398 (research; rate-limited)'})
    queried=0; errors=0
    for row in candidates.itertuples(index=False):
        kod=row.KOD_verified
        key=f'{kod}|{row.Primary_RefID}|{norm_title(row.Source_title)}'
        if key not in cache and queried>=max_requests:
            continue
        if key not in cache:
            try:
                cache[key]=crossref_query_source(row.Source_title,row.Primary_author,row.Primary_year,sess)
            except Exception as exc:
                cache[key]={'status':'Сетевая ошибка Crossref','error':str(exc)[:200]};errors+=1
            queried+=1
            time.sleep(.30)
        got=cache[key]
        if got.get('status')=='Предварительное автоматическое соответствие Crossref' and got.get('doi'):
            target=df.KOD_verified.eq(kod)&df.Primary_RefID.eq(row.Primary_RefID)&df.Source_DOI.fillna('').eq('')
            df.loc[target,'Source_DOI']=got['doi']
            df.loc[target,'Source_URL']='https://doi.org/'+got['doi']
            df.loc[target,'Source_link_type']='DOI'
            df.loc[target,'Source_DOI_confidence']='Автоматический Crossref 398; ручная проверка не выполнена'
        else:
            target=df.KOD_verified.eq(kod)&df.Primary_RefID.eq(row.Primary_RefID)&df.Source_DOI.fillna('').eq('')
            df.loc[target,'Crossref_398_status']=str(got.get('status',''))
    cache_json.parent.mkdir(parents=True,exist_ok=True)
    cache_json.write_text(json.dumps(cache,ensure_ascii=False,indent=2),encoding='utf-8')
    return df,queried,errors


def make_coverage(matches,recipes):
    rows=[]
    for p in PROPERTIES:
        part=matches.loc[matches.Property.eq(p)] if not matches.empty else pd.DataFrame()
        rows.append({'Property':p,'Recommended_or_input_candidates':len(recipes),
            'Candidates_with_measured_analogue':part.Input_Row.nunique() if not part.empty else 0,
            'Candidates_with_archive_verified_source':part.loc[part.Source_archive_confirmed.eq(True),'Input_Row'].nunique() if not part.empty else 0,
            'Candidates_with_DOI':part.loc[part.Source_DOI.fillna('').ne(''),'Input_Row'].nunique() if not part.empty else 0,
            'Candidates_with_link':part.loc[part.Source_URL.fillna('').ne(''),'Input_Row'].nunique() if not part.empty else 0,
            'Unique_SciGlass_ID':part.SciGlass_ID.nunique() if not part.empty else 0,
            'Unique_KOD':part.KOD_verified.nunique() if not part.empty else 0,
            'Analogue_rows':len(part)})
    return pd.DataFrame(rows)


def save_step398(recipes,matches,lookup,reviewed,archive_info,arrays,outdir,input_path,radius,top_k,crossref_cache=None,run_crossref=True):
    outdir=Path(outdir)
    outdir.mkdir(parents=True,exist_ok=False)
    linked,sources=attach_sources(matches,lookup,reviewed)
    queried=errors=0
    if not sources.empty:
        if crossref_cache is not None:
            sources,queried,errors=enrich_with_crossref(sources,crossref_cache,enable=run_crossref)
        update_cols=['SciGlass_ID','Source_DOI','Source_URL','Source_link_type','Source_DOI_confidence','Crossref_398_status']
        for col in update_cols:
            if col not in sources:
                sources[col]=''
        core=linked.drop(columns=[c for c in update_cols[1:] if c in linked.columns])
        linked=core.merge(sources[update_cols],how='left',on='SciGlass_ID',validate='many_to_one')
    cov=make_coverage(linked,recipes)
    if not linked.empty:
        assert len(linked)==len(matches),'После библиографии изменилось количество строк'
        assert linked.groupby(['Input_Row','Property']).size().le(top_k).all(), 'Превышено top_k'
        assert (linked.Distance_halfL1_mol<=radius+1e-3).all(),'Ближний аналог за пределами порога'
        assert linked[['Input_Row','Property','Rank']].duplicated().sum()==0
        assert (linked.loc[linked.Source_DOI.fillna('').ne(''),'Source_archive_confirmed']==True).all(), 'DOI присвоен неподтверждённой связи'
        assert (linked.loc[linked.Source_DOI.fillna('').ne(''),'Source_URL'].str.startswith('https://doi.org/')).all(), 'DOI URL должен быть DOI URL'
    files={
        'sciglass_analogues_with_sources_step398.csv':linked,
        'sciglass_source_catalog_used_step398.csv':sources,
        'coverage_step398.csv':cov,
        'input_candidates_step398.csv':recipes,
    }
    for name,df in files.items():
        df.to_csv(outdir/name,index=False,encoding='utf-8-sig')
    report={'step':398,'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'input_file':str(input_path),'input_sha256':hash_file(input_path),
            'input_candidates':len(recipes),'analogue_rows':len(linked),
            'unique_sciglass_id':int(linked.SciGlass_ID.nunique()) if not linked.empty else 0,
            'unique_source_kod':int(linked.KOD_verified.nunique()) if not linked.empty else 0,
            'unique_doi':int(sources.loc[sources.Source_DOI.fillna('').ne(''),'Source_DOI'].nunique()) if not sources.empty else 0,
            'max_distance_halfL1_mol':radius,'top_k':top_k,
            'sciglass_rows_total':arrays['source_rows_total'],'sciglass_rows_valid_compositions':arrays['source_rows'],
            'sciglass_property_measurement_rows':arrays['property_rows'],
            'oxide_columns':len(arrays['oxide_columns']),
            'recipe_oxides_absent_from_sciglass':arrays['source_not_recognized'],
            'archive_files':archive_info,'crossref_network_requests':queried,'crossref_errors':errors,
            'models_changed':False,'generator_changed':False,'interface_changed':False,
            'limitations':['Измеренный аналог не доказывает, что кандидат будет стеклообразным',
                           'Не является независимой оценкой точности GlassDesign ML или GlassNet',
                           'Отсутствие близких экспериментальных аналогов не означает отсутствие литературы',
                           'Кандидат и SciGlass должны иметь молярные составы (не массовые проценты)',
                           'DOI Crossref являются предварительными, если не проверены вручную',
                           'Ссылка на страницу SciGlass по KOD не проверена на живом сервере',
                           'Для состава с одинаковыми компонентами и погрешностью источников одна статья может содержать много записей']}
    (outdir/'manifest_step398.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report,cov,linked,sources
