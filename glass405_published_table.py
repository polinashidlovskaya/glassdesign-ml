"""GlassDesign ML 405 — literature table grouped by measured glass composition.

Presentation-only transformation. Do NOT merge measured values blindly:
- one output row per exact oxide composition as recorded in SciGlass (3 decimal mol.%);
- measurements from different SciGlass IDs or sources are not averaged;
- every displayed value gets reference number(s) pointing to its source;
- no properties are inferred from ML predictions or other compositions.
"""
from __future__ import annotations

from collections import OrderedDict
from html import escape
from urllib.parse import urlparse
import math
import re
import pandas as pd
from glass408_source_access import resolve_source

PROPERTY_FIELDS = OrderedDict([
    ('Density293K', ('ρ, г/см³', 3)),
    ('RefractiveIndex', ('n', 3)),
    ('Tg', ('Tg, °C', 1)),
    ('TdilatometricSoftening', ('Td, °C', 1)),
])
COMPONENT = re.compile(r'^([^\s;]+)\s+([+-]?(?:\d+(?:[.,]\d*)?|\.\d+))$')
SUBSCRIPTS = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')


def _present(v):
    if v is None or v is pd.NA:
        return False
    try:
        return bool(pd.notna(v)) and str(v).strip() not in ('', 'nan', 'None')
    except (TypeError, ValueError):
        return False


def parse_composition(raw):
    """Canonical formula components; do not convert wt% to mol% or round coarse.

    The index stores numbers to 3 decimal places; group only exact recorded
    compositions, apart from ordering of oxides. Unparseable rows stay separate.
    """
    if not _present(raw):
        return None
    parts=[]
    for chunk in str(raw).split(';'):
        chunk=chunk.strip()
        if not chunk:
            continue
        match=COMPONENT.fullmatch(chunk)
        if not match:
            return None
        name,num=match.groups()
        value=float(num.replace(',','.'))
        if not math.isfinite(value) or value < 0 or value > 100:
            return None
        if value >= 0.0005:
            parts.append((name,round(value,3)))
    if not parts or len({k for k,v in parts})!=len(parts):
        return None
    return tuple(sorted(parts))


def formatted_composition(parts):
    if parts is None:
        return 'Состав не указан в SciGlass'
    # Preserve original oxide appearance order in the display if possible.
    chunks=[]
    for oxide,num in parts:
        value=f'{num:.3f}'.rstrip('0').rstrip('.').replace('.',',')
        chunks.append(f'{oxide.translate(SUBSCRIPTS)} {value}%')
    return ' · '.join(chunks)


def _first(row,fields):
    for field in fields:
        if _present(row.get(field)):
            return str(row.get(field)).strip()
    return ''


def _source_key(row):
    # Original SciGlass joins, never DOI/title guesses. Each table has one main source.
    kod=_first(row,('KOD_verified',))
    rid=_first(row,('Primary_RefID',))
    link=_first(row,('Source_URL','SciGlass_table_URL'))
    sid=_first(row,('SciGlass_ID',))
    return (kod,rid) if kod and rid else ('source',link,sid)


def _source_name(row):
    title=_first(row,('Source_title','Source_bibliography'))
    if title:
        return title
    kind=_first(row,('Source_kind',))
    author=_first(row,('Primary_author','SciGlass_author'))
    year=_first(row,('Primary_year','SciGlass_year'))
    return ' · '.join(x for x in (kind or 'Источник SciGlass', author,year) if x)


def safe_url(value):
    if not _present(value):
        return None
    s=str(value).strip()
    try:
        u=urlparse(s)
        return s if u.scheme=='https' and u.netloc and not u.username and not u.password else None
    except ValueError:
        return None


def _number(value,dp):
    return f'{float(value):.{dp}f}'.replace('.',',')


def group_measured_glasses(analogues):
    """Returns grouped records, preserving source linkage per single measurement.

    Every entry has `composition`, `distance`, `properties`, `sources`. A
    property maps to a list of (value, source_index) *without averaging*.
    Each source has its original URL and publication. One chemical formula may
    have several experimental results and several publications.
    """
    if analogues is None or len(analogues)==0:
        return []
    expected={'Property','Measured_value','Analogue_composition_mol','SciGlass_ID',
              'Distance_halfL1_mol'}
    absent=expected.difference(analogues.columns)
    if absent:
        raise ValueError(f'Отсутствуют поля SciGlass: {sorted(absent)}')
    result=OrderedDict()
    for _,record in analogues.iterrows():
        prop=record.get('Property')
        if prop not in PROPERTY_FIELDS:
            continue
        try:
            measured=float(record.Measured_value)
            distance=float(record.Distance_halfL1_mol)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(measured) or not math.isfinite(distance):
            continue
        composition=parse_composition(record.Analogue_composition_mol)
        # Never combine unknown compositions based on their inferred system.
        key=('composition',composition) if composition is not None else ('unknown_id',str(record.SciGlass_ID))
        if key not in result:
            result[key]={
                'composition':formatted_composition(composition),
                'distance':distance,
                'properties':{p:[] for p in PROPERTY_FIELDS},
                'sources':[],
                '_source_positions':{},
                '_used_measurements':set(),
            }
        group=result[key]
        group['distance']=min(group['distance'],distance)
        source=_source_key(record)
        if source not in group['_source_positions']:
            group['_source_positions'][source]=len(group['sources'])+1
            group['sources'].append({
                'number':len(group['sources'])+1,
                'label':_source_name(record),
                'access':resolve_source(record),
                'kind':_first(record,('Source_kind',)),
                'doi_confidence':_first(record,('Source_DOI_confidence',)),
            })
        number=group['_source_positions'][source]
        # Preserve two published values if different, but remove exact duplicate
        # rows from the same SciGlass ID/property/source/value.
        dedupe=(str(record.SciGlass_ID),str(prop),round(measured,6),source)
        if dedupe not in group['_used_measurements']:
            group['_used_measurements'].add(dedupe)
            group['properties'][prop].append((measured,number))
    output=[]
    for group in result.values():
        group.pop('_source_positions')
        group.pop('_used_measurements')
        for prop,values in group['properties'].items():
            # Some separate original records repeat the same measurement and
            # primary paper; one display value is sufficient for these.
            group['properties'][prop]=list(dict.fromkeys((round(v,6),ref) for v,ref in values))
        output.append(group)
    return sorted(output,key=lambda g:(g['distance'],-sum(bool(x) for x in g['properties'].values()),g['composition']))


def render_grouped_table(groups):
    """Safe HTML, one *composition* per row and multiple clickable sources."""
    if not groups:
        return '<p>Экспериментальные записи для отображения отсутствуют.</p>'
    styles='''<style>
.gd405-scroll{overflow-x:auto;width:100%;margin:.35rem 0 1rem}
.gd405-table{border-collapse:collapse;width:100%;min-width:850px;font-size:.93rem}
.gd405-table th,.gd405-table td{padding:10px 12px;border-bottom:1px solid rgba(128,128,128,.26);vertical-align:top;text-align:left}
.gd405-table th{font-weight:600;background:rgba(130,130,130,.07);white-space:nowrap}
.gd405-table .gd405-formula{min-width:235px;font-weight:500;line-height:1.5}
.gd405-table .gd405-prop{min-width:78px;white-space:normal;line-height:1.6}
.gd405-table .gd405-refs{min-width:270px;line-height:1.4}
.gd405-table .gd405-ref{margin-bottom:7px}
.gd405-table a{color:inherit;text-decoration:underline;text-underline-offset:2px}
.gd405-table .gd405-empty{color:#858585}
.gd405-table small{font-size:.78rem;color:#737373}
.gd405-table .gd405-unavailable{color:#76695a}
</style>'''
    headers=['Состав экспериментального стекла, мол.%','ρ, г/см³','n','Tg, °C','Td, °C','Отличие, мол.%','Публикации']
    html=[styles,'<div class="gd405-scroll"><table class="gd405-table">','<thead><tr>']
    for h in headers:html.append(f'<th scope="col">{escape(h)}</th>')
    html.append('</tr></thead><tbody>')
    for group in groups:
        html.append('<tr>')
        html.append('<td class="gd405-formula">'+escape(group['composition'])+'</td>')
        for prop,(label,dp) in PROPERTY_FIELDS.items():
            values=group['properties'][prop]
            if not values:
                html.append('<td class="gd405-empty">—</td>')
                continue
            contents=[]
            for value,ref in values:
                contents.append(f'{escape(_number(value,dp))} <small>[{int(ref)}]</small>')
            html.append('<td class="gd405-prop">'+'; '.join(contents)+'</td>')
        html.append('<td>'+escape(_number(group['distance'],2))+'</td>')
        sources=[]
        for s in group['sources']:
            number=s['number']
            title=s['label']
            shorter=title[:80]+'…' if len(title)>80 else title
            access=s['access']
            url=access.get('url','')
            search=access.get('search_url','')
            if url and access.get('direct',False):
                link=(f'<a href="{escape(url,quote=True)}" target="_blank" rel="noopener noreferrer">'
                      f'{escape(access["action"])}</a>')
                val=f'{number}. {escape(shorter)}<br><small>{link}</small>'
                if 'автоматически' in str(s.get('doi_confidence','')).casefold():
                    val+=' <small>· DOI найден автоматически</small>'
                if search:
                    val+=(f' <small>· <a href="{escape(search,quote=True)}" target="_blank" '
                          f'rel="noopener noreferrer">Поиск другого экземпляра</a></small>')
            else:
                # Source is bibliographically identified in original SciGlass
                # but no verified accessible destination; do not pretend search
                # is a link directly to the source.
                val=(f'{number}. {escape(shorter)}<br>'
                     f'<small class="gd405-unavailable">Библиография есть; прямая ссылка не установлена.</small>')
                if search:
                    val+=(f' <small>· <a href="{escape(search,quote=True)}" target="_blank" '
                          f'rel="noopener noreferrer">Искать в интернете</a></small>')
            sources.append('<div class="gd405-ref">'+val+'</div>')
        html.append('<td class="gd405-refs">'+''.join(sources)+'</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    return ''.join(html)
