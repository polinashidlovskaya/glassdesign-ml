"""GlassDesign ML 407 — public article links, independent of SciGlass Next login.

Presentation-only source routing: DOI / directly published URL, otherwise
explicitly marked bibliography-based search. Never invent DOI or full-text access.
"""
from __future__ import annotations
import re
from urllib.parse import quote_plus, urlparse, unquote
import pandas as pd

_DOI = re.compile(r'^10\.\d{4,9}/[^\s<>]+$', re.I)
_PATENT = re.compile(r'\b([A-Z]{2})\s+Patent\s+(?:No\.?\s*)?([A-Z0-9\-]+)', re.I)
# These are literature DOIs verified against public publication pages, not guessed
# from the SciGlass record ID; all title/author/year checks must pass.
_CHECKED = {
  '29884': ('10.1080/1042015042000272866', 'physical, optical and electrical properties of calcium bismuth borate glasses', 'sanghi', '2004'),
  '38228': ('10.1134/S0036023610110185', 'glass formation in the cao-bi2o3-b2o3 and sro-bi2o3-b2o3 systems', 'egorysheva', '2010'),
  '38803': ('10.1016/j.jnoncrysol.2011.08.002', 'structural investigations on 20mo-', 'rejisha', '2011'),
}

def clean(value):
    if value is None or value is pd.NA:
        return ''
    try:
        if not bool(pd.notna(value)):
            return ''
    except (TypeError, ValueError):
        return ''
    s=str(value).strip()
    return '' if s.lower() in ('nan','none','null','nat') else s

def _https(value):
    s=clean(value)
    try:
        u=urlparse(s)
        return s if u.scheme=='https' and u.hostname and not u.username and not u.password else ''
    except ValueError:
        return ''

def _sciglass(url):
    try:
        host=(urlparse(url).hostname or '').lower()
        return host=='sciglass.uni-jena.de' or host.endswith('.sciglass.uni-jena.de')
    except ValueError:
        return False

def _doi(value):
    d=clean(value)
    if d.lower().startswith('https://doi.org/'):
        d=unquote(d.split('/',3)[-1])
    d=d.strip().rstrip('.,;')
    return d if _DOI.fullmatch(d) else ''

def _scholar_query(row):
    title=clean(row.get('Source_title'))
    author=clean(row.get('Primary_author')) or clean(row.get('SciGlass_author'))
    year=clean(row.get('Primary_year')) or clean(row.get('SciGlass_year'))
    if not title:
        bib=clean(row.get('Source_bibliography'))
        # Exact bibliographic description is better than a fabricated title.
        query=bib or ('SciGlass '+clean(row.get('KOD_verified')))
    else:
        query='"'+title+'" '+author+' '+year
    return 'https://scholar.google.com/scholar?q='+quote_plus(query[:750].strip())

def resolve_source(row):
    """UI links only. Keys url, action, direct, alternative_url, note.

    Search links aren't source-verified direct links; do not present them as such.
    The original SciGlass provenance stays in CSV and is unaffected.
    """
    source_url=_https(row.get('Source_URL'))
    bib=clean(row.get('Source_bibliography'))
    kind=clean(row.get('Source_kind'))
    search=_scholar_query(row)
    doi=_doi(row.get('Source_DOI'))
    if not doi and source_url and 'doi.org' in urlparse(source_url).netloc.lower():
        doi=_doi(source_url)
    if not doi:
        code=clean(row.get('KOD_verified')).split('.')[0]
        checked=_CHECKED.get(code)
        if checked:
            verified_doi, needle, surname, year=checked
            title=clean(row.get('Source_title')).lower()
            author=(clean(row.get('Primary_author')) or clean(row.get('SciGlass_author'))).lower()
            y=clean(row.get('Primary_year')) or clean(row.get('SciGlass_year'))
            if (needle in title and surname in author and year==y.split('.')[0]):
                doi=verified_doi
    if doi:
        return dict(url='https://doi.org/'+doi, action='Открыть публикацию по DOI',
                    direct=True, alternative_url=search,
                    note='DOI ведёт на страницу публикации; полный текст может быть платным.', doi=doi)
    if source_url and not _sciglass(source_url):
        is_patent='patents.google.com' in (urlparse(source_url).hostname or '').lower()
        return dict(url=source_url, action='Открыть патент' if is_patent else 'Открыть публикацию',
                    direct=True, alternative_url='' if is_patent else search,
                    note='Внешняя страница источника; доступ к полному тексту зависит от сайта.', doi='')
    if kind=='Патент' or bool(_PATENT.search(bib)):
        match=_PATENT.search(bib)
        query=(match.group(1).upper()+match.group(2)) if match else bib
        return dict(url='https://patents.google.com/?q='+quote_plus(query),
                    action='Найти патент по номеру', direct=False, alternative_url='',
                    note='Поисковая ссылка, не подтверждённая страница конкретного патента.', doi='')
    return dict(url=search, action=('Найти статью по названию' if clean(row.get('Source_title'))
                                    else 'Найти источник по библиографии'),
                direct=False, alternative_url='',
                note='Поиск по библиографическим данным; прямую ссылку на статью установить не удалось.', doi='')
