"""GlassDesign ML step408: conservative public source destinations.

Avoid misleading links to beta-only SciGlass or search results that appear to
be the source itself. Bibliographic provenance is kept unchanged.
"""
from __future__ import annotations
import re
import unicodedata
from urllib.parse import quote_plus,urlparse,unquote
import pandas as pd

_DOI=re.compile(r'^10\.\d{4,9}/[^\s<>]+$',re.I)
_VERIFIED_DOIS={
    '29884':('10.1080/1042015042000272866','physical optical and electrical properties of calcium bismuth borate glasses','sanghi','2004'),
    '38228':('10.1134/S0036023610110185','glass formation in the cao bi2o3 b2o3 and sro bi2o3 b2o3 systems','egorysheva','2010'),
    '38803':('10.1016/j.jnoncrysol.2011.08.002','structural investigations on 20mo','rejisha','2011'),
}
# Verified against the original journal PDF; this paper appears without a DOI.
_VERIFIED_PDFS=(
    ('effect of different concentrations of doped rare earth element on borate silica oxide glass structure',
     'bahra','2015','https://chalcogen.ro/47_Bahra.pdf'),
)

def clean(value):
    if value is None or value is pd.NA:return ''
    try:
        if not bool(pd.notna(value)):return ''
    except (TypeError,ValueError):return ''
    value=str(value).strip()
    return '' if value.lower() in ('nan','none','null','nat') else value

def normalized(s):
    s=unicodedata.normalize('NFKD',clean(s)).casefold()
    s=re.sub(r'[^a-z0-9а-яё]+',' ',s)
    return ' '.join(s.split())

def valid_https(value):
    s=clean(value)
    try:
        u=urlparse(s)
        return s if u.scheme=='https' and u.hostname and not u.username and not u.password else ''
    except ValueError:return ''

def _doi(value):
    d=clean(value)
    if d.lower().startswith('https://doi.org/'):
        d=unquote(d.split('/',3)[-1])
    d=d.strip().rstrip('.,;')
    return d if _DOI.fullmatch(d) else ''

def _search(row):
    title=clean(row.get('Source_title'))
    author=(clean(row.get('Primary_author')) or clean(row.get('SciGlass_author')))
    year=(clean(row.get('Primary_year')) or clean(row.get('SciGlass_year'))).split('.')[0]
    bib=clean(row.get('Source_bibliography'))
    kind=clean(row.get('Source_kind'))
    # Do not build overly restrictive exact Scholar queries: older translated
    # references, theses, and pre-digital patent numbers commonly return zero.
    if title:
        words=re.findall(r'[\w]+',title)
        query=' '.join(words[:10])+' '+author.split(' ')[0]+' '+year
    elif bib:
        query=' '.join(re.findall(r'[\w]+',bib)[:13])
    else:
        return ''
    query=query.strip()
    if kind=='Патент' and not query:
        return ''
    return 'https://www.google.com/search?q='+quote_plus(query[:240]) if query else ''

def resolve_source(row):
    """Return a *verified link* or an honest 'bibliography only' fallback.

    url='' means there is NO verified direct source URL. `search_url` is only
    auxiliary discovery; search results are not claimed as a publication.
    """
    author=(clean(row.get('Primary_author')) or clean(row.get('SciGlass_author')))
    year=(clean(row.get('Primary_year')) or clean(row.get('SciGlass_year'))).split('.')[0]
    title=normalized(row.get('Source_title'))
    surname=normalized(author).split(' ')[0] if normalized(author) else ''
    search=_search(row)
    doi=_doi(row.get('Source_DOI'))
    url=valid_https(row.get('Source_URL'))
    if not doi and urlparse(url).hostname=='doi.org': doi=_doi(url)
    code=clean(row.get('KOD_verified')).split('.')[0]
    if not doi and code in _VERIFIED_DOIS:
        candidate,needle,expected_surname,expected_year=_VERIFIED_DOIS[code]
        if normalized(needle) in title and surname==expected_surname and year==expected_year:
            doi=candidate
    if doi:
        return {'url':'https://doi.org/'+doi,'action':'Открыть публикацию по DOI',
                'direct':True,'search_url':search,'note':'Ссылка ведёт на страницу издателя; полный текст может быть платным.',
                'doi':doi,'kind':'doi'}
    for expected_title,expected_author,expected_year,pdf in _VERIFIED_PDFS:
        if expected_title in title and surname==expected_author and year==expected_year:
            return {'url':pdf,'action':'Открыть PDF статьи',
                    'direct':True,'search_url':'','note':'Прямой PDF на сайте журнала, сверенный по названию, автору и году.',
                    'doi':'','kind':'verified_pdf'}
    # An archive entry may contain an unverified Google Patents URL. We do
    # not treat that as a successful link unless it is checked independently.
    # Same for generic URLs: a valid HTTPS scheme does not prove a page exists.
    return {'url':'','action':'Прямой источник недоступен',
            'direct':False,'search_url':search,
            'note':'Библиография подтверждена архивом SciGlass, но открытая ссылка не установлена. Поиск может ничего не найти.',
            'doi':'','kind':'bibliography_only'}
