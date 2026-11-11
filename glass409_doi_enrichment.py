"""GlassDesign ML step409 — verify DOI metadata for NEW SciGlass sources.

This module never alters models, original SciGlass, step391 runs or step403
literature exports. Only high-confidence title/author/year matches become links.
The global reusable cache is keyed by provenance + metadata fingerprint.
An unverifiable/ambiguous DOI stays a review candidate, NOT a source URL.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pandas as pd

_VERSION = '409.1'
_DOI_RE = re.compile(r'^10\.\d{4,9}/\S+$', re.I)
# Provenance-verified manually against publishers/publication catalogues.
# Only apply if SOURCE KOD + exact title, author and publication year match.
_KNOWN = {
  '43046': ('10.1039/C4TB01504E','Structure and solubility behaviour of zinc containing phosphate glasses','kanwal','2015'),
  '39808': ('10.1016/j.jnoncrysol.2012.03.031','Viscosity and kinetic fragility of alkaline earth zinc phosphate glasses','striepe','2012'),
  '44624': ('10.2320/matertrans.MI201503','Preparation of antibacterial ZnO-CaO-P2O5-Nb2O5 invert glasses','lee','2016'),
  '25754': ('10.1016/S0022-3093(02)01795-7','The structure and properties of binary zinc phosphate glasses studied by molecular dynamics simulations','tischendorf','2003'),
  '43169': ('10.1080/00387010.2014.892512','Physical and optical properties of calcium zinc borophosphate glasses doped with manganese ions','wan','2015'),
}


def clean(value):
    if value is None or value is pd.NA: return ''
    try:
        if not bool(pd.notna(value)): return ''
    except (TypeError, ValueError): return ''
    result = str(value).strip()
    return '' if result.lower() in ('nan','none','null','nat') else result


def normalized(value):
    result = unicodedata.normalize('NFKD', clean(value)).casefold()
    result = ''.join(c for c in result if not unicodedata.combining(c))
    result = re.sub(r'[^a-z0-9а-яё]+',' ',result)
    return ' '.join(result.split())


def tokens(value):
    return set(normalized(value).split())


def normalize_doi(value):
    s=clean(value)
    if s.lower().startswith('https://doi.org/'):
        s=s[len('https://doi.org/'):]
    s=s.rstrip('.,; ') 
    return s if _DOI_RE.fullmatch(s) else ''


def as_year(value):
    match=re.search(r'\b(?:18|19|20)\d{2}\b',clean(value))
    return int(match.group()) if match else None


def _surname(value):
    text=normalized(value)
    if not text: return ''
    # SciGlass first author: "Tischendorf B.C.", "Lee Sungho", "Wan Ming Hua".
    return text.split()[0]


def _candidate_author_match(author, source):
    want=_surname(source)
    if not want or not author: return False
    if isinstance(author, list):
        people=[]
        for a in author:
            if isinstance(a,dict):people.append(' '.join([clean(a.get('family')),clean(a.get('given'))]))
            else:people.append(clean(a))
    else: people=[clean(author)]
    # Accept family/surname of the FIRST author, not coincidental coauthors.
    first=normalized(people[0]) if people else ''
    return bool(first and want in first.split())


def _similarity(a,b):
    a=normalized(a);b=normalized(b)
    if not a or not b:return 0.0,0.0
    ratio=SequenceMatcher(None,a,b).ratio()
    A=set(a.split());B=set(b.split())
    overlap=len(A&B)/len(A|B) if A|B else 0
    return ratio,overlap


def _source_key(row):
    k=clean(row.get('KOD_verified')).split('.')[0]
    r=clean(row.get('Primary_RefID')).split('.')[0]
    if not k or not r:
        raise ValueError('Неизвестен проверенный KOD либо Primary_RefID')
    return f'{k}:{r}'


def _metadata_fingerprint(row):
    pieces=[_source_key(row),clean(row.get('Source_title')),clean(row.get('Primary_author')),clean(row.get('Primary_year')),clean(row.get('Source_kind')),clean(row.get('Source_bibliography'))]
    return hashlib.sha256('\x1f'.join(pieces).encode('utf8')).hexdigest()


def unique_documents(df):
    need=('KOD_verified','Primary_RefID','Source_title','Primary_author','Primary_year','Source_kind','Source_bibliography')
    missing=[n for n in need if n not in df]
    if missing:raise ValueError('В данных SciGlass отсутствуют поля: '+str(missing))
    d=df.drop_duplicates(subset=['KOD_verified','Primary_RefID']).copy()
    d['_Source_key_409']=[_source_key(r) for r in d.to_dict('records')]
    d['_Metadata_fingerprint_409']=[_metadata_fingerprint(r) for r in d.to_dict('records')]
    if d['_Source_key_409'].duplicated().any():raise ValueError('Несогласованная библиографическая связь KOD/RefID')
    return d.reset_index(drop=True)


def _json_get(endpoint, params=None, timeout=9, mailto=None):
    query=urlencode(params or {})
    url=endpoint+('?' + query if query else '')
    agent='GlassDesignML/1.0 (research source verification; contact: see SciGlass project administrator)'
    if mailto: agent='GlassDesignML/1.0 (mailto:'+mailto+')'
    req=Request(url,headers={'User-Agent':agent,'Accept':'application/json'})
    with urlopen(req,timeout=timeout) as resp:
        return json.loads(resp.read(1200000).decode('utf8'))


def fetch_crossref(title, *, mailto=None):
    data=_json_get('https://api.crossref.org/works',{
        'query.bibliographic': title,'rows':5,
        'select':'DOI,title,author,issued,container-title,page,volume,type'
    },mailto=mailto)
    rows=data.get('message',{}).get('items',[])
    out=[]
    for row in rows:
        parts=row.get('issued',{}).get('date-parts',[]) or [[]]
        yr=parts[0][0] if parts and parts[0] else None
        out.append({'provider':'Crossref','doi':row.get('DOI',''),
            'title':(row.get('title') or [''])[0],
            'year':yr,'authors':row.get('author',[]),
            'journal':(row.get('container-title') or [''])[0],
            'page':row.get('page',''),'type':row.get('type','')})
    return out


def fetch_openalex(title, *, api_key=''):
    params={'search':title,'per_page':5,
            'select':'doi,title,publication_year,authorships,type'}
    if api_key: params['api_key']=api_key
    data=_json_get('https://api.openalex.org/works',params)
    rows=[]
    for row in data.get('results',[]):
        rows.append({'provider':'OpenAlex','doi':row.get('doi',''),
             'title':row.get('title',''),'year':row.get('publication_year'),
             'authors':[p.get('author',{}).get('display_name','') for p in row.get('authorships',[])],
             'journal':'','page':'','type':row.get('type','')})
    return rows


def _first_page(text, *, bibliography=False):
    s=clean(text)
    if bibliography:
        m=re.search(r'\bp\.\s*(\d+)\b',s,re.I)
    else:
        m=re.search(r'\b(\d+)\b',s)
    return m.group(1) if m else ''


def _journal_evidence(reference_bibliography, journal):
    # Journal abbreviations often differ (J.Non-Cryst.Solids versus Journal of
    # Non-Crystalline Solids). Use only POSITIVE corroboration here.
    words=[w for w in normalized(journal).split() if len(w)>=5]
    original=normalized(reference_bibliography)
    matched=sum(1 for w in words if w in original)
    return bool(words and matched>=min(2,len(words)))


def _evaluate_candidate(candidate,source):
    doi=normalize_doi(candidate.get('doi'))
    if not doi:return None
    ts,word=_similarity(source.get('Source_title'),candidate.get('title'))
    sy=as_year(source.get('Primary_year'))
    cy=as_year(candidate.get('year'))
    year_ok= bool(sy and cy and abs(sy-cy)<=1)
    author_ok=_candidate_author_match(candidate.get('authors'),source.get('Primary_author'))
    # Don't auto-accept misleading short/common titles or year/author mismatches.
    meaningful=len(tokens(source.get('Source_title')))>=5
    original_page=_first_page(source.get('Source_bibliography'),bibliography=True)
    found_page=_first_page(candidate.get('page'))
    page_conflict=bool(original_page and found_page and int(original_page)!=int(found_page))
    journal_corroborated=_journal_evidence(source.get('Source_bibliography'),candidate.get('journal'))
    good=bool(ts>=.94 and word>=.85 and meaningful and year_ok and author_ok and not page_conflict)
    return {'doi':doi,'provider':candidate.get('provider',''),
        'page_match':bool(original_page and found_page and not page_conflict),
        'page_conflict':page_conflict,'journal_corroborated':journal_corroborated,
        'title_found':clean(candidate.get('title')),
        'title_similarity':round(ts,4),'token_overlap':round(word,4),
        'year_found':cy,'year_match':year_ok,'author_match':author_ok,
        'eligible':good,'score':round(.75*ts+.25*word,4)}


def _seed_match(source):
    k=_source_key(source).split(':')[0]
    if k not in _KNOWN:return None
    doi,title,surname,year=_KNOWN[k]
    similarity,word=_similarity(source.get('Source_title'),title)
    if similarity<.97 or word<.90 or _surname(source.get('Primary_author'))!=surname or as_year(source.get('Primary_year'))!=int(year):
        return None
    return doi


def _validate_hits(source,hits):
    checked=[x for x in (_evaluate_candidate(c,source) for c in hits) if x]
    unique={}
    for c in checked:
        key=c['doi'].casefold()
        if key not in unique or c['score']>unique[key]['score']:
            unique[key]=c
    ranked=sorted(unique.values(),key=lambda x:x['score'],reverse=True)
    eligible=[x for x in ranked if x['eligible']]
    if not eligible:
        return {'status':'review_required' if ranked else 'not_found','doi':'',
                'provider':'','candidates':ranked[:4]}
    top=eligible[0]
    # Two *different* strong DOI records with nearly identical title imply ambiguity.
    ambiguous=any(x['doi'].casefold()!=top['doi'].casefold() and x['eligible'] and x['score']>=top['score']-.035 for x in eligible[1:])
    if ambiguous:
        return {'status':'review_required','doi':'','provider':'', 'candidates':ranked[:4]}
    return {'status':'verified_automatic','doi':top['doi'], 'provider':top['provider'],
            'candidates':ranked[:4]}


def _empty_decision(status='not_found'):
    return {'status':status,'doi':'','provider':'','candidates':[]}


def _default_fetch_crossref(title, **kwargs):
    return fetch_crossref(title,**kwargs)


def _default_fetch_openalex(title, **kwargs):
    return fetch_openalex(title,**kwargs)


def verify_document(source, *, crossref=None, openalex=None, mailto=None, openalex_key='', use_openalex=True):
    if clean(source.get('Source_kind'))=='Патент' or not clean(source.get('Source_title')):
        return _empty_decision('not_applicable')
    manual=_seed_match(source)
    if manual:
        return {'status':'verified_external','doi':manual,'provider':'publisher/catalogue', 'candidates':[]}
    cr=crossref or _default_fetch_crossref
    oa=openalex or _default_fetch_openalex
    hits=[];errors=[]
    try:hits+=cr(clean(source['Source_title']),mailto=mailto)
    except Exception as e:errors.append('Crossref: '+str(e)[:160])
    result=_validate_hits(source,hits)
    # OpenAlex search is a fallback, not a requirement for verified Crossref matches.
    if result['status']!='verified_automatic' and use_openalex:
        try:
            hits+=oa(clean(source['Source_title']),api_key=openalex_key)
            result=_validate_hits(source,hits)
        except Exception as e:errors.append('OpenAlex: '+str(e)[:160])
    if errors and result['status'] in ('not_found','review_required'):
        result['errors']=errors
        if len(hits)==0: result['status']='network_error'
    return result


def load_cache(path):
    p=Path(path)
    if not p.is_file():return {}
    try:
        data=json.loads(p.read_text(encoding='utf8'))
        if not isinstance(data,dict) or data.get('version')!=_VERSION:return {}
        entries=data.get('entries',{})
        return entries if isinstance(entries,dict) else {}
    except (OSError,ValueError):return {}


def write_cache(path,entries):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.temp.json')
    temp.write_text(json.dumps({'version':_VERSION,'entries':entries},ensure_ascii=False,indent=2),encoding='utf8')
    temp.replace(p)


def enrich(literature, *, cache_path, out_dir, input_fingerprint,
           crossref=None,openalex=None,mailto=None,openalex_key='',
           use_openalex=True,progress=None,sleep_sec=.25, max_new_queries=120):
    """Return a copy of original measurements with DOIs verified for new sources.

    Offline errors are not auto-certified. Cache avoids repeating old requests;
    separate exports and a fingerprint prevent mismatched run evidence.
    """
    if not isinstance(literature,pd.DataFrame):raise TypeError('Нужна таблица SciGlass')
    docs=unique_documents(literature)
    cache=load_cache(cache_path)
    decisions={};fresh_queries=0; errors=[]; consecutive_api_failures=0
    for i,source in enumerate(docs.to_dict('records')):
        key=_source_key(source);finger=_metadata_fingerprint(source)
        old=cache.get(key,{})
        if old.get('fingerprint')==finger and old.get('result',{}).get('status')!='network_error':
            answer=old['result']
        else:
            if consecutive_api_failures>=3:
                answer=_empty_decision('network_unavailable')
            elif fresh_queries>=max_new_queries:
                answer=_empty_decision('not_checked_budget')
            else:
                answer=verify_document(source,crossref=crossref,openalex=openalex,
                    mailto=mailto,openalex_key=openalex_key,use_openalex=use_openalex)
                if answer['status']=='network_error':consecutive_api_failures+=1
                elif answer['status'] not in ('not_applicable','verified_external'):consecutive_api_failures=0
                if answer['status'] not in ('not_applicable','verified_external'):
                    fresh_queries+=1
                    if sleep_sec:time.sleep(sleep_sec)
                cache[key]={'fingerprint':finger,'result':answer,
                       'last_checked_utc':datetime.now(timezone.utc).isoformat()}
                if (i+1)%8==0:write_cache(cache_path,cache)
        if answer.get('errors'):errors.append((key,answer['errors']))
        decisions[key]=answer
        if progress:progress(i+1,len(docs),answer['status'])
    write_cache(cache_path,cache)
    d=docs.copy()
    d['DOI_Status_409']=[decisions[_source_key(r)]['status'] for r in docs.to_dict('records')]
    d['DOI_409']=[decisions[_source_key(r)].get('doi','') for r in docs.to_dict('records')]
    d['DOI_Provider_409']=[decisions[_source_key(r)].get('provider','') for r in docs.to_dict('records')]
    d['DOI_Review_Candidates_409']=[json.dumps(decisions[_source_key(r)].get('candidates',[])[:3],ensure_ascii=False) for r in docs.to_dict('records')]
    out=literature.copy()
    if 'Source_DOI' not in out:out['Source_DOI']=''
    if 'Source_URL' not in out:out['Source_URL']=''
    for col in ('Source_DOI','Source_URL','Source_DOI_confidence'):
        if col not in out:out[col]=''
        out[col]=out[col].astype('object')
    out['DOI_Status_409']='';out['DOI_Provider_409']='';out['DOI_409']=''
    for key,group_idx in out.groupby(['KOD_verified','Primary_RefID'],sort=False,dropna=False).groups.items():
        lookup=f'{str(key[0]).split(".")[0]}:{str(key[1]).split(".")[0]}'
        dec=decisions.get(lookup,_empty_decision())
        out.loc[group_idx,'DOI_Status_409']=dec['status']
        out.loc[group_idx,'DOI_Provider_409']=dec.get('provider','')
        doi=dec.get('doi','')
        if doi and dec['status'] in ('verified_automatic','verified_external'):
            out.loc[group_idx,'DOI_409']=doi
            out.loc[group_idx,'Source_DOI']=doi
            out.loc[group_idx,'Source_URL']='https://doi.org/'+doi
            out.loc[group_idx,'Source_DOI_confidence']=('Проверено по публикации' if dec['status']=='verified_external' else 'Автоматически сверено: название, автор, год')
    outdir=Path(out_dir);outdir.mkdir(parents=True,exist_ok=True)
    out.to_csv(outdir/'sciglass_literature_doi_step409.csv',encoding='utf-8-sig',index=False)
    d.to_csv(outdir/'source_doi_audit_step409.csv',encoding='utf-8-sig',index=False)
    review=d.loc[d.DOI_Status_409.isin(['review_required','not_found','network_error','not_checked_budget','not_applicable','network_unavailable'])].copy()
    review.to_csv(outdir/'source_doi_review_step409.csv',encoding='utf-8-sig',index=False)
    meta={'step':409,'input_fingerprint':str(input_fingerprint),'source_rows':int(len(d)),
        'linked_rows':int(len(out)), 'unique_sciglass_ids':int(out.SciGlass_ID.nunique()) if 'SciGlass_ID' in out else None,
        'with_doi':int((d.DOI_409!='').sum()),
        'automatic_doi':int((d.DOI_Status_409=='verified_automatic').sum()),
        'externally_checked_doi':int((d.DOI_Status_409=='verified_external').sum()),
        'needs_review':int(len(review)),
        'network_errors':len(errors),'fresh_searches':int(fresh_queries),
        'cache':str(cache_path), 'note':'Automatic DOI metadata matching ≠ external manual confirmation, SciGlass experimental glass ≠ independent ML test.'}
    (outdir/'doi_meta_step409.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    return out,d,meta


def load_enriched(out_dir, input_fingerprint):
    p=Path(out_dir)
    try:
        meta=json.loads((p/'doi_meta_step409.json').read_text(encoding='utf8'))
        if meta.get('input_fingerprint')!=str(input_fingerprint):return None
        table=pd.read_csv(p/'sciglass_literature_doi_step409.csv',encoding='utf-8-sig',dtype={'SciGlass_ID':str,'KOD_verified':str,'Primary_RefID':str},low_memory=False)
        if len(table)!=meta.get('linked_rows'):return None
        return table
    except (OSError,ValueError,KeyError):return None
