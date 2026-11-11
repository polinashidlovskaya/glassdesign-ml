"""GlassDesign ML step399: strictly additive integration of 391 + 385 + 398.

No original models or training data modified. Never confuse model consensus,
experimental analogue proximity, or heuristic A/B/C with experimental success.
"""
from __future__ import annotations

import hashlib
import json
import re
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from glass391_engine import open_saved, predicted_pass, load_catalog

SCIENCE_VERSION = "399.1"
PROPERTIES = {"Density293K":"Density_g_cm3", "RefractiveIndex":"RefractiveIndex",
              "Tg":"Tg_C", "TdilatometricSoftening":"TdilatometricSoftening_C"}


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as inp:
        for piece in iter(lambda: inp.read(1024*1024),b""):
            h.update(piece)
    return h.hexdigest()


def load_verified_run(folder):
    """Only open step391; robustly recheck *physics*, criteria, partition and IDs."""
    folder=Path(folder).resolve()
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("step")!=391:
        raise ValueError("Интерфейс 399 открывает только запуски шага 391; старые 380/381/390 не прошли исправленный фильтр.")
    result=open_saved(folder)
    tables=result["tables"]
    allrows=tables["all_new_recipes"]
    good=tables["both_pass_recipes"]
    bad=tables["excluded_from_recommendations"]
    criterion=result["criteria"]
    if len(allrows)==0 or allrows.Candidate_ID.duplicated().any() or allrows.Run_Unique_ID.duplicated().any():
        raise ValueError("Набор кандидатов пуст или содержит повторяющиеся ID")
    if len(allrows)!=len(good)+len(bad):
        raise ValueError("Нарушено разбиение на рекомендованные и исключённые")
    if set(good.Run_Unique_ID)&set(bad.Run_Unique_ID) or set(allrows.Run_Unique_ID)!=set(good.Run_Unique_ID)|set(bad.Run_Unique_ID):
        raise ValueError("Кандидаты двух разделов пересекаются или потеряны")
    expected_gd=predicted_pass(allrows,criterion)
    expected_gn=predicted_pass(allrows,criterion,prefix="GN_")
    expected=expected_gd & expected_gn
    for flag,computed in (("GD_pass",expected_gd),("GN_pass",expected_gn),
                          ("Both_pass",expected),("Recommendation_visible",expected)):
        raw=allrows[flag]
        observed=raw if raw.dtype==bool else raw.astype(str).str.lower().map({"true":True,"false":False})
        if observed.isna().any() or not np.array_equal(observed.to_numpy(dtype=bool),computed):
            raise ValueError(f"В сохранённом результате поле {flag} противоречит правилам отбора")
    expected_ids=set(allrows.loc[expected,"Run_Unique_ID"].astype(str))
    if set(good.Run_Unique_ID.astype(str))!=expected_ids:
        raise ValueError("В рекомендациях есть неверный состав или отсутствует подходящий")
    if not set(tables["selected_representatives"].Run_Unique_ID.astype(str)).issubset(expected_ids):
        raise ValueError("В коротком списке есть нерекомендованные составы")
    if len(allrows)!=int(result["summary"]["new"]) or len(good)!=int(result["summary"]["both_pass"]):
        raise ValueError("Статистика запуска не совпала с таблицами")
    for name,rows in (("all_new_recipes",allrows),("both_pass_recipes",good)):
        if rows.Run_Unique_ID.isna().any():raise ValueError(f"Пустой Run_Unique_ID: {name}")
    result["input_fingerprint"]=sha256_file(folder/"both_pass_recipes.csv")
    result["run_folder"]=folder
    return result



def latest_verified_run(root):
    """Select current step391 result, including the verified `smoke_*` pilot runs.

    The original engine.latest_saved() ignores smoke_* and may return a legacy
    step390 run. That is unsuitable as default for the new UI.
    """
    root=Path(root)
    base=root/"GlassDesignML_Upgrade"/"Temperature_Screen_step391"
    candidates=list(base.glob("run_*/manifest.json"))+list(base.glob("smoke_*/manifest.json"))
    choices=[]
    for path in candidates:
        try:
            meta=json.loads(path.read_text(encoding="utf-8"))
            if meta.get("step") != 391: continue
            date=meta.get("created_at_utc") or ""
            choices.append((date,path))
        except (ValueError,OSError,TypeError):continue
    if not choices:
        raise FileNotFoundError("Нет сохранённых запусков 391: создай расчёт или выполни контрольный шаг 391")
    choices.sort(reverse=True)
    errors=[]
    for _,path in choices:
        try:return load_verified_run(path.parent)
        except (OSError,ValueError,KeyError,AssertionError) as ex:errors.append(f"{path.parent.name}: {ex}")
    raise ValueError("Все сохранённые запуски 391 повреждены или неполны: "+"; ".join(errors[:3]))


def report_directory(root, run_folder):
    root=Path(root)
    run_folder=Path(run_folder).resolve()
    unique=hashlib.sha256(str(run_folder).encode("utf-8")).hexdigest()[:12]
    path=root/"GlassDesignML_Upgrade"/"Integrated_Release_step399"/"reports"/(run_folder.name+"_"+unique)
    return path


def report_exists(path, meta_file, fingerprint):
    path=Path(path)
    meta=path/meta_file
    if not meta.is_file():return False
    try:
        info=json.loads(meta.read_text(encoding="utf-8"))
        return info.get("input_fingerprint")==fingerprint and info.get("module_version")==SCIENCE_VERSION
    except Exception:return False


def _save_csv(df,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    part=path.with_name(path.name+".tmp")
    df.to_csv(part,index=False,encoding="utf-8-sig")
    part.replace(path)


def _save_meta(reportdir,name,result,additional):
    d={"step":399,"module_version":SCIENCE_VERSION,
       "original_run_path":str(result["run_folder"]),
       "input_fingerprint":result["input_fingerprint"],
       "created_at_utc":datetime.now(timezone.utc).isoformat(),
       "models_modified":False,"source_runs_modified":False,
       **additional}
    target=Path(reportdir)/name
    part=target.with_name(target.name+".tmp")
    part.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding="utf-8")
    part.replace(target)
    return d


def compute_scientific_report(result,root,out_dir):
    """Original 385 live audit, plus correctly defined Td and conservative near-Tg A/B rule."""
    from glass385_scientific import run_live_audit,screen_reliability,diverse_shortlist
    eligible=result["tables"]["both_pass_recipes"].copy().reset_index(drop=True)
    if eligible.empty:
        raise ValueError("Нет рекомендованных составов для научного аудита")
    audited,issues=run_live_audit(root,eligible)
    ranked=screen_reliability(audited)
    # 385 legacy Tsoft means actually dilatometric Td after property-alignment 390.
    ranked["GD_Td_minus_Tg_C"]=ranked.Tsoft_C.astype(float)-ranked.Tg_C.astype(float)
    ranked["GN_Td_minus_Tg_C"]=ranked.GN_Tsoft_C.astype(float)-ranked.GN_Tg_C.astype(float)
    ranked["Td_margin_borderline_GD"]=ranked.GD_Td_minus_Tg_C.lt(5.)
    ranked["Td_margin_borderline_GN"]=ranked.GN_Td_minus_Tg_C.lt(5.)
    if (ranked.GD_Td_minus_Tg_C < -1e-9).any() or (ranked.GN_Td_minus_Tg_C < -1e-9).any():
        raise AssertionError("Несогласованный температурный прогноз в рекомендациях")
    ranked["Screening_tier_original_385"]=ranked["Screening_tier"]
    borderline=ranked.Td_margin_borderline_GD | ranked.Td_margin_borderline_GN
    promote=ranked.Screening_tier.str.startswith("A —") & borderline
    ranked.loc[promote,"Screening_tier"]="B — нужны дополнительные проверки"
    ranked["Scientific_warnings"]=ranked.Scientific_warnings.astype(str).str.replace(
        "GlassNet: Tsoft почти равна Tg, нужна проверка определений",
        "GlassNet: дилатометрическая Td почти равна Tg; учесть погрешность моделей",regex=False)
    ranked.loc[ranked.Td_margin_borderline_GD,"Scientific_warnings"] += "; GlassDesign: Td−Tg <5 °C"
    ranked.loc[ranked.Td_margin_borderline_GN,"Scientific_warnings"] += "; GlassNet: Td−Tg <5 °C"
    if "Local_Tg_check_limited_to_top_6" in ranked.columns:
        ranked.loc[ranked.Local_Tg_check_limited_to_top_6.astype(bool),"Scientific_warnings"] += "; локальный Tg: проверены только основные шесть оксидов"
    counts=ranked.Screening_tier.str[:1].value_counts().to_dict()
    tier_order={"A":0,"B":1,"C":2,"D":3}
    ranked["__order"]=ranked.Screening_tier.str[:1].map(tier_order).fillna(9)
    ranked=ranked.sort_values(["__order","Nearest_Tsoft_halfL1_mol","Candidate_ID"],kind="stable",na_position="last").drop(columns="__order")
    a=ranked.loc[ranked.Screening_tier.str.startswith("A —")]
    short=diverse_shortlist(a,max_items=6,min_distance=2.) if not a.empty else a.copy()
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    _save_csv(ranked,out_dir/"scientific_screen_step399.csv")
    _save_csv(short,out_dir/"scientific_shortlist_step399.csv")
    _save_meta(out_dir,"science_meta_step399.json",result,{
        "eligible_rows":len(eligible),"screened_rows":len(ranked),"tier_counts":counts,
        "issues":list(map(str,issues)) if isinstance(issues,(tuple,list)) else [str(issues)],
        "reference_distances":"Original 385 near_analogues of TRAIN/reference data; NOT independent source validation",
        "tier_cutoffs":"Heuristic, uncalibrated. Borderline either-model Td-Tg <5 C demotes A to B",
        "VITRIFY":"Not part of the 399 A/B/C scientific tier; separate validation needed",
        "limitations":"An A tier is NOT a probability or guarantee of glass formation."})
    return ranked,short,issues


def find_index(root):
    base=Path(root)/"GlassDesignML_Upgrade"/"SciGlass_Universal_Literature_step398"/"cache"
    candidates=sorted(base.glob("sciglass_four_properties_mol_398_*.npz"))
    if not candidates:raise FileNotFoundError(f"Индекс SciGlass шага 398 не найден: {base}")
    return candidates[-1]


def find_reviewed_catalog(root):
    base=Path(root)/"GlassDesignML_Upgrade"/"SciGlass_Literature_Catalog_step397"
    runs=sorted(base.glob("run_*/source_catalog_step397.csv"),reverse=True)
    return runs[0] if runs else None


def compute_literature_report(result,root,out_dir,max_half_l1=5.,top_k=3,search_crossref=False):
    """Read source-only SciGlass cache 398; verify archival bibliography and joins for this run.

    The journal links of step397 are reused only when KOD+primary RefID+author+year match.
    """
    from glass398_universal_literature import (load_measurement_index,search_nearest,
        load_archive,build_archive_lookup,read_reviewed_catalog,attach_sources,
        make_coverage,enrich_with_crossref)
    if not 0 < float(max_half_l1) <= 100:raise ValueError("Недопустимое расстояние")
    if not 1 <= int(top_k) <= 10:raise ValueError("Число аналогов должно быть 1–10")
    root=Path(root);out_dir=Path(out_dir)
    eligible=result["tables"]["both_pass_recipes"].copy().reset_index(drop=True)
    if eligible.empty:raise ValueError("Нет рекомендованных составов")
    if eligible.Run_Unique_ID.isna().any() or eligible.Run_Unique_ID.duplicated().any():
        raise ValueError("Неуникальные Run_Unique_ID в исходных данных")
    if "Input_Row" in eligible:raise ValueError("Input_Row зарезервирован для поиска")
    recipes=eligible.copy()
    recipes.insert(0,"Input_Row",np.arange(1,len(recipes)+1,dtype=int))
    index=find_index(root)
    arrays=load_measurement_index(index,recipes)
    matched=search_nearest(recipes,arrays,max_half_l1=float(max_half_l1),top_k=int(top_k))
    cache_dir=root/"GlassDesignML_Upgrade"/"SciGlass_Universal_Literature_step398"/"cache"/"archive"
    archive,info=load_archive(cache_dir,force_download=False)
    lookup=build_archive_lookup(archive)
    reviewed_path=find_reviewed_catalog(root)
    reviewed=read_reviewed_catalog(reviewed_path) if reviewed_path else {}
    linked,sources=attach_sources(matched,lookup,reviewed)
    doi_requests=0;doi_errors=0
    if search_crossref and not sources.empty:
        sources,doi_requests,doi_errors=enrich_with_crossref(sources,root/"GlassDesignML_Upgrade"/"SciGlass_Universal_Literature_step398"/"cache"/"crossref_399.json",max_requests=10,enable=True)
        fields=["Source_DOI","Source_URL","Source_link_type","Source_DOI_confidence"]
        aligned=sources.set_index("SciGlass_ID")
        for field in fields:
            linked[field]=linked.SciGlass_ID.map(aligned[field])
    if not linked.empty:
        if len(linked)!=len(matched) or linked.duplicated(["Input_Row","Property","Rank"]).any():
            raise AssertionError("Нарушена уникальность/целостность литературного сопоставления")
        if (linked.Distance_halfL1_mol > max_half_l1+0.002).any():
            raise AssertionError("Аналог находится вне заданного порога")
        if linked.Source_DOI.fillna("").ne("").any():
            with_doi=linked.Source_DOI.fillna("").ne("")
            if not linked.loc[with_doi,"Source_archive_confirmed"].astype(bool).all():
                raise AssertionError("DOI привязан к неподтверждённой записи")
        keys=eligible[["Run_Unique_ID","Candidate_ID"]].copy()
        for _,row in linked[["Run_Unique_ID","Candidate_ID"]].drop_duplicates().iterrows():
            if not ((keys.Run_Unique_ID.astype(str)==str(row.Run_Unique_ID)) & (keys.Candidate_ID.astype(str)==str(row.Candidate_ID))).any():
                raise AssertionError("В литературном отчёте найден ID другого запуска")
    coverage=make_coverage(linked,recipes)
    out_dir.mkdir(parents=True,exist_ok=True)
    _save_csv(linked,out_dir/"sciglass_analogues_step399.csv")
    _save_csv(sources,out_dir/"sciglass_sources_step399.csv")
    _save_csv(coverage,out_dir/"literature_coverage_step399.csv")
    meta=_save_meta(out_dir,"literature_meta_step399.json",result,{
        "eligible_rows":len(eligible),"analogue_rows":len(linked),
        "different_sciglass_ids":int(linked.SciGlass_ID.nunique()) if len(linked) else 0,
        "unique_source_kod":int(linked.KOD_verified.nunique()) if len(linked) else 0,
        "unique_doi":int(sources.loc[sources.Source_DOI.fillna("").ne(""),"Source_DOI"].nunique()) if len(sources) else 0,
        "index_path":str(index),"index_sha256":sha256_file(index),
        "reviewed_source_catalog":str(reviewed_path) if reviewed_path else None,
        "max_half_l1_mol":max_half_l1,"top_k":top_k,
        "crossref_requests":doi_requests,"crossref_errors":doi_errors,
        "archive_sha256":{k:d["sha256"] for k,d in info.items()},
        "scope":"All SciGlass indexed compositions, each property independent",
        "limitations":"Measured neighbour is not an independent validation, DOI may be auto-matched."})
    return linked,coverage,sources,meta


def load_cached_science(reportdir,fingerprint):
    p=Path(reportdir)
    if not report_exists(p,"science_meta_step399.json",fingerprint):return None
    if not (p/"scientific_screen_step399.csv").is_file():return None
    df=pd.read_csv(p/"scientific_screen_step399.csv",encoding="utf-8-sig")
    if df.Run_Unique_ID.duplicated().any():raise ValueError("Повтор в научном отчёте")
    return df


def load_cached_literature(reportdir,fingerprint):
    p=Path(reportdir)
    if not report_exists(p,"literature_meta_step399.json",fingerprint):return None
    names=["sciglass_analogues_step399.csv","literature_coverage_step399.csv"]
    if not all((p/n).is_file() for n in names):return None
    return tuple(pd.read_csv(p/n,encoding="utf-8-sig") for n in names)
