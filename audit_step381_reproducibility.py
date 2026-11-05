"""Read-only comparison of the live GlassDesign ML with the saved 2026-11-04 step381 run.

Uses the author's Google Drive OAuth to read reference results privately.
Never trains, overwrites models, or edits the archived step381 run.
Run from the same Colab session that started the standalone Streamlit demo.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/content/glassdesign-ml")
REFERENCE = {
    "manifest.json": "1R7dEKgbDuwKJNWizFdrBUv0QkGtez1AM",
    "all_new_recipes.csv": "1wHTapjg1_5M3QaXj6L8PhLx7kJ_6j1uC",
    "selected_representatives.csv": "1ZPKLKU3I5jXihYfQL2Iv5AMq0bTnzDaC",
}
CORE = (
    "glass_benchmark_v21", "glass_consensus_ga_v25",
    "glass_consensus_pilot_v27", "glass_pilot_rank_only",
    "GlassDesignML_step364_dual_feedback",
)
NUMERIC = (
    "Density_g_cm3", "RefractiveIndex", "Tg_C", "Tsoft_C",
    "GN_Density_g_cm3", "GN_RefractiveIndex", "GN_Tg_C", "GN_Tsoft_C",
)
SUMMARY = (
    "pilot", "new", "gd_pass", "glassnet_pass", "both_pass",
    "selected_representatives", "successful_families",
)


def private_download(service, file_id):
    from googleapiclient.http import MediaIoBaseDownload
    buff = io.BytesIO()
    downloader = MediaIoBaseDownload(buff, service.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buff.seek(0)
    return buff.read()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def comparable(meta, reference):
    options = ["seed", "mode", "criteria", "selected_names", "manual_bounds",
               "excluded_oxides", "budget", "max_families", "pilot_per_family"]
    return all(meta.get(k) == reference.get(k) for k in options)


def compare_numbers(old, new, columns, tolerance=1e-6):
    cols = [c for c in columns if c in old.columns and c in new.columns]
    if len(cols) != len(columns):
        return False, float("inf"), f"Нет столбцов: {set(columns) - set(cols)}"
    oldarr = old[list(cols)].to_numpy(dtype=float)
    newarr = new[list(cols)].to_numpy(dtype=float)
    if oldarr.shape != newarr.shape:
        return False, float("inf"), f"Форма {oldarr.shape} vs {newarr.shape}"
    delta = np.abs(oldarr - newarr)
    if not np.isfinite(delta).all():
        return False, float("inf"), "Есть NaN/Inf"
    worst = float(delta.max()) if delta.size else 0.0
    return bool(np.all(delta <= tolerance)), worst, f"Максимальное отклонение {worst:.9g}"


def main():
    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build

    print("=== КОНТРОЛЬ ВОСПРОИЗВОДИМОСТИ GLASSDESIGN ML — ШАГ 381 ===")
    auth.authenticate_user()
    credentials, _ = google.auth.default()
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    raw = {name: private_download(service, file_id)
           for name, file_id in REFERENCE.items()}
    reference = json.loads(raw["manifest.json"].decode("utf-8-sig"))
    original = pd.read_csv(io.BytesIO(raw["all_new_recipes.csv"]), encoding="utf-8-sig")
    original_selected = pd.read_csv(
        io.BytesIO(raw["selected_representatives.csv"]), encoding="utf-8-sig")
    print("Контрольный запуск:", "2026-11-04 21:45", "seed", reference["seed"])
    print("Оригинал: пробных", reference["summary"]["pilot"],
          "| новых", reference["summary"]["new"],
          "| BOTH", reference["summary"]["both_pass"])

    # Verify identical GA source and chemical/model input configuration.
    checks = {}
    for name in CORE:
        code = ROOT / (name + ".py")
        checks["core_" + name] = code.is_file() and (
            sha256(code) == reference["backend_sha256"][name])
    for path, expected in reference["configuration_sha256"].items():
        file = ROOT / "runtime" / path
        checks["config_" + path] = file.is_file() and sha256(file) == expected
    print("Код алгоритма:", "СОВПАДАЕТ" if all(v for k, v in checks.items()
          if k.startswith("core_")) else "ЕСТЬ ОТЛИЧИЯ")
    print("Химические настройки:", "СОВПАДАЮТ" if all(v for k, v in checks.items()
          if k.startswith("config_")) else "ЕСТЬ ОТЛИЧИЯ")

    # Prefer the current Streamlit-run results from its private temporary directory.
    candidates = sorted(Path("/tmp").glob("glassdesign_public_*/result/manifest.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    selected = None
    for mpath in candidates:
        try:
            candidate = json.loads(mpath.read_text(encoding="utf-8"))
            if comparable(candidate, reference):
                selected = mpath.parent
                print("Текущий результат найден:", selected)
                break
        except (ValueError, OSError):
            continue

    if selected is None:
        print("Подходящего веб-запуска нет. Выполняю один контрольный поиск...")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from glass380_engine import run
        out = Path(tempfile.mkdtemp(prefix="glassdesign_repro_")) / "result"
        result = run(
            reference["criteria"],
            root=ROOT / "runtime",
            mode=reference["mode"],
            selected_names=reference.get("selected_names"),
            manual_bounds=reference.get("manual_bounds"),
            excluded_oxides=reference.get("excluded_oxides") or [],
            seed=int(reference["seed"]),
            budget=int(reference["budget"]),
            max_families=int(reference["max_families"]),
            out_dir=out,
        )
        selected = Path(result["folder"])

    current_meta = json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
    new = pd.read_csv(selected / "all_new_recipes.csv", encoding="utf-8-sig")
    new_selected = pd.read_csv(selected / "selected_representatives.csv",
                               encoding="utf-8-sig")

    id_ok = (len(original) == len(new)
             and original["Candidate_ID"].equals(new["Candidate_ID"])
             and original["Run_Unique_ID"].equals(new["Run_Unique_ID"])
             and original["Glass_family"].equals(new["Glass_family"]))
    oxide_columns = [
        col for col in original.columns
        if col in new.columns
        and col in __import__("json").loads(
            (ROOT / "runtime/GlassDesignML/model_features.json")
            .read_text(encoding="utf-8"))["Плотность"]
        and col not in NUMERIC
    ]
    # Stable chemical comparison: use every oxide present in all four models.
    features = json.loads(
        (ROOT / "runtime/GlassDesignML/model_features.json")
        .read_text(encoding="utf-8"))
    oxides = sorted(set.intersection(*(set(v) for v in features.values())))
    chem_ok, chem_max, _ = compare_numbers(original, new, oxides)
    pred_ok, pred_max, _ = compare_numbers(original, new, NUMERIC)
    shortlist_ok = (
        list(original_selected["Run_Unique_ID"]) ==
        list(new_selected["Run_Unique_ID"]))
    summary_ok = all(
        reference["summary"].get(k) == current_meta["summary"].get(k)
        for k in SUMMARY
    )
    config_ok = all(checks.values())

    print("\n--- ЧИСЛЕННОЕ СРАВНЕНИЕ С 4 НОЯБРЯ ---")
    print("ID и порядок всех 120 составов:", "✅ СОВПАЛИ" if id_ok else "❌ РАЗЛИЧАЮТСЯ")
    print("Химический состав (63 общих оксида):",
          "✅ СОВПАЛ" if chem_ok else "❌ РАЗЛИЧАЕТСЯ",
          f"| макс. разница {chem_max:.10g} мол.%")
    print("8 прогнозов (4 GlassDesign + 4 GlassNet):",
          "✅ СОВПАЛИ" if pred_ok else "❌ РАЗЛИЧАЮТСЯ",
          f"| макс. разница {pred_max:.10g}")
    print("15 отобранных представителей:",
          "✅ СОВПАЛИ" if shortlist_ok else "❌ РАЗЛИЧАЮТСЯ")
    print("Сводные показатели:", "✅ СОВПАЛИ" if summary_ok else "❌ РАЗЛИЧАЮТСЯ")
    print("\nСТАРЫЙ результат: BOTH =", reference["summary"]["both_pass"],
          "/ 120; отобрано =", reference["summary"]["selected_representatives"])
    print("НОВЫЙ результат: BOTH =", current_meta["summary"]["both_pass"],
          "/ 120; отобрано =", current_meta["summary"]["selected_representatives"])

    passed = bool(config_ok and id_ok and chem_ok and pred_ok
                  and shortlist_ok and summary_ok)
    print("\nИТОГ:", "✅ ПОЛНОЕ СОВПАДЕНИЕ" if passed else
          "⚠️ ЕСТЬ ОТЛИЧИЯ — НУЖЕН РАЗБОР")
    report = {
        "historical_reference": "Final_Generator_step381/run_20261104_214513_922452_71",
        "current_folder": str(selected), "source_checks": checks,
        "same_ids": bool(id_ok), "same_chemical_compositions": bool(chem_ok),
        "maximum_composition_difference_mol": chem_max,
        "same_eight_predictions": bool(pred_ok),
        "maximum_prediction_difference": pred_max,
        "same_selected_representatives": bool(shortlist_ok),
        "same_summary": bool(summary_ok),
        "full_match": passed,
    }
    destination = Path("/content/GlassDesignML_step381_reproducibility_report.json")
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print("Отчёт сохранён:", destination)


if __name__ == "__main__":
    main()
