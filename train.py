#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Глобальные настройки
warnings.filterwarnings("ignore")   # глушим спам FutureWarning/UserWarning от sklearn/pandas
SEED = 42                          # единый seed для воспроизводимости
np.random.seed(SEED)

# RDKit опциональная зависимость: без неё работаем только на IC50/CC50/культуре.
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    RDLogger.DisableLog("rdApp.*")   # отключаем stderr-спам RDKit
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# LightGBM
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# Поиск файлов
def find(name, hint):
    """
    Ищет файл в таком порядке:
        1) явный путь из аргументов CLI (если существует);
        2) файл с таким именем в текущей папке;
        3) рекурсивный поиск по подпапкам (**/name);
        4) fallback — возвращаем name как есть.
    """
    if hint and os.path.exists(hint):
        return hint
    if os.path.exists(name):
        return name
    hits = glob.glob(f"**/{name}", recursive=True)
    return hits[0] if hits else name


# Загрузка и унификация данных
def load_data(train_path, test_path, extra_path):
    """
    Читает train/test/extra, приводит имена колонок к единому виду
    (SMILES, SI, IC50_uM_VACV, CC50_uM) и конкатенирует train + extra.
    """
    train = pd.read_csv(find("train_v2.csv", train_path))
    test = pd.read_csv(find("test.csv", test_path))

    # В датасетах встречаются разные варианты имен — нормализуем.
    if "SI VACV" in train.columns:
        train = train.rename(columns={"SI VACV": "SI"})
    if "smiles" in train.columns and "SMILES" not in train.columns:
        train = train.rename(columns={"smiles": "SMILES"})
    if "smiles" in test.columns and "SMILES" not in test.columns:
        test = test.rename(columns={"smiles": "SMILES"})

    train["is_extra"] = 0   # флаг: строка из основного датасета

    extra_file = find("vacv_extra.csv", extra_path)
    if extra_file and os.path.exists(extra_file):
        ex = pd.read_csv(extra_file).rename(columns={
            "smiles": "SMILES", "IC50_uM": "IC50_uM_VACV", "CC50_uM": "CC50_uM"})
        if "Culture" not in ex.columns:
            ex["Culture"] = np.nan
        if "strain" not in ex.columns:
            ex["strain"] = np.nan
        ex["is_extra"] = 1
        keep = [c for c in ["SMILES", "CC50_uM", "IC50_uM_VACV", "SI",
                            "Culture", "strain", "is_extra"] if c in ex.columns]
        train = pd.concat([train, ex[keep]], ignore_index=True)
        print(f"Добавлено из доп. датасета ({extra_file}): {len(ex)} строк")

    return train, test

# Целевая переменная: SI = CC50 / IC50
def rule_si(ic50, cc50):
    """
    Считает SI по формуле из условия + соглашения для пропусков:
        - нет IC50; SI = 10000 (считаем соединение неактивным);
        - есть IC50, нет CC50; SI = 0     (считаем не цитотоксичным).
    Используется как baseline и как основа для residual-режима.
    """
    ic50, cc50 = np.asarray(ic50, float), np.asarray(cc50, float)
    out = cc50 / ic50
    out = np.where(np.isnan(ic50), 10000.0, out)
    out = np.where(~np.isnan(ic50) & np.isnan(cc50), 0.0, out)
    return out



# Признаки молекул
def parent_key(smi):
    """
    Нормализует SMILES для группировки в CV:
        - берёт крупнейший фрагмент (отбрасывает противоионы);
        - убирает изомерию (стерео).
    Нужно, чтобы один и тот же "скаффолд" не попадал одновременно
    в train и validation (иначе утечка).
    """
    if not HAS_RDKIT:
        return smi
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    big = max(Chem.GetMolFrags(m, asMols=True), key=lambda x: x.GetNumHeavyAtoms())
    return Chem.MolToSmiles(big, isomericSmiles=False)


def mol_features(smiles_list, n_bits=512):
    """
    Строит две матрицы признаков:
        * Morgan fingerprints (radius=2, хэшированные, счётчики);
        * ~200 RDKit физико-химических дескрипторов.
    Битые SMILES заменяются нулями/NaN, чтобы пайплайн не падал.
    """
    fps, descs = [], []
    names = [d[0] for d in Descriptors._descList]
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            fps.append(np.zeros(n_bits))
            descs.append(np.full(len(names), np.nan))
            continue

        # Morgan fingerprint (radius 2) ловит субструктуры.
        fp = AllChem.GetHashedMorganFingerprint(m, 2, nBits=n_bits)
        arr = np.zeros(n_bits)
        for k, v in fp.GetNonzeroElements().items():
            arr[k] = v
        fps.append(arr)

        # RDKit дескрипторы — глобальные свойства молекулы.
        row = []
        for _, f in Descriptors._descList:
            try:
                row.append(f(m))
            except Exception:
                row.append(np.nan)
        descs.append(row)

    fps = pd.DataFrame(fps, columns=[f"fp{i}" for i in range(n_bits)])
    descs = pd.DataFrame(descs, columns=names).replace([np.inf, -np.inf], np.nan)
    return fps, descs


def build_matrix(df, use_fp, culture_cols, desc_cols=None, n_bits=512):
    """
    Собирает матрицу признаков X:
        - ic_missing / cc_missing (индикаторы пропусков);
        - log_ic50, log_cc50, log_rule (числовые признаки от IC50/CC50);
        - one-hot культура клеток;
        - Morgan fingerprints + RDKit дескрипторы (если use_fp).
    """
    X = pd.DataFrame(index=range(len(df)))

    # 1) Числовые признаки, полученные из IC50/CC50.
    ic_col = next((c for c in ["IC50_uM_VACV", "IC50_uM", "ic50"] if c in df.columns), None)
    cc_col = next((c for c in ["CC50_uM", "cc50"] if c in df.columns), None)
    ic = df[ic_col].values if ic_col else np.full(len(df), np.nan)
    cc = df[cc_col].values if cc_col else np.full(len(df), np.nan)

    X["ic_missing"] = np.isnan(ic).astype(int)
    X["cc_missing"] = np.isnan(cc).astype(int)
    X["log_ic50"] = np.log10(np.where(np.isnan(ic), np.nan, ic) + 1e-9)
    X["log_cc50"] = np.log10(np.where(np.isnan(cc), np.nan, cc) + 1e-9)
    X["log_rule"] = np.log1p(rule_si(ic, cc))

    # 2) One-hot культура клеток.
    cult_series = df["Culture"] if "Culture" in df.columns else pd.Series(["unknown"] * len(df))
    cult = cult_series.fillna("unknown").str.lower().str.extract(
        r"(vero|hela|fibroblast|bsc|a549|mdck|unknown)", expand=False).fillna("other")
    for c in culture_cols:
        X[f"cult_{c}"] = (cult.values == c).astype(int)

    # 3) Молекулярные признаки (если RDKit доступен и не отключён).
    if use_fp:
        smi_col = next((c for c in ["SMILES", "smiles"] if c in df.columns), None)
        fps, descs = mol_features(df[smi_col].tolist() if smi_col else [""] * len(df),
                                  n_bits=n_bits)
        if desc_cols is not None:
            descs = descs[desc_cols]   # на тесте оставляем только те же колонки, что на train
        X = pd.concat([X, fps, descs], axis=1)

    return X.reset_index(drop=True)


# Модели
def get_models():
    """
    Возвращает словарь {имя: необученная модель}.
    Все модели настроены консервативно — данные маленькие,
    поэтому боремся с переобучением.
    """
    models = {
        # Линейная модель на стандартизованных признаках с L2-регуляризацией.
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),

        # RandomForest: ограничиваем глубину и размер листа,
        # max_features="sqrt" декоррелирует деревья.
        "rf": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=15,
            max_depth=8,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
        ),

        # ExtraTrees: сильнее рандомизирует разбиения для меньше дисперсия.
        "extra_trees": ExtraTreesRegressor(
            n_estimators=150,
            min_samples_leaf=15,
            max_depth=8,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
        ),
    }
    if HAS_LGBM:
        models["lgbm"] = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.02,
            num_leaves=16,
            max_depth=5,
            subsample=0.7,
            colsample_bytree=0.4,
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=SEED,
            verbose=-1,
        )
    return models


def fit_predict(model, Xtr, ytr, Xte, mode, rule_tr, rule_te):
    """
    Обучает модель и возвращает предсказания в исходной шкале SI.

    mode == "direct"   : таргет = log1p(SI)
    mode == "residual" : таргет = log1p(SI) - log1p(rule)
    """
    # 1) inf -> NaN.
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan)
    Xte = Xte.replace([np.inf, -np.inf], np.nan)

    # 2) Заполняем NaN медианами тренировочной выборки.
    medians = Xtr.median()
    medians = medians.where(np.isfinite(medians), 0.0)
    Xtr = Xtr.fillna(medians).fillna(0)
    Xte = Xte.fillna(medians).fillna(0)

    # 3) Ещё раз убираем inf (на случай, если fillna не справился).
    Xtr = Xtr.replace([np.inf, -np.inf], 0.0)
    Xte = Xte.replace([np.inf, -np.inf], 0.0)

    # 4) Защита от переполнения float32.
    CLIP_MAX = 1e6
    Xtr = np.clip(Xtr.values if hasattr(Xtr, "values") else Xtr, -CLIP_MAX, CLIP_MAX)
    Xte = np.clip(Xte.values if hasattr(Xte, "values") else Xte, -CLIP_MAX, CLIP_MAX)

    # 5) Финальная страховка от NaN.
    Xtr = np.where(np.isfinite(Xtr), Xtr, 0.0)
    Xte = np.where(np.isfinite(Xte), Xte, 0.0)

    # 6) sklearn требует float32 — приводим явно.
    Xtr = Xtr.astype(np.float32)
    Xte = Xte.astype(np.float32)

    # Базовая линия (для residual-режима).
    base_tr = np.log1p(rule_tr) if mode == "residual" else 0.0
    base_te = np.log1p(rule_te) if mode == "residual" else 0.0

    model.fit(Xtr, ytr - base_tr)
    pred_log = model.predict(Xte) + base_te
    # expm1 обратен log1p; клиппинг в 0 — SI не может быть отрицательным.
    return np.clip(np.expm1(pred_log), 0, None)


def sanitize_features(X):
    """
    Дополнительная очистка матрицы признаков перед обучением:
        - inf -> NaN -> медиана столбца;
        - клиппинг дескрипторов RDKit 1%/99% перцентилями (убираем тяжёлые хвосты).
    Fingerprints и one-hot культуры не клиппим — они уже в [0, 1] или счётчики.
    """
    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        col_data = X[col]
        if col_data.isna().all():
            X[col] = 0.0
        else:
            med = col_data.median()
            if not np.isfinite(med):
                med = 0.0
            X[col] = col_data.fillna(med)

    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    for col in X.columns:
        if not col.startswith("fp") and not col.startswith("cult_") \
                and col not in ["ic_missing", "cc_missing"]:
            q01 = X[col].quantile(0.01)
            q99 = X[col].quantile(0.99)
            if np.isfinite(q01) and np.isfinite(q99):
                X[col] = X[col].clip(q01, q99)
    return X


# Метрики
def scores(y_true, y_pred):
    """Возвращает RMSE/MAE в лог- и исходной шкалах."""
    lt, lp = np.log1p(y_true), np.log1p(y_pred)
    return dict(
        RMSE_log=np.sqrt(np.mean((lt - lp) ** 2)),   # главная метрика задачи
        MAE_log=np.mean(np.abs(lt - lp)),
        RMSE_raw=np.sqrt(np.mean((y_true - y_pred) ** 2)),
        MAE_raw=np.mean(np.abs(y_true - y_pred)),
    )



# Основной пайплайн
def main():
    # Парсинг аргументов командной строки.
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--extra", default="vacv_extra.csv")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--no-fp", action="store_true", help="без молекулярных признаков")
    ap.add_argument("--n-bits", type=int, default=512, help="число бит Morgan fingerprint")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    # Подготовка данных.
    use_fp = HAS_RDKIT and not args.no_fp
    if not use_fp:
        print("RDKit не используется: модели видят только IC50/CC50/культуру.")

    train, test = load_data(args.train, args.test, args.extra)
    smi_col = next((c for c in ["SMILES", "smiles"] if c in train.columns), "SMILES")
    si_col = next((c for c in ["SI", "SI VACV"] if c in train.columns), "SI")

    # Убираем строки без SMILES или без таргета.
    train = train.dropna(subset=[smi_col, si_col]).reset_index(drop=True)

    # Группы для GroupKFold
    groups = train[smi_col].map(parent_key).values
    y = train[si_col].values.astype(float)

    #  Матрицы признаков.
    culture_cols = ["vero", "hela", "fibroblast", "bsc", "a549", "mdck", "unknown", "other"]
    X = build_matrix(train, use_fp, culture_cols, n_bits=args.n_bits)
    X = sanitize_features(X)

    X_test = build_matrix(test, use_fp, culture_cols, n_bits=args.n_bits)
    X_test = sanitize_features(X_test)
    X_test = X_test[X.columns]   # гарантируем одинаковый порядок колонок

    # Baseline по правилу (CC50/IC50).
    ic_tr_col = next((c for c in ["IC50_uM_VACV", "IC50_uM", "ic50"] if c in train.columns), None)
    cc_tr_col = next((c for c in ["CC50_uM", "cc50"] if c in train.columns), None)
    ic_te_col = next((c for c in ["IC50_uM_VACV", "IC50_uM", "ic50"] if c in test.columns), None)
    cc_te_col = next((c for c in ["CC50_uM", "cc50"] if c in test.columns), None)

    rule_tr = rule_si(train[ic_tr_col] if ic_tr_col else np.nan,
                      train[cc_tr_col] if cc_tr_col else np.nan)
    rule_te = rule_si(test[ic_te_col] if ic_te_col else np.nan,
                      test[cc_te_col] if cc_te_col else np.nan)

    print(f"Обучающих строк: {len(train)}, признаков: {X.shape[1]}")

    #  Кросс-валидация.
    gkf = GroupKFold(n_splits=args.folds)
    rows = [dict(approach="rule", model="CC50/IC50", **scores(y, rule_tr))]
    oof_store = {}   # OOF-предсказания

    for mode in ("direct", "residual"):
        for name in get_models():
            oof = np.zeros(len(y))
            for tr_i, va_i in gkf.split(X, y, groups):
                m = get_models()[name]   # свежая модель на каждый фолд
                oof[va_i] = fit_predict(
                    m,
                    X.iloc[tr_i], np.log1p(y[tr_i]),
                    X.iloc[va_i], mode,
                    rule_tr[tr_i], rule_tr[va_i],
                )
            rows.append(dict(approach=mode, model=name, **scores(y, oof)))
            oof_store[(mode, name)] = oof

    res = pd.DataFrame(rows).sort_values("RMSE_log").reset_index(drop=True)
    print("\nРезультаты кросс-валидации:")
    print(res.round(3).to_string(index=False))

    # Выбор лучшего варианта и финальное обучение на всех данных.
    best = res.iloc[0]
    print(f"\nЛучший вариант по RMSE_log: {best.approach} / {best.model}")

    if best.approach == "rule":
        pred = rule_te       # правило оказалось лучше любой модели
    else:
        m = get_models()[best.model]
        pred = fit_predict(m, X, np.log1p(y), X_test,
                           best.approach, rule_tr, rule_te)

    # Сохранение submission.
    sub_path = find("sample_submission.csv", "sample_submission.csv")
    if sub_path and os.path.exists(sub_path):
        sub = pd.read_csv(sub_path)
        sub[sub.columns[-1]] = pred   # заполняем последнюю колонку
    else:
        # Fallback: строим submission с нуля.
        idx_col = "index" if "index" in test.columns else test.columns[0]
        sub = pd.DataFrame({idx_col: test[idx_col], "SI VACV": pred})
        print("sample_submission.csv не найден: создан базовый submission.")

    sub.to_csv(args.out, index=False)
    print(f"Сохранено: {args.out} ({len(sub)} строк)")

    # Контроль распределения предсказаний (для отладки).
    print("\nСтатистика предсказаний:")
    print(f"  Медиана SI: {np.median(pred):.2f}")
    print(f"  Среднее SI: {np.mean(pred):.2f}")
    print(f"  Min/Max: {np.min(pred):.4f} / {np.max(pred):.4f}")
    print(f"  Доля SI > 10000: {(pred > 10000).mean():.2%}")
    print(f"  Доля SI < 1:     {(pred < 1).mean():.2%}")


if __name__ == "__main__":
    main()