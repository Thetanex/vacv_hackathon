#!/usr/bin/env python3
"""
Источники (все публичные, без ключей):
    ChEMBL REST API https://www.ebi.ac.uk/chembl/api/data
    PubChem PUG REST https://pubchem.ncbi.nlm.nih.gov/rest/pug
    NCBI E-utilities https://eutils.ncbi.nlm.nih.gov/entrez/eutils  
"""
import argparse
import hashlib
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
except ImportError: 
    def tqdm(x, **kw):
        return x
# RDKit нужен для канонизации SMILES и пересчёта мкг/мл -> мкМ (через MW).
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
#Базовые URL'ы публичных API

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Дисковый кэш ответов, повторный запуск не ходит в сеть и работает офлайн.
CACHE = Path("cache")
CACHE.mkdir(exist_ok=True)
# Типы измерений, которые считаем «противовирусными» / «цитотоксичными».
# EC50 — тоже противовирусная активность (но помечается отдельно: менее точная, чем IC50).
ANTIVIRAL_TYPES = ["IC50", "EC50"]                   
CYTOTOX_TYPES = ["CC50", "CD50", "TC50", "CTC50"]
CYTOTOX_DESC = re.compile(r"cytotox|toxicity|cell viability|CC50|cell growth inhibition", re.I)


# HTTP
session = requests.Session()

session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504))))
session.headers["User-Agent"] = "vacv-hackathon-collector/1.0"


def get(url, params=None, as_json=True, post_data=None, pause=0.25):
    """GET/POST с дисковым кэшем (повторный запуск не ходит в сеть)."""
    key = hashlib.md5(json.dumps([url, params, post_data], sort_keys=True,
                                 default=str).encode()).hexdigest()
    f = CACHE / (key + (".json" if as_json else ".txt"))
    if f.exists():
        txt = f.read_text(encoding="utf-8")
        return json.loads(txt) if as_json else txt
    r = (session.post(url, params=params, data=post_data, timeout=120) if post_data
         else session.get(url, params=params, timeout=120))
    time.sleep(pause)  # (NCBI: <=3 запросов/с)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    r.encoding = "utf-8"
    f.write_text(r.text, encoding="utf-8")
    return r.json() if as_json else r.text



# Единицы измерения / химия
def mol_weight(smiles):
    if not HAS_RDKIT or not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(m) if m else None


def to_uM(value, unit, smiles=None):
    """Перевод в мкМ. Массовые концентрации требуют молярной массы."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    u = (unit or "").strip().lower().replace("μ", "u").replace("µ", "u")
    molar = {"nm": 1e-3, "um": 1.0, "mm": 1e3, "m": 1e6, "pm": 1e-6}
    if u in molar:
        return v * molar[u]
    mass = {"ug.ml-1": 1.0, "ug/ml": 1.0, "ug ml-1": 1.0, "mg/ml": 1e3,
            "ng/ml": 1e-3, "mg.ml-1": 1e3, "ng.ml-1": 1e-3}
    if u in mass:
        mw = mol_weight(smiles)
        return v * mass[u] * 1000.0 / mw if mw else np.nan  # ug/mL -> uM = ug/mL*1000/MW
    return np.nan


def canon(smiles):
    #Канонический SMILES (None, если не парсится).
    if not smiles:
        return None
    if not HAS_RDKIT:
        return smiles
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m else None


def parent_key(smiles):
    #Ключ для дедупликации: самый крупный фрагмент без стереохимии.
    if not HAS_RDKIT or not smiles:
        return smiles
    m = Chem.MolFromSmiles(smiles)
    if not m:
        return None
    frags = Chem.GetMolFrags(m, asMols=True)
    big = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    return Chem.MolToSmiles(big, isomericSmiles=False)



# ChEMBL

def chembl_paginate(endpoint, params, key):
    params = dict(params, limit=1000, format="json")
    url = f"{CHEMBL}/{endpoint}.json"
    while url:
        data = get(url, params=params)
        if not data:
            return
        for item in data[key]:
            yield item
        nxt = data["page_meta"]["next"]
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
        params = None  # next-ссылка уже содержит параметры


def chembl_activities(extra_params):
    fields = ("molecule_chembl_id,canonical_smiles,standard_type,standard_value,"
              "standard_units,standard_relation,document_chembl_id,assay_chembl_id,"
              "assay_description,target_organism,pchembl_value")
    p = dict(extra_params, only=fields)
    return list(chembl_paginate("activity", p, "activities"))


def collect_chembl():
    print("[ChEMBL] поиск мишеней Vaccinia virus ...")
    targets = list(chembl_paginate("target", {"organism__icontains": "vaccinia"}, "targets"))
    tids = sorted({t["target_chembl_id"] for t in targets})
    print(f"[ChEMBL] мишеней: {len(tids)}")
    if not tids:
        return pd.DataFrame()

    # 1) противовирусные активности
    antiviral = chembl_activities({
        "target_chembl_id__in": ",".join(tids),
        "standard_type__in": ",".join(ANTIVIRAL_TYPES),
        "standard_units__isnull": "false"})
    av = pd.DataFrame(antiviral)
    print(f"[ChEMBL] противовирусных измерений: {len(av)}")
    if av.empty:
        return av
    docs = sorted(av.document_chembl_id.dropna().unique())

    # 2) цитотоксичность из тех же публикаций
    tox_rows = []
    for d in tqdm(docs, desc="[ChEMBL] CC50 по документам"):
        tox_rows += chembl_activities({
            "document_chembl_id": d,
            "standard_type__in": ",".join(CYTOTOX_TYPES + ["IC50"]),
            "standard_units__isnull": "false"})
    tox = pd.DataFrame(tox_rows)
    if not tox.empty:
        is_cc = tox.standard_type.isin(CYTOTOX_TYPES)
        is_ic_tox = (tox.standard_type == "IC50") & tox.assay_description.fillna("").str.contains(CYTOTOX_DESC)
        not_vacv = ~tox.target_organism.fillna("").str.contains("accinia", case=False)
        tox = tox[(is_cc | is_ic_tox) & not_vacv]

    # 3) DOI
    doi = {}
    for d in tqdm(docs, desc="[ChEMBL] DOI"):
        j = get(f"{CHEMBL}/document/{d}.json")
        doi[d] = (j or {}).get("doi")

    def norm(df, kind):
        df = df.copy()
        df["smiles"] = df.canonical_smiles
        df["val_uM"] = [to_uM(v, u, s) for v, u, s in
                        zip(df.standard_value, df.standard_units, df.smiles)]
        df = df[df.standard_relation.fillna("=").isin(["=", "~"])]  # без ">" и "<"
        df = df.dropna(subset=["val_uM", "smiles"])
        df["kind"] = kind
        return df

    parts = [norm(av, "antiviral")]
    if not tox.empty:
        parts.append(norm(tox, "cytotox"))
    all_ = pd.concat(parts, ignore_index=True)
    all_["doi"] = all_.document_chembl_id.map(doi)

    out = []
    for (mol, d), g in all_.groupby(["molecule_chembl_id", "document_chembl_id"]):
        a = g[g.kind == "antiviral"]
        c = g[g.kind == "cytotox"]
        if a.empty:
            continue
        # приоритет IC50 над EC50
        a_ic = a[a.standard_type == "IC50"]
        a_used = a_ic if len(a_ic) else a
        out.append(dict(
            smiles=g.smiles.iloc[0],
            IC50_uM=a_used.val_uM.median(),
            CC50_uM=c.val_uM.median() if len(c) else np.nan,
            doi=g.doi.iloc[0],
            source="ChEMBL",
            source_id=f"{mol}|{d}",
            antiviral_type=a_used.standard_type.iloc[0],
            description=" ; ".join(a_used.assay_description.dropna().unique())[:300]))
    return pd.DataFrame(out)



# PubChem BioAssay
PC_QUERIES = ['vaccinia[All Fields] AND "antiviral"[All Fields]',
              'vaccinia virus[All Fields] AND (IC50 OR EC50 OR CC50)']


def pubchem_find_aids():
    aids = set()
    for q in PC_QUERIES:
        j = get(f"{EUTILS}/esearch.fcgi",
                {"db": "pcassay", "term": q, "retmax": 2000, "retmode": "json"})
        aids |= set(j["esearchresult"]["idlist"])
    return sorted(aids, key=int)


def pubchem_doi(aid):
    #AID - PMID - DOI 
    j = get(f"{PUG}/assay/aid/{aid}/description/JSON")
    if not j:
        return None
    try:
        desc = j["PC_AssayContainer"][0]["assay"]["descr"]
    except (KeyError, IndexError):
        return None
    for x in desc.get("xref", []):
        ref = x.get("xref", {})
        if "pmid" in ref:
            s = get(f"{EUTILS}/esummary.fcgi",
                    {"db": "pubmed", "id": ref["pmid"], "retmode": "json"})
            try:
                for a in s["result"][str(ref["pmid"])]["articleids"]:
                    if a["idtype"] == "doi":
                        return a["value"]
            except (KeyError, TypeError):
                pass
    return None


def pubchem_assay_name(aid):
    j = get(f"{PUG}/assay/aid/{aid}/description/JSON")
    try:
        return j["PC_AssayContainer"][0]["assay"]["descr"]["name"]
    except Exception:
        return ""


def cid_to_smiles(cids):
    res = {}
    cids = list(cids)
    for i in range(0, len(cids), 200):
        chunk = cids[i:i + 200]
        j = get(f"{PUG}/compound/cid/property/IsomericSMILES,CanonicalSMILES/JSON",
                post_data={"cid": ",".join(map(str, chunk))})
        for p in (j or {}).get("PropertyTable", {}).get("Properties", []):
            res[p["CID"]] = p.get("IsomericSMILES") or p.get("SMILES") or p.get("CanonicalSMILES")
    return res


COL_RE = {"IC50": re.compile(r"\bIC[-_ ]?50\b", re.I),
          "EC50": re.compile(r"\bEC[-_ ]?50\b", re.I),
          "CC50": re.compile(r"\b(CC[-_ ]?50|CD50|TC50|CTC50)\b", re.I)}


def pubchem_unit_from_col(col):
    c = col.lower().replace("µ", "u").replace("μ", "u")
    for u in ("um", "nm", "mm", "ug/ml", "ng/ml", "mg/ml"):
        if re.search(rf"[\(\[ _]{re.escape(u)}[\)\] _]?$|\({re.escape(u)}\)|_{re.escape(u)}\b", c):
            return u
    return "um"  # по умолчанию в таких тестах мкМ 


def collect_pubchem():
    aids = pubchem_find_aids()
    print(f"[PubChem] найдено AID: {len(aids)}")
    rows = []
    for aid in tqdm(aids, desc="[PubChem] тесты"):
        txt = get(f"{PUG}/assay/aid/{aid}/CSV", as_json=False, pause=0.4)
        if not txt:
            continue
        try:
            df = pd.read_csv(io.StringIO(txt), low_memory=False)
        except Exception:
            continue
        if "PUBCHEM_CID" not in df.columns:
            continue
        df = df[pd.to_numeric(df.PUBCHEM_CID, errors="coerce").notna()].copy()
        df["PUBCHEM_CID"] = df.PUBCHEM_CID.astype(int)
        name = pubchem_assay_name(aid)
        is_tox_assay = bool(CYTOTOX_DESC.search(name))
        for col in df.columns:
            kind = None
            if COL_RE["CC50"].search(col) or (is_tox_assay and COL_RE["IC50"].search(col)):
                kind = "CC50"
            elif COL_RE["IC50"].search(col):
                kind = "IC50"
            elif COL_RE["EC50"].search(col):
                kind = "EC50"
            if not kind:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            unit = pubchem_unit_from_col(col)
            for cid, v in zip(df.PUBCHEM_CID, vals):
                if pd.notna(v) and v > 0:
                    rows.append(dict(aid=aid, cid=cid, kind=kind, value=v, unit=unit, name=name))
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    smi = cid_to_smiles(raw.cid.unique())
    raw["smiles"] = raw.cid.map(smi)
    raw["val_uM"] = [to_uM(v, u, s) for v, u, s in zip(raw.value, raw.unit, raw.smiles)]
    raw = raw.dropna(subset=["val_uM", "smiles"])

    dois = {a: pubchem_doi(a) for a in tqdm(raw.aid.unique(), desc="[PubChem] DOI")}
    raw["doi"] = raw.aid.map(dois)

    # Антивирусная и цитотоксичная части часто лежат в разных AID одной серии,
    # поэтому CC50 для молекулы берём по всем найденным тестам (медиана).
    out = []
    for cid, g in raw.groupby("cid"):
        a = g[g.kind.isin(["IC50", "EC50"])]
        if a.empty:
            continue
        a_ic = a[a.kind == "IC50"]
        a_used = a_ic if len(a_ic) else a
        c = g[g.kind == "CC50"]
        out.append(dict(
            smiles=g.smiles.iloc[0],
            IC50_uM=a_used.val_uM.median(),
            CC50_uM=c.val_uM.median() if len(c) else np.nan,
            doi=a_used.doi.dropna().iloc[0] if a_used.doi.notna().any() else None,
            source="PubChem",
            source_id=f"CID{cid}|AID{','.join(map(str, sorted(a_used.aid.unique())))}",
            antiviral_type=a_used.kind.iloc[0],
            description=" ; ".join(a_used.name.unique())[:300]))
    return pd.DataFrame(out)



def compute_si(ic50, cc50):
    if pd.isna(ic50):
        return 10000.0
    if pd.isna(cc50):
        return 0.0
    return cc50 / ic50


def validate(df, train_path=None, test_path=None):
    n0 = len(df)
    df = df.copy()
    df["smiles"] = df.smiles.map(canon)
    df = df.dropna(subset=["smiles"])
    print(f"[valid] парсится RDKit: {len(df)}/{n0}")

    df = df[(df.IC50_uM.isna() | (df.IC50_uM > 0)) & (df.CC50_uM.isna() | (df.CC50_uM > 0))]
    # физически нереалистичные значения (ошибки единиц): >1e5 мкМ или <1e-6 мкМ
    for c in ("IC50_uM", "CC50_uM"):
        df = df[df[c].isna() | df[c].between(1e-6, 1e5)]

    df["SI"] = [compute_si(i, c) for i, c in zip(df.IC50_uM, df.CC50_uM)]
    df["_key"] = df.smiles.map(parent_key)

    # убрать молекулы из test.csv — иначе утечка и некорректная проверка
    if test_path and Path(test_path).exists():
        test_keys = set(pd.read_csv(test_path).SMILES.map(parent_key))
        before = len(df)
        df = df[~df._key.isin(test_keys)]
        print(f"[valid] удалено пересечений с test: {before - len(df)}")

    # дубликаты: одна молекула + один источник/doi -> медиана в логарифмической шкале
    def agg(g):
        r = g.iloc[0].copy()
        for c in ("IC50_uM", "CC50_uM"):
            v = g[c].dropna()
            r[c] = float(np.exp(np.log(v).median())) if len(v) else np.nan
        r["SI"] = compute_si(r["IC50_uM"], r["CC50_uM"])
        return r
    df["_doi"] = df.doi.fillna(df.source_id)
    df = df.groupby(["_key", "_doi"], as_index=False, group_keys=False).apply(agg)

    # доля молекул, уже присутствующих в train 
    if train_path and Path(train_path).exists():
        tr_keys = set(pd.read_csv(train_path).SMILES.map(parent_key))
        print(f"[valid] молекул, уже есть в train: {df._key.isin(tr_keys).sum()}")

    cols = ["smiles", "IC50_uM", "CC50_uM", "SI", "doi", "source", "source_id",
            "antiviral_type", "description"]
    return df[cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="vacv_extra.csv")
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--skip-chembl", action="store_true")
    ap.add_argument("--skip-pubchem", action="store_true")
    args = ap.parse_args()

    if not HAS_RDKIT:
        print("RDKit не установлен — валидация SMILES и пересчёт мг/мл нет")

    parts = []
    if not args.skip_chembl:
        parts.append(collect_chembl())
    if not args.skip_pubchem:
        parts.append(collect_pubchem())
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        raise SystemExit("Ничего не собрано — проверьте доступ в интернет.")

    df = validate(pd.concat(parts, ignore_index=True), args.train, args.test)
    df.to_csv(args.out, index=False)
    print(f"\nГотово: {len(df)} записей: {args.out}")
    print(df.groupby("source").size())
    print(f"с DOI: {df.doi.notna().sum()}, с IC50: {df.IC50_uM.notna().sum()}, "
          f"с CC50: {df.CC50_uM.notna().sum()}")


if __name__ == "__main__":
    main()
