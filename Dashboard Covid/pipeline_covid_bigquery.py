"""
pipeline_covid_bigquery.py
==========================

Pipeline de extração, limpeza, validação e exportação de dados de COVID-19.

FLUXO
-----
1. Configuração e autenticação  -> cria o cliente BigQuery (ou usa o CSV local como fallback)
2. Extração                     -> Query A (top 20 países) e Query B (série temporal do Brasil)
3. Limpeza e padronização       -> nomes de colunas, tipos, texto
4. Validação mínima             -> schema, tipos, nulos, duplicidades, consistência
5. Exportação                   -> output/csv/*.csv e output/parquet/*.parquet
6. Relatório                    -> output/validation_report.txt

DEPENDÊNCIAS
------------
    pip install google-cloud-bigquery db-dtypes pandas pyarrow

AUTENTICAÇÃO (uma vez)
----------------------
    gcloud auth application-default login
    gcloud config set project ebac1-487019
Permissões necessárias no projeto: roles/bigquery.jobUser e roles/bigquery.dataViewer
(no dataset db_covid19).

PARÂMETROS (linha de comando; todos opcionais)
----------------------------------------------
    --project     ID do projeto no Google Cloud           (padrão: ebac1-487019)
    --table       tabela de origem  projeto.dataset.tabela (padrão: ver TABLE abaixo)
    --country     país da série temporal                  (padrão: Brazil)
    --top-n       quantidade de países no ranking         (padrão: 20)
    --location    região do dataset, ex.: US ou EU        (padrão: None)
    --output-dir  pasta raiz de saída                     (padrão: output)
    --source      auto | bigquery | csv                   (padrão: auto)
    --local-csv   CSV usado quando source=csv ou no fallback (padrão: db_covid19.csv)

EXEMPLOS
--------
    python pipeline_covid_bigquery.py
    python pipeline_covid_bigquery.py --country Italy --top-n 10
    python pipeline_covid_bigquery.py --source csv --local-csv data/raw/db_covid19.csv

REUTILIZAÇÃO
------------
- Troque --table / --country / --top-n para outras tabelas e países.
- `clean_*` e `validate_*` são funções puras (DataFrame in, DataFrame/resultado out) e podem
  ser importadas em outros projetos.
- Para nova métrica, acrescente a coluna no SQL e em EXPECTED_SCHEMA.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd

# ======================================================================
# CONFIGURAÇÕES PADRÃO
# ======================================================================
DEFAULT_PROJECT = "ebac1-487019"
DEFAULT_TABLE = "ebac1-487019.db_covid19.covid_19_geographic_distribution_worldwide"

# Schema esperado após a limpeza: coluna -> dtype pandas
SCHEMA_TOP = {
    "country_region": "object",
    "total_confirmed": "int64",
    "total_deaths": "int64",
    "max_confirmed_accumulated": "int64",
}
SCHEMA_TS = {
    "date": "datetime64",
    "country_region": "object",
    "confirmed": "int64",
    "deaths": "int64",
    "confirmed_accumulated": "int64",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pipeline_covid")


# ======================================================================
# 1. EXTRAÇÃO
# ======================================================================
def sql_top_countries(table: str, top_n: int) -> str:
    """Query A: ranking dos países por casos confirmados.

    - SUM(daily_*): total de casos/mortes no período (soma dos valores diários).
    - MAX(confirmed_cases): acumulado final. Usa MAX porque `confirmed_cases` já é
      cumulativo; somá-lo (como na aula 3) não tem significado.
    """
    return f"""
    SELECT
        countries_and_territories                    AS country_region,
        SUM(CAST(daily_confirmed_cases AS INT64))    AS total_confirmed,
        SUM(CAST(daily_deaths AS INT64))             AS total_deaths,
        MAX(CAST(confirmed_cases AS INT64))          AS max_confirmed_accumulated
    FROM `{table}`
    GROUP BY country_region
    ORDER BY total_confirmed DESC
    LIMIT {int(top_n)}
    """


def sql_timeseries(table: str, country: str) -> str:
    """Query B (a 'Query 3' das aulas): série temporal diária filtrada por país."""
    safe_country = country.replace("'", "\\'")
    return f"""
    SELECT
        date,
        countries_and_territories                    AS country_region,
        SUM(CAST(daily_confirmed_cases AS INT64))    AS confirmed,
        SUM(CAST(daily_deaths AS INT64))             AS deaths,
        MAX(CAST(confirmed_cases AS INT64))          AS confirmed_accumulated
    FROM `{table}`
    WHERE countries_and_territories = '{safe_country}'
    GROUP BY date, country_region
    ORDER BY date
    """


def create_bq_client(project: str):
    """Cria e testa o cliente BigQuery (projeto, credenciais e permissão de job)."""
    from google.cloud import bigquery  # import tardio: permite rodar o fallback sem a lib

    client = bigquery.Client(project=project)
    client.query("SELECT 1").result()  # falha aqui se faltar credencial/permissão
    logger.info("BigQuery OK | project=%s | credenciais=%s", client.project, type(client._credentials).__name__)
    return client


def extract_from_bigquery(args):
    client = create_bq_client(args.project)
    kw = {"location": args.location} if args.location else {}
    df_top = client.query(sql_top_countries(args.table, args.top_n), **kw).result().to_dataframe()
    df_ts = client.query(sql_timeseries(args.table, args.country), **kw).result().to_dataframe()
    return df_top, df_ts


def extract_from_csv(args):
    """Fallback: reproduz as mesmas queries sobre o CSV local usando pandas."""
    if not os.path.exists(args.local_csv):
        raise FileNotFoundError(f"CSV local não encontrado: {args.local_csv}")
    raw = pd.read_csv(args.local_csv, low_memory=False)
    raw = raw.rename(columns={"countries_and_territories": "country_region"})

    df_top = (
        raw.groupby("country_region", as_index=False)
        .agg(total_confirmed=("daily_confirmed_cases", "sum"),
             total_deaths=("daily_deaths", "sum"),
             max_confirmed_accumulated=("confirmed_cases", "max"))
        .sort_values("total_confirmed", ascending=False)
        .head(args.top_n)
        .reset_index(drop=True)
    )
    df_ts = (
        raw[raw["country_region"] == args.country]
        .groupby(["date", "country_region"], as_index=False)
        .agg(confirmed=("daily_confirmed_cases", "sum"),
             deaths=("daily_deaths", "sum"),
             confirmed_accumulated=("confirmed_cases", "max"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df_top, df_ts


def extract(args):
    """Tenta o BigQuery; em modo 'auto', cai para o CSV local se falhar."""
    if args.source == "csv":
        logger.info("Fonte: CSV local (%s)", args.local_csv)
        return extract_from_csv(args), "csv_local"
    try:
        logger.info("Fonte: BigQuery (%s)", args.table)
        return extract_from_bigquery(args), "bigquery"
    except Exception as exc:
        if args.source == "bigquery":
            raise
        logger.warning("BigQuery indisponível (%s: %s). Usando CSV local.", type(exc).__name__, exc)
        return extract_from_csv(args), "csv_local"


# ======================================================================
# 2. LIMPEZA E PADRONIZAÇÃO
# ======================================================================
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _to_int(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    return df


def clean_top(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_columns(df)
    df["country_region"] = df["country_region"].astype("string").str.strip().astype("object")
    df = _to_int(df, ["total_confirmed", "total_deaths", "max_confirmed_accumulated"])
    return df.sort_values("total_confirmed", ascending=False).reset_index(drop=True)


def clean_ts(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_columns(df)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["country_region"] = df["country_region"].astype("string").str.strip().astype("object")
    df = _to_int(df, ["confirmed", "deaths", "confirmed_accumulated"])
    return df.sort_values("date").reset_index(drop=True)


# ======================================================================
# 3. VALIDAÇÃO
# ======================================================================
class Report:
    """Acumula o resultado das validações (OK / ALERTA / FALHA)."""

    def __init__(self):
        self.lines, self.failures = [], 0

    def add(self, status, name, detail=""):
        if status == "FALHA":
            self.failures += 1
        line = f"[{status:<6}] {name}" + (f" -> {detail}" if detail else "")
        self.lines.append(line)
        logger.info(line)


def _check_schema(df, expected, label, rep):
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    rep.add("OK" if not missing and not extra else "FALHA", f"{label}: colunas",
            f"faltando={missing} extras={extra}" if missing or extra else f"{len(df.columns)} colunas esperadas")
    for col, dtype in expected.items():
        if col not in df.columns:
            continue
        ok = str(df[col].dtype).startswith(dtype) if dtype == "datetime64" else str(df[col].dtype) == dtype
        rep.add("OK" if ok else "FALHA", f"{label}: tipo de '{col}'", f"esperado={dtype} obtido={df[col].dtype}")


def validate_top(df, top_n, rep):
    L = "top_paises"
    _check_schema(df, SCHEMA_TOP, L, rep)
    rep.add("OK" if len(df) == top_n else "FALHA", f"{L}: quantidade de linhas", f"{len(df)} (esperado {top_n})")
    n = int(df.isna().sum().sum())
    rep.add("OK" if n == 0 else "FALHA", f"{L}: valores nulos", str(n))
    d = int(df["country_region"].duplicated().sum())
    rep.add("OK" if d == 0 else "FALHA", f"{L}: países duplicados", str(d))
    rep.add("OK" if df["total_confirmed"].is_monotonic_decreasing else "FALHA", f"{L}: ordenado por casos (desc)")
    neg = int((df[["total_confirmed", "total_deaths"]] < 0).sum().sum())
    rep.add("OK" if neg == 0 else "FALHA", f"{L}: totais negativos", str(neg))
    bad = int((df["total_deaths"] > df["total_confirmed"]).sum())
    rep.add("OK" if bad == 0 else "FALHA", f"{L}: mortes > casos", str(bad))
    # Consistência: soma dos casos diários deve igualar o acumulado final.
    diff = (df["total_confirmed"] - df["max_confirmed_accumulated"]).abs()
    rep.add("OK" if (diff == 0).all() else "ALERTA", f"{L}: soma diária == acumulado final",
            f"{int((diff != 0).sum())} país(es) divergente(s)")


def validate_ts(df, country, rep):
    L = f"serie_{country.lower()}"
    _check_schema(df, SCHEMA_TS, L, rep)
    rep.add("OK" if len(df) > 0 else "FALHA", f"{L}: linhas", str(len(df)))
    n = df.isna().sum()
    rep.add("OK" if n.sum() == 0 else "FALHA", f"{L}: valores nulos", str(n[n > 0].to_dict() or 0))
    d = int(df["date"].duplicated().sum())
    rep.add("OK" if d == 0 else "FALHA", f"{L}: datas duplicadas", str(d))
    rep.add("OK" if (df["country_region"] == country).all() else "FALHA", f"{L}: filtro de país aplicado")
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    gaps = len(full.difference(df["date"]))
    rep.add("OK" if gaps == 0 else "ALERTA", f"{L}: datas faltantes no intervalo", str(gaps))
    # Valores negativos são correções de reporte da fonte: mantidos e sinalizados.
    neg = int((df[["confirmed", "deaths"]] < 0).sum().sum())
    rep.add("OK" if neg == 0 else "ALERTA", f"{L}: valores diários negativos (correções da fonte, mantidos)", str(neg))
    rep.add("OK" if df["confirmed_accumulated"].is_monotonic_increasing else "ALERTA",
            f"{L}: acumulado não decrescente")
    diff = int(df["confirmed"].sum() - df["confirmed_accumulated"].max())
    rep.add("OK" if diff == 0 else "ALERTA", f"{L}: soma diária - acumulado final", str(diff))


def cross_validate(df_top, df_ts, country, rep):
    """Consistência entre as duas bases: o total do país no ranking == soma da série."""
    row = df_top[df_top["country_region"] == country]
    if row.empty:
        rep.add("ALERTA", f"cruzamento: {country} fora do top {len(df_top)}", "não comparável")
        return
    ok = int(row["total_confirmed"].iloc[0]) == int(df_ts["confirmed"].sum())
    rep.add("OK" if ok else "FALHA", f"cruzamento: total {country} no ranking == soma da série")


# ======================================================================
# 4. EXPORTAÇÃO
# ======================================================================
def export(df, name, out_dir, rep):
    csv_dir, pq_dir = os.path.join(out_dir, "csv"), os.path.join(out_dir, "parquet")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(pq_dir, exist_ok=True)

    out = df.copy()
    if "date" in out.columns:
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")  # datas ISO no CSV
    csv_path = os.path.join(csv_dir, f"{name}.csv")
    out.to_csv(csv_path, index=False, encoding="utf-8")
    rep.add("OK", f"exportação CSV: {name}", csv_path)

    pq_path = os.path.join(pq_dir, f"{name}.parquet")
    try:
        df.to_parquet(pq_path, index=False)
        back = pd.read_parquet(pq_path)
        same = list(back.columns) == list(df.columns) and len(back) == len(df)
        rep.add("OK" if same else "FALHA", f"exportação Parquet: {name}", pq_path)
    except ImportError:
        rep.add("FALHA", f"exportação Parquet: {name}", "instale pyarrow:  pip install pyarrow")


# ======================================================================
# MAIN
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Pipeline COVID-19: BigQuery -> DataFrame -> CSV/Parquet")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--table", default=DEFAULT_TABLE)
    p.add_argument("--country", default="Brazil")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--location", default=None)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--source", choices=["auto", "bigquery", "csv"], default="auto")
    p.add_argument("--local-csv", default="db_covid19.csv")
    return p.parse_args()


def main():
    args = parse_args()
    rep = Report()
    started = datetime.now(timezone.utc)

    (raw_top, raw_ts), source = extract(args)
    logger.info("Extração concluída (%s) | top=%d linhas | série=%d linhas", source, len(raw_top), len(raw_ts))

    print("\n=== Amostra bruta: top países ===")
    print(raw_top.head(5).to_string(index=False))
    print(f"\n=== Amostra bruta: série {args.country} ===")
    print(raw_ts.head(5).to_string(index=False))

    df_top, df_ts = clean_top(raw_top), clean_ts(raw_ts)

    validate_top(df_top, args.top_n, rep)
    validate_ts(df_ts, args.country, rep)
    cross_validate(df_top, df_ts, args.country, rep)

    export(df_top, "covid_top20_paises", args.output_dir, rep)
    export(df_ts, f"covid_serie_temporal_{args.country.lower()}", args.output_dir, rep)

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Pipeline COVID-19 | execução UTC: {started.isoformat(timespec='seconds')} | fonte: {source}\n")
        f.write(f"Parâmetros: {vars(args)}\n\n")
        f.write("\n".join(rep.lines) + "\n")
        f.write(f"\nResumo: {rep.failures} falha(s)\n")
    print(f"\nRelatório salvo em {report_path} | falhas: {rep.failures}")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
