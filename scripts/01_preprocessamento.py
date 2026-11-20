#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 1: Carregamento dos 8 CSVs oficiais do CICIDS2017, remoção da coluna
duplicada (Fwd Header Length.1), exclusão de registros com NaN/Infinity
e filtragem das quatro classes do estudo.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd

from utils import (
    AUDITORIA_DIR,
    DADOS_DIR,
    DADOS_PROCESSADOS_DIR,
    garantir_diretorios,
)

ARQUIVOS_OFICIAIS_CICIDS2017 = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]


def localizar_csvs_oficiais(diretorio_dados: Path) -> list[Path]:
    """
    Localiza e valida exatamente os 8 arquivos CSV oficiais do CICIDS2017
    em dados/ (incluindo subpastas como MachineLearningCVE), ignorando
    outros CSVs eventualmente presentes na pasta.
    """
    encontrados: dict[str, list[Path]] = {nome: [] for nome in ARQUIVOS_OFICIAIS_CICIDS2017}
    ignorados: list[Path] = []

    for caminho in sorted(diretorio_dados.rglob("*.csv")):
        if DADOS_PROCESSADOS_DIR in caminho.parents:
            continue
        if caminho.name in encontrados:
            encontrados[caminho.name].append(caminho)
        else:
            ignorados.append(caminho)

    if ignorados:
        print(
            f"[AVISO] {len(ignorados)} arquivo(s) CSV não oficial(is) ignorado(s): "
            + ", ".join(p.name for p in ignorados)
        )

    duplicados = {nome: paths for nome, paths in encontrados.items() if len(paths) > 1}
    if duplicados:
        detalhes = "\n".join(
            f" - {nome}: {[str(p) for p in paths]}" for nome, paths in duplicados.items()
        )
        raise ValueError(
            "Foram encontradas múltiplas cópias de arquivos oficiais do CICIDS2017:\n"
            f"{detalhes}"
        )

    faltando = [nome for nome, paths in encontrados.items() if not paths]
    if faltando:
        lista_faltando = "\n".join(f" - {nome}" for nome in faltando)
        raise FileNotFoundError(
            f"Faltam {len(faltando)} de 8 arquivos oficiais do CICIDS2017 em {diretorio_dados}:\n"
            f"{lista_faltando}\n"
            "Consulte dados/README.md para instruções de download."
        )

    return [encontrados[nome][0] for nome in ARQUIVOS_OFICIAIS_CICIDS2017]


def normalizar_rotulo(label: str) -> str:
    """Padroniza nomes das classes originais corrigindo variações de codificação."""
    label = str(label).strip()
    if "Web Attack" in label:
        if "Brute Force" in label:
            return "Web Attack Brute Force"
        if "XSS" in label:
            return "Web Attack XSS"
        if "Sql Injection" in label or "SQL Injection" in label:
            return "Web Attack SQL Injection"
    return label


def main() -> None:
    inicio_total = time.time()
    garantir_diretorios()

    arquivos_csv = localizar_csvs_oficiais(DADOS_DIR)
    print(f"Validados os {len(arquivos_csv)} arquivos CSV oficiais em {DADOS_DIR}...")
    lista_dfs = []
    resumo_arquivos = []

    for caminho_csv in arquivos_csv:
        t0 = time.time()
        # O arquivo de WebAttacks contém travessões codificados em cp1252
        df_temp = pd.read_csv(caminho_csv, low_memory=False, encoding="cp1252")
        df_temp.columns = df_temp.columns.str.strip()

        classes_presentes = df_temp["Label"].dropna().apply(normalizar_rotulo).unique()
        resumo_arquivos.append({
            "Arquivo": caminho_csv.name,
            "Linhas": len(df_temp),
            "Colunas": df_temp.shape[1],
            "Tempo_s": round(time.time() - t0, 2),
            "Classes": ", ".join(classes_presentes),
        })
        print(f"[OK] {caminho_csv.name:<58} | {len(df_temp):>9,} linhas")
        lista_dfs.append(df_temp)

    df = pd.concat(lista_dfs, ignore_index=True)
    del lista_dfs

    if df["Label"].isna().any():
        df = df.dropna(subset=["Label"]).reset_index(drop=True)
    df["Label"] = df["Label"].apply(normalizar_rotulo)
    print(f"Total consolidado: {df.shape[0]:,} linhas x {df.shape[1]} colunas")

    # Remove coluna duplicada presente nos CSVs originais do CICIDS2017
    if "Fwd Header Length.1" in df.columns:
        identicas = (df["Fwd Header Length"] == df["Fwd Header Length.1"]).all()
        print(f"Fwd Header Length.1 idêntica a Fwd Header Length: {identicas}")
        df = df.drop(columns=["Fwd Header Length.1"])

    colunas_features = [c for c in df.columns if c != "Label"]
    for col in colunas_features:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    matriz_valores = df[colunas_features].to_numpy(dtype=np.float64)
    mask_nan = np.isnan(matriz_valores).any(axis=1)
    mask_inf = np.isinf(matriz_valores).any(axis=1)
    mask_ruido = mask_nan | mask_inf

    print(
        f"Ruídos identificados -> NaN: {mask_nan.sum():,} | "
        f"Infinity: {mask_inf.sum():,} | Total (união): {mask_ruido.sum():,}"
    )

    df["is_nan"] = mask_nan
    df["is_inf"] = mask_inf
    df["is_ruido"] = mask_ruido

    tabela_15 = (
        df.groupby("Label", as_index=False)
        .agg(
            Amostras_Originais=("Label", "count"),
            Ruido_NaN=("is_nan", "sum"),
            Ruido_Inf=("is_inf", "sum"),
            Amostras_Ruido=("is_ruido", "sum"),
        )
        .sort_values(by="Amostras_Originais", ascending=False)
        .reset_index(drop=True)
    )
    tabela_15["Amostras_Finais"] = tabela_15["Amostras_Originais"] - tabela_15["Amostras_Ruido"]
    total_finais_15 = tabela_15["Amostras_Finais"].sum()
    tabela_15["Perc_Dataset (%)"] = (tabela_15["Amostras_Finais"] / total_finais_15 * 100).round(4)

    linha_total_15 = pd.DataFrame([{
        "Label": "TOTAL",
        "Amostras_Originais": tabela_15["Amostras_Originais"].sum(),
        "Ruido_NaN": tabela_15["Ruido_NaN"].sum(),
        "Ruido_Inf": tabela_15["Ruido_Inf"].sum(),
        "Amostras_Ruido": tabela_15["Amostras_Ruido"].sum(),
        "Amostras_Finais": total_finais_15,
        "Perc_Dataset (%)": 100.0000,
    }])
    tabela_15_completa = pd.concat([tabela_15, linha_total_15], ignore_index=True)

    mapa_escopo = {
        "BENIGN": "Benign",
        "DDoS": "DDoS",
        "PortScan": "PortScan",
        "FTP-Patator": "Brute Force",
        "SSH-Patator": "Brute Force",
    }
    ordem_tabela = ["Benign", "PortScan", "DDoS", "Brute Force"]

    df["Classe_TCC"] = df["Label"].map(mapa_escopo)
    df_escopo_bruto = df.dropna(subset=["Classe_TCC"])

    tabela_4 = (
        df_escopo_bruto.groupby("Classe_TCC", as_index=False)
        .agg(
            Amostras_Originais=("Classe_TCC", "count"),
            Ruido_Removido=("is_ruido", "sum"),
        )
    )
    tabela_4["Amostras_Finais"] = tabela_4["Amostras_Originais"] - tabela_4["Ruido_Removido"]
    tabela_4["Classe_TCC"] = pd.Categorical(
        tabela_4["Classe_TCC"], categories=ordem_tabela, ordered=True
    )
    tabela_4 = tabela_4.sort_values("Classe_TCC").reset_index(drop=True)
    total_finais_4 = tabela_4["Amostras_Finais"].sum()
    tabela_4["Perc_Escopo (%)"] = (tabela_4["Amostras_Finais"] / total_finais_4 * 100).round(2)

    linha_total_4 = pd.DataFrame([{
        "Classe_TCC": "TOTAL (Escopo TCC)",
        "Amostras_Originais": tabela_4["Amostras_Originais"].sum(),
        "Ruido_Removido": tabela_4["Ruido_Removido"].sum(),
        "Amostras_Finais": total_finais_4,
        "Perc_Escopo (%)": 100.00,
    }])
    tabela_4_completa = pd.concat([tabela_4, linha_total_4], ignore_index=True)

    df_limpo = (
        df.loc[~df["is_ruido"], colunas_features + ["Classe_TCC"]]
        .dropna(subset=["Classe_TCC"])
        .rename(columns={"Classe_TCC": "Label"})
        .reset_index(drop=True)
    )

    caminho_parquet = DADOS_PROCESSADOS_DIR / "cicids2017_4classes_limpo.parquet"
    df_limpo.to_parquet(caminho_parquet, index=False)

    pd.DataFrame(resumo_arquivos).to_csv(AUDITORIA_DIR / "resumo_arquivos_csv.csv", index=False)
    tabela_15_completa.to_csv(AUDITORIA_DIR / "tabela_limpeza_15_classes.csv", index=False)
    tabela_4_completa.to_csv(AUDITORIA_DIR / "tabela_limpeza_4_classes.csv", index=False)

    print("\nResumo do escopo (4 classes):")
    print(tabela_4_completa.to_string(index=False))
    print(f"\nParquet salvo em: {caminho_parquet} ({df_limpo.shape[0]:,} x {df_limpo.shape[1]})")
    print(f"Tempo total: {time.time() - inicio_total:.2f}s")


if __name__ == "__main__":
    main()
