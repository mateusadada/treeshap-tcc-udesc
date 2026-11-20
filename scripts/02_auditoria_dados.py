#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 2: Auditoria estrutural da base limpa (colunas constantes, pares
numericamente idênticos, duplicatas e conflitos de rótulos em float64 vs.
float32, sobreposição no StratifiedKFold vs. StratifiedGroupKFold e teste de
seleção MDI restrito ao treino).
"""

import time
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from utils import (
    AUDITORIA_DIR,
    DADOS_PROCESSADOS_DIR,
    MAPA_CLASSES,
    NOMES_CLASSES,
    agrupar_vetores_exatos,
    garantir_diretorios,
)


def encontrar_colunas_constantes_e_identicas(df: pd.DataFrame, features: list):
    """Identifica colunas constantes e pares idênticos usando filtro por prefixo."""
    colunas_constantes = []
    buckets = defaultdict(list)

    for col in features:
        arr_prefix = df[col].iloc[:2048].to_numpy(dtype=np.float64, copy=True)
        arr_prefix[arr_prefix == 0.0] = 0.0
        col_min = float(df[col].min())
        col_max = float(df[col].max())

        if col_min == col_max:
            colunas_constantes.append((col, col_min))

        assinatura = (col_min, col_max, arr_prefix.tobytes())
        buckets[assinatura].append(col)

    pares_identicos = []
    for cols_candidatas in buckets.values():
        if len(cols_candidatas) > 1:
            for i in range(len(cols_candidatas)):
                arr_i = df[cols_candidatas[i]].to_numpy(dtype=np.float64, copy=True)
                arr_i[arr_i == 0.0] = 0.0
                for j in range(i + 1, len(cols_candidatas)):
                    arr_j = df[cols_candidatas[j]].to_numpy(dtype=np.float64, copy=True)
                    arr_j[arr_j == 0.0] = 0.0
                    if np.array_equal(arr_i, arr_j):
                        pares_identicos.append((cols_candidatas[i], cols_candidatas[j]))

    return colunas_constantes, pares_identicos


def analisar_grupos_e_conflitos(group_ids: np.ndarray, y: np.ndarray, n_grupos: int):
    """Separa grupos de vetores em únicos, duplicatas puras e conflitos de rótulo."""
    n_classes = len(NOMES_CLASSES)
    mat_gc = np.zeros((n_grupos, n_classes), dtype=np.int32)
    for c in range(n_classes):
        mat_gc[:, c] = np.bincount(group_ids[y == c], minlength=n_grupos)

    total_por_grupo = mat_gc.sum(axis=1)
    labels_distintos = (mat_gc > 0).sum(axis=1)

    mask_unico = (total_por_grupo == 1) & (labels_distintos == 1)
    mask_dup_puro = (total_por_grupo > 1) & (labels_distintos == 1)
    mask_conflito = labels_distintos > 1

    resumo_geral = {
        "Total_Registros": int(len(y)),
        "Total_Grupos_X_Distintos": int(n_grupos),
        "Grupos_Estritamente_Unicos": int(mask_unico.sum()),
        "Registros_Estritamente_Unicos": int(total_por_grupo[mask_unico].sum()),
        "Grupos_Duplicatas_Puras": int(mask_dup_puro.sum()),
        "Registros_Em_Duplicatas_Puras": int(total_por_grupo[mask_dup_puro].sum()),
        "Copias_Excedentes_Puras": int((total_por_grupo[mask_dup_puro] - 1).sum()),
        "Grupos_Com_Conflito_Rotulo": int(mask_conflito.sum()),
        "Registros_Em_Grupos_Conflitantes": int(total_por_grupo[mask_conflito].sum()),
        "Registros_Majoritarios_No_Conflito": (
            int(mat_gc[mask_conflito].max(axis=1).sum()) if mask_conflito.any() else 0
        ),
        "Registros_Minoritarios_Discordantes": (
            int(total_por_grupo[mask_conflito].sum() - mat_gc[mask_conflito].max(axis=1).sum())
            if mask_conflito.any()
            else 0
        ),
    }

    linhas_classe = []
    for c, nome_c in enumerate(NOMES_CLASSES):
        tot_c = int((y == c).sum())
        unicos_c = int(mat_gc[mask_unico, c].sum())
        grupos_puros_c = int((mask_dup_puro & (mat_gc[:, c] > 0)).sum())
        regs_puros_c = int(mat_gc[mask_dup_puro, c].sum())
        excedentes_puros_c = regs_puros_c - grupos_puros_c
        regs_conflito_c = int(mat_gc[mask_conflito, c].sum())
        unicos_x_label_c = unicos_c + grupos_puros_c + int((mask_conflito & (mat_gc[:, c] > 0)).sum())

        linhas_classe.append({
            "Classe": nome_c,
            "Total_Limpo": tot_c,
            "Unicos_Estritos": unicos_c,
            "Grupos_Dup_Pura": grupos_puros_c,
            "Regs_Em_Dup_Pura": regs_puros_c,
            "Copias_Excedentes_Puras": excedentes_puros_c,
            "Regs_Em_Conflito_Rotulo": regs_conflito_c,
            "Padroes_Distintos_(X_Label)": unicos_x_label_c,
            "Taxa_Redundancia (%)": round((tot_c - unicos_x_label_c) / tot_c * 100, 2),
        })
    df_por_classe = pd.DataFrame(linhas_classe)

    linhas_conflito = []
    if mask_conflito.any():
        mat_conf = mat_gc[mask_conflito]
        padroes_bool = mat_conf > 0
        for padrao in np.unique(padroes_bool, axis=0):
            idx_p = np.all(padroes_bool == padrao, axis=1)
            sub_mat = mat_conf[idx_p]
            classes_envolvidas = " vs. ".join(
                [NOMES_CLASSES[i] for i in range(n_classes) if padrao[i]]
            )
            tot_regs_p = int(sub_mat.sum())
            minoritarios_p = int(tot_regs_p - sub_mat.max(axis=1).sum())
            registro = {
                "Combinacao_Conflito": classes_envolvidas,
                "Grupos_X_Afetados": int(idx_p.sum()),
                "Total_Registros_Afetados": tot_regs_p,
                "Registros_Minoritarios": minoritarios_p,
            }
            for i, nome_c in enumerate(NOMES_CLASSES):
                registro[f"Regs_{nome_c}"] = int(sub_mat[:, i].sum())
            linhas_conflito.append(registro)

    df_conflitos = pd.DataFrame(linhas_conflito)
    if not df_conflitos.empty:
        df_conflitos = df_conflitos.sort_values(
            by="Total_Registros_Afetados", ascending=False
        ).reset_index(drop=True)

    return resumo_geral, df_por_classe, df_conflitos


def main() -> None:
    t0 = time.time()
    garantir_diretorios()

    caminho_parquet = DADOS_PROCESSADOS_DIR / "cicids2017_4classes_limpo.parquet"
    if not caminho_parquet.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho_parquet}. Execute 01_preprocessamento.py primeiro."
        )

    df = pd.read_parquet(caminho_parquet)
    features = [c for c in df.columns if c != "Label"]
    y = df["Label"].map(MAPA_CLASSES).to_numpy(dtype=np.int32)
    print(f"Base carregada: {len(df):,} linhas x {len(features)} atributos")

    cols_const, pares_id = encontrar_colunas_constantes_e_identicas(df, features)
    nomes_const = {c[0] for c in cols_const}
    print(f"\nColunas constantes [{len(cols_const)}]: {[c[0] for c in cols_const]}")
    print(f"Pares numericamente idênticos encontrados: {len(pares_id)}")
    for c1, c2 in pares_id:
        if c1 not in nomes_const:
            print(f"  * {c1} == {c2}")

    X_f64 = df[features].to_numpy(dtype=np.float64, copy=True)
    g_ids_f64, n_grupos_f64 = agrupar_vetores_exatos(X_f64)
    res_f64, _, _ = analisar_grupos_e_conflitos(g_ids_f64, y, n_grupos_f64)
    del X_f64

    X_f32 = df[features].to_numpy(dtype=np.float32, copy=True)
    g_ids_f32, n_grupos_f32 = agrupar_vetores_exatos(X_f32)
    res_f32, df_classe_f32, df_conf_f32 = analisar_grupos_e_conflitos(
        g_ids_f32, y, n_grupos_f32
    )

    df_comp_precisao = pd.DataFrame([
        {
            "Metrica": k,
            "Float64 (Parquet)": res_f64[k],
            "Float32 (Arvores)": res_f32[k],
            "Diferenca": res_f32[k] - res_f64[k],
        }
        for k in res_f64
    ])

    print("\nComparação float64 vs. float32:")
    print(df_comp_precisao.to_string(index=False))
    print("\nDuplicatas por classe (float32):")
    print(df_classe_f32.to_string(index=False))
    if not df_conf_f32.empty:
        print("\nConflitos de rótulo (float32):")
        print(df_conf_f32.to_string(index=False))

    # Sobreposição treino-teste no StratifiedKFold padrão (Fold 1)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf.split(X_f32, y))
    grupos_treino = set(g_ids_f32[train_idx])
    mask_visto = np.isin(g_ids_f32[test_idx], list(grupos_treino))
    print(
        f"\nStratifiedKFold (Fold 1): {mask_visto.sum():,}/{len(test_idx):,} "
        f"({mask_visto.mean() * 100:.2f}%) registros de teste com vetor X idêntico no treino."
    )

    # Verificação de balanceamento no StratifiedGroupKFold (10 partições)
    sgkf = StratifiedGroupKFold(n_splits=10)
    stats_sgkf = []
    for fold_id, (tr_g, te_g) in enumerate(sgkf.split(X_f32, y, groups=g_ids_f32), start=1):
        vazados = len(set(g_ids_f32[tr_g]).intersection(set(g_ids_f32[te_g])))
        cont = np.bincount(y[te_g], minlength=4)
        stats_sgkf.append({
            "Fold": fold_id,
            "Total_Teste": len(te_g),
            "Grupos_Vazados": vazados,
            "Benign": cont[0],
            "DDoS": cont[1],
            "PortScan": cont[2],
            "Brute_Force": cont[3],
            "Perc_PortScan (%)": round(cont[2] / len(te_g) * 100, 2),
        })
    df_sgkf = pd.DataFrame(stats_sgkf)
    print("\nDistribuição StratifiedGroupKFold (10 folds):")
    print(df_sgkf.to_string(index=False))

    # Teste exploratório de seleção MDI restrita ao treino do Fold 1
    rf_mdi_tr = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_mdi_tr.fit(X_f32[train_idx], y[train_idx])
    imp = rf_mdi_tr.feature_importances_
    cumsum = np.cumsum(np.sort(imp)[::-1])
    n_corte_99 = int(np.searchsorted(cumsum, 0.99) + 1)
    print(
        f"\nMDI no treino do Fold 1 -> zeros: {int((imp == 0.0).sum())} | "
        f"atributos para >= 99%: {n_corte_99} ({cumsum[n_corte_99 - 1] * 100:.4f}%)"
    )

    df_comp_precisao.to_csv(AUDITORIA_DIR / "comparacao_float64_vs_float32.csv", index=False)
    df_classe_f32.to_csv(AUDITORIA_DIR / "duplicatas_por_classe_float32.csv", index=False)
    df_conf_f32.to_csv(AUDITORIA_DIR / "conflitos_rotulo_float32.csv", index=False)
    df_sgkf.to_csv(AUDITORIA_DIR / "distribuicao_stratified_group_kfold.csv", index=False)
    print(f"\nRelatórios salvos em: {AUDITORIA_DIR} ({time.time() - t0:.2f}s)")


if __name__ == "__main__":
    main()
