#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 4: Análise de explicabilidade com TreeSHAP sobre os modelos do Fold 10
usando a mesma subamostra balanceada de 1.200 fluxos (300 por classe) para
Decision Tree e Random Forest, verificação de aditividade e estudos de caso
locais (três acertos e um falso negativo de PortScan).
"""

import time
import warnings
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from utils import (
    FIGURAS_DIR,
    METRICAS_DIR,
    MODELOS_DIR,
    NOMES_CLASSES,
    SHAP_DIR,
    garantir_diretorios,
)

warnings.filterwarnings("ignore")


def extrair_matriz_shap_3d(shap_vals, n_samples: int, n_features: int, n_classes: int) -> np.ndarray:
    """Padroniza a saída de TreeExplainer.shap_values para (n_samples, n_features, n_classes)."""
    if isinstance(shap_vals, list):
        return np.stack(shap_vals, axis=-1)
    arr = np.asarray(shap_vals)
    if arr.ndim == 3:
        if arr.shape == (n_samples, n_features, n_classes):
            return arr
        if arr.shape == (n_classes, n_samples, n_features):
            return np.transpose(arr, (1, 2, 0))
    raise ValueError(f"Formato inesperado retornado por shap_values: {arr.shape}")


def plotar_barras_empilhadas_global(
    df_shap: pd.DataFrame,
    titulo: str,
    caminho_saida: Path,
    top_n: int = 15,
) -> None:
    """Gera gráfico horizontal empilhado dos Top N atributos globais (300 DPI)."""
    top_df = df_shap.head(top_n).iloc[::-1].copy()
    cores = {
        "Benign": "#1f77b4",
        "DDoS": "#d62728",
        "PortScan": "#ff7f0e",
        "Brute Force": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(10.5, 7))
    acumulado = np.zeros(len(top_df))

    for classe in NOMES_CLASSES:
        valores = top_df[f"SHAP_{classe}"].to_numpy()
        ax.barh(
            top_df["Atributo"],
            valores,
            left=acumulado,
            label=classe,
            color=cores[classe],
            edgecolor="black",
            linewidth=0.4,
            height=0.72,
        )
        acumulado += valores

    for idx, total_val in enumerate(acumulado):
        ax.text(
            total_val + max(acumulado) * 0.01,
            idx,
            f"{total_val:.4f}",
            va="center",
            fontsize=8.5,
        )

    ax.set_xlim(0, max(acumulado) * 1.12)
    ax.set_xlabel("Impacto Médio Absoluto na Probabilidade Predita — mean(|SHAP|)", fontsize=10.5)
    ax.set_title(titulo, fontsize=11.5, fontweight="bold", pad=12)
    ax.legend(title="Classe", loc="lower right", frameon=True, fontsize=9.5)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plotar_top20_mdi(df_mdi: pd.DataFrame, caminho_saida: Path) -> None:
    """Gera o gráfico dos 20 principais atributos via MDI no treino do Fold 10."""
    top20 = df_mdi.head(20).iloc[::-1]
    plt.figure(figsize=(10, 7))
    bars = plt.barh(
        top20["Atributo"],
        top20["Importancia_Perc (%)"],
        color="#1f77b4",
        edgecolor="black",
    )
    plt.xlabel("Importância relativa MDI no conjunto de treinamento (%)", fontsize=11)
    plt.title("Top 20 atributos mais relevantes via MDI (Treino - Sem Vazamento)", fontsize=11.5, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        largura = bar.get_width()
        plt.text(
            largura + 0.12,
            bar.get_y() + bar.get_height() / 2,
            f"{largura:.2f}%",
            va="center",
            fontsize=9,
        )

    plt.xlim(0, top20["Importancia_Perc (%)"].max() * 1.15)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300)
    plt.close()


def formatar_valor_fisico(val: float) -> str:
    """Formata valores inteiros sem casas decimais e contínuos com 4 casas."""
    if abs(val - round(val)) < 1e-5:
        return f"{int(round(val))}"
    return f"{val:.4f}"


def construir_tabela_global(sv_3d: np.ndarray, features: list) -> pd.DataFrame:
    """Agrega o impacto médio absoluto por atributo e por classe."""
    mean_abs_por_classe = np.mean(np.abs(sv_3d), axis=0)
    soma_global = np.sum(mean_abs_por_classe, axis=1)
    df_res = pd.DataFrame({
        "Atributo": features,
        "SHAP_Global_Sum": soma_global,
        "SHAP_Global_Mean": np.mean(mean_abs_por_classe, axis=1),
        "SHAP_Benign": mean_abs_por_classe[:, 0],
        "SHAP_DDoS": mean_abs_por_classe[:, 1],
        "SHAP_PortScan": mean_abs_por_classe[:, 2],
        "SHAP_Brute Force": mean_abs_por_classe[:, 3],
    }).sort_values(by="SHAP_Global_Sum", ascending=False).reset_index(drop=True)

    df_res["Rank_SHAP"] = df_res.index + 1
    df_res["Perc_Impacto (%)"] = (
        df_res["SHAP_Global_Sum"] / df_res["SHAP_Global_Sum"].sum() * 100
    ).round(2)
    df_res["Perc_Acumulado (%)"] = df_res["Perc_Impacto (%)"].cumsum().round(2)

    cols = [
        "Rank_SHAP",
        "Atributo",
        "SHAP_Global_Sum",
        "Perc_Impacto (%)",
        "Perc_Acumulado (%)",
        "SHAP_Benign",
        "SHAP_DDoS",
        "SHAP_PortScan",
        "SHAP_Brute Force",
    ]
    return df_res[cols]


def main() -> None:
    t0 = time.time()
    garantir_diretorios()

    arquivos_obrigatorios = [
        MODELOS_DIR / "modelo_dt_mdi_fold10.joblib",
        MODELOS_DIR / "modelo_rf_mdi_fold10.joblib",
        MODELOS_DIR / "amostra_balanceada_1200_shap.parquet",
        MODELOS_DIR / "erros_teste_fold10_shap.parquet",
        METRICAS_DIR / "features_selecionadas_fold10.csv",
        METRICAS_DIR / "ranking_mdi_treino_fold10.csv",
    ]
    faltando = [str(p) for p in arquivos_obrigatorios if not p.exists()]
    if faltando:
        raise FileNotFoundError(
            "Arquivos necessários não encontrados (execute 03_treinamento_avaliacao.py primeiro):\n"
            + "\n".join(f" - {f}" for f in faltando)
        )

    clf_dt = joblib.load(MODELOS_DIR / "modelo_dt_mdi_fold10.joblib")
    clf_rf = joblib.load(MODELOS_DIR / "modelo_rf_mdi_fold10.joblib")
    features = pd.read_csv(METRICAS_DIR / "features_selecionadas_fold10.csv")["Atributo"].tolist()
    df_mdi_f10 = pd.read_csv(METRICAS_DIR / "ranking_mdi_treino_fold10.csv")

    df_amostra = pd.read_parquet(MODELOS_DIR / "amostra_balanceada_1200_shap.parquet")
    df_erros = pd.read_parquet(MODELOS_DIR / "erros_teste_fold10_shap.parquet")

    X_df = df_amostra[features].astype(np.float64).copy()
    X_f32 = df_amostra[features].to_numpy(dtype=np.float32)
    y_real = df_amostra["Label_Num"].to_numpy(dtype=np.int32)
    n_samples, n_features = X_df.shape
    n_classes = len(NOMES_CLASSES)

    print(f"Amostra balanceada carregada: {n_samples:,} fluxos x {n_features} atributos")
    plotar_top20_mdi(df_mdi_f10, FIGURAS_DIR / "top20_atributos_mdi_auditado.png")

    t_dt = time.time()
    explainer_dt = shap.TreeExplainer(clf_dt)
    sv_dt = extrair_matriz_shap_3d(explainer_dt.shap_values(X_df), n_samples, n_features, n_classes)
    print(f"TreeSHAP (DT) concluído em {time.time() - t_dt:.2f}s")

    t_rf = time.time()
    explainer_rf = shap.TreeExplainer(clf_rf)
    sv_rf = extrair_matriz_shap_3d(explainer_rf.shap_values(X_df), n_samples, n_features, n_classes)
    print(f"TreeSHAP (RF) concluído em {time.time() - t_rf:.2f}s")

    # Verificação de aditividade local nos 54 atributos
    probs_dt = clf_dt.predict_proba(X_f32)
    probs_rf = clf_rf.predict_proba(X_f32)
    base_dt = np.asarray(explainer_dt.expected_value).flatten()
    base_rf = np.asarray(explainer_rf.expected_value).flatten()

    erro_max_dt = float(np.max(np.abs((base_dt[np.newaxis, :] + np.sum(sv_dt, axis=1)) - probs_dt)))
    erro_max_rf = float(np.max(np.abs((base_rf[np.newaxis, :] + np.sum(sv_rf, axis=1)) - probs_rf)))
    print(f"Erro máximo de aditividade -> DT: {erro_max_dt:.2e} | RF: {erro_max_rf:.2e}")

    df_shap_dt = construir_tabela_global(sv_dt, features)
    df_shap_rf = construir_tabela_global(sv_rf, features)
    df_shap_dt.to_csv(SHAP_DIR / "ranking_shap_global_dt_auditado.csv", index=False)
    df_shap_rf.to_csv(SHAP_DIR / "ranking_shap_global_rf_auditado.csv", index=False)

    plotar_barras_empilhadas_global(
        df_shap_rf,
        "Importância global TreeSHAP por classe - Random Forest (top 15 atributos)",
        FIGURAS_DIR / "shap_global_bar_rf_auditado.png",
    )
    plotar_barras_empilhadas_global(
        df_shap_dt,
        "Importância global TreeSHAP por classe - Decision Tree (top 15 atributos)",
        FIGURAS_DIR / "shap_global_bar_dt_auditado.png",
    )

    mapa_ataques = [
        (1, "DDoS", "shap_beeswarm_ddos_rf_auditado.png"),
        (2, "PortScan", "shap_beeswarm_portscan_rf_auditado.png"),
        (3, "Brute Force", "shap_beeswarm_bruteforce_rf_auditado.png"),
    ]
    for idx_c, nome_c, nome_arq in mapa_ataques:
        plt.figure(figsize=(10, 6.5))
        shap.summary_plot(sv_rf[:, :, idx_c], X_df, feature_names=features, max_display=15, show=False)
        plt.title(
            f"Impacto direcional TreeSHAP (Beeswarm) — Classe {nome_c} (Random Forest)",
            fontsize=11.5,
            fontweight="bold",
            pad=12,
        )
        plt.tight_layout()
        plt.savefig(FIGURAS_DIR / nome_arq, dpi=300, bbox_inches="tight")
        plt.close()

    # Estudos de caso locais: 3 acertos + 1 falso negativo de PortScan
    registros_casos = []
    for idx_c, nome_c, _ in mapa_ataques:
        candidatos = np.where((y_real == idx_c) & (probs_rf[:, idx_c] >= 0.99))[0]
        idx_escolhido = int(candidatos[0])

        vals_shap = sv_rf[idx_escolhido, :, idx_c]
        vals_reais = X_df.iloc[idx_escolhido].to_numpy(dtype=np.float64)
        base_val = float(base_rf[idx_c])
        prob_final = float(probs_rf[idx_escolhido, idx_c])
        soma_todos = float(np.sum(vals_shap))

        exp_local = shap.Explanation(
            values=vals_shap,
            base_values=base_val,
            data=np.round(vals_reais, 2),
            feature_names=features,
        )
        nome_fig = f"shap_local_waterfall_{nome_c.lower().replace(' ', '')}_rf_auditado.png"
        plt.figure(figsize=(10, 6.5))
        shap.plots.waterfall(exp_local, max_display=12, show=False)
        plt.title(
            f"Explicação local TreeSHAP — Estudo de caso: {nome_c} (Prob = {prob_final:.3f})",
            fontsize=11.5,
            fontweight="bold",
            pad=12,
        )
        plt.tight_layout()
        plt.savefig(FIGURAS_DIR / nome_fig, dpi=300, bbox_inches="tight")
        plt.close()

        ordem_top = np.argsort(np.abs(vals_shap))[::-1][:10]
        soma_top10 = float(np.sum(vals_shap[ordem_top]))
        soma_outras = soma_todos - soma_top10
        erro_adit = abs((base_val + soma_todos) - prob_final)

        for rank_l, f_idx in enumerate(ordem_top, start=1):
            registros_casos.append({
                "Tipo_Caso": f"Acerto_{nome_c}",
                "Classe_Explicada": nome_c,
                "Indice_Amostra": idx_escolhido,
                "Valor_Base_Esperado": round(base_val, 6),
                "Soma_SHAP_54_Atributos": round(soma_todos, 6),
                "Soma_Outras_44_Features": round(soma_outras, 6),
                "Probabilidade_Predita": round(prob_final, 6),
                "Erro_Aditividade": erro_adit,
                "Rank_Local": rank_l,
                "Atributo": features[f_idx],
                "Valor_Fisico_Real": vals_reais[f_idx],
                "Valor_Fisico_Formatado": formatar_valor_fisico(vals_reais[f_idx]),
                "Contribuicao_SHAP": round(float(vals_shap[f_idx]), 6),
            })

    # Caso 4: Falso negativo de PortScan no Random Forest
    erros_fn_portscan = df_erros[
        (df_erros["Label_Num"] == 2) & (df_erros["Pred_RF"] == 0)
    ].reset_index(drop=True)
    if len(erros_fn_portscan) > 0:
        X_erro_df = erros_fn_portscan[features].iloc[[0]].astype(np.float64)
        X_erro_f32 = erros_fn_portscan[features].iloc[[0]].to_numpy(dtype=np.float32)
        probs_erro = clf_rf.predict_proba(X_erro_f32)[0]
        sv_erro = extrair_matriz_shap_3d(explainer_rf.shap_values(X_erro_df), 1, n_features, n_classes)[0]

        idx_c_err = 2
        vals_shap_err = sv_erro[:, idx_c_err]
        vals_reais_err = X_erro_df.iloc[0].to_numpy(dtype=np.float64)
        base_val_err = float(base_rf[idx_c_err])
        prob_ps_err = float(probs_erro[idx_c_err])
        prob_ben_err = float(probs_erro[0])
        soma_todos_err = float(np.sum(vals_shap_err))

        exp_erro = shap.Explanation(
            values=vals_shap_err,
            base_values=base_val_err,
            data=np.round(vals_reais_err, 2),
            feature_names=features,
        )
        plt.figure(figsize=(10, 6.5))
        shap.plots.waterfall(exp_erro, max_display=12, show=False)
        plt.title(
            f"Análise de erro TreeSHAP — Falso negativo de PortScan "
            f"(prob PortScan={prob_ps_err:.2f} | Benign={prob_ben_err:.2f})",
            fontsize=10.5,
            fontweight="bold",
            pad=12,
        )
        plt.tight_layout()
        plt.savefig(FIGURAS_DIR / "shap_local_waterfall_erro_portscan_rf_auditado.png", dpi=300, bbox_inches="tight")
        plt.close()

        ordem_top_err = np.argsort(np.abs(vals_shap_err))[::-1][:10]
        soma_top10_err = float(np.sum(vals_shap_err[ordem_top_err]))
        soma_outras_err = soma_todos_err - soma_top10_err
        erro_adit_err = abs((base_val_err + soma_todos_err) - prob_ps_err)

        for rank_l, f_idx in enumerate(ordem_top_err, start=1):
            registros_casos.append({
                "Tipo_Caso": "Erro_FN_PortScan",
                "Classe_Explicada": "PortScan (Predito Benign)",
                "Indice_Amostra": 0,
                "Valor_Base_Esperado": round(base_val_err, 6),
                "Soma_SHAP_54_Atributos": round(soma_todos_err, 6),
                "Soma_Outras_44_Features": round(soma_outras_err, 6),
                "Probabilidade_Predita": round(prob_ps_err, 6),
                "Erro_Aditividade": erro_adit_err,
                "Rank_Local": rank_l,
                "Atributo": features[f_idx],
                "Valor_Fisico_Real": vals_reais_err[f_idx],
                "Valor_Fisico_Formatado": formatar_valor_fisico(vals_reais_err[f_idx]),
                "Contribuicao_SHAP": round(float(vals_shap_err[f_idx]), 6),
            })

    pd.DataFrame(registros_casos).to_csv(
        SHAP_DIR / "estudos_de_caso_locais_shap_auditado.csv", index=False
    )
    print(f"Artefatos SHAP salvos em {SHAP_DIR} e {FIGURAS_DIR} ({time.time() - t0:.2f}s)")


if __name__ == "__main__":
    main()
