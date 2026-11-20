#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 3: Treinamento e avaliação (Decision Tree e Random Forest) com seleção
MDI (>= 99% acumulada) restrita ao treinamento de cada partição, comparando:
1. Baseline com 77 atributos (Stratified 10-Fold);
2. Seleção MDI intra-fold (Stratified 10-Fold);
3. Seleção MDI intra-fold agrupada por vetor X idêntico (StratifiedGroupKFold).
"""

import time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from utils import (
    DADOS_PROCESSADOS_DIR,
    FIGURAS_DIR,
    MAPA_CLASSES,
    MAPA_CLASSES_INV,
    METRICAS_DIR,
    MODELOS_DIR,
    NOMES_CLASSES,
    agrupar_vetores_exatos,
    garantir_diretorios,
)


def selecionar_mdi_treino(importancias: np.ndarray, limiar_acumulado: float = 0.99):
    """Retorna os índices dos atributos que acumulam >= limiar_acumulado no treino."""
    ordem = np.argsort(importancias)[::-1]
    cumsum = np.cumsum(importancias[ordem])
    n_sel = min(int(np.searchsorted(cumsum, limiar_acumulado) + 1), len(importancias))
    idx_sel = ordem[:n_sel]
    return idx_sel, float(cumsum[n_sel - 1])


def registrar_metricas(
    protocolo: str,
    cenario: str,
    nome_modelo: str,
    fold: int,
    n_features: int,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    tempo_treino: float,
    tempo_teste: float,
    lista_globais: list,
    lista_classes: list,
    cm_dict: dict,
) -> float:
    """Calcula e armazena métricas globais, por classe e matriz de confusão."""
    acc = accuracy_score(y_test, y_pred)
    pr_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rc_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)

    lista_globais.append({
        "Protocolo": protocolo,
        "Cenario": cenario,
        "Modelo": nome_modelo,
        "Fold": fold,
        "N_Features": n_features,
        "Accuracy": acc,
        "Precision_Weighted": pr_w,
        "Recall_Weighted": rc_w,
        "F1_Weighted": f1_w,
        "F1_Macro": f1_m,
        "Train_Time_s": tempo_treino,
        "Test_Time_s": tempo_teste,
    })

    pr_c, rc_c, f1_c, sup_c = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    for idx_c, nome_c in enumerate(NOMES_CLASSES):
        lista_classes.append({
            "Protocolo": protocolo,
            "Cenario": cenario,
            "Modelo": nome_modelo,
            "Fold": fold,
            "Classe": nome_c,
            "Precision": pr_c[idx_c],
            "Recall": rc_c[idx_c],
            "F1_Score": f1_c[idx_c],
            "Suporte_Fold": sup_c[idx_c],
        })

    chave_cm = (protocolo, cenario, nome_modelo)
    if chave_cm not in cm_dict:
        cm_dict[chave_cm] = np.zeros((4, 4), dtype=np.int64)
    cm_dict[chave_cm] += confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    return f1_w


def plotar_matriz_confusao(cm: np.ndarray, caminho_saida: Path) -> None:
    """Salva a matriz de confusão em 400 DPI com fontes ampliadas para leitura lado a lado."""
    cm_perc = cm.astype(np.float64) / cm.sum(axis=1)[:, np.newaxis] * 100.0
    anotacoes = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            anotacoes[i, j] = f"{cm[i, j]:,}\n({cm_perc[i, j]:.2f}%)"

    fig, ax = plt.subplots(figsize=(6.0, 5.1))
    hm = sns.heatmap(
        cm_perc,
        annot=anotacoes,
        fmt="",
        cmap="Blues",
        xticklabels=NOMES_CLASSES,
        yticklabels=NOMES_CLASSES,
        vmin=0,
        vmax=100,
        square=True,
        linewidths=0.6,
        linecolor="#d9d9d9",
        annot_kws={"size": 13.5, "weight": "semibold"},
        cbar_kws={"fraction": 0.044, "pad": 0.025},
        ax=ax,
    )

    ax.set_xlabel("Classe predita", fontsize=13.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Classe real", fontsize=13.5, fontweight="bold", labelpad=8)
    ax.tick_params(axis="x", labelsize=12.5, rotation=0)
    ax.tick_params(axis="y", labelsize=12.5, rotation=90)
    for tick in ax.get_yticklabels():
        tick.set_va("center")

    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11.5)
    cbar.set_label("Taxa por classe real (%)", fontsize=11.5, labelpad=6)

    plt.tight_layout(pad=0.3)
    plt.savefig(caminho_saida, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t_inicio = time.time()
    garantir_diretorios()

    caminho_parquet = DADOS_PROCESSADOS_DIR / "cicids2017_4classes_limpo.parquet"
    if not caminho_parquet.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho_parquet}. Execute 01_preprocessamento.py primeiro."
        )

    df = pd.read_parquet(caminho_parquet)
    features_77 = [c for c in df.columns if c != "Label"]
    y = df["Label"].map(MAPA_CLASSES).to_numpy(dtype=np.int32)
    X = df[features_77].to_numpy(dtype=np.float32, copy=True)
    del df

    print(f"Base carregada: {X.shape[0]:,} linhas x {X.shape[1]} atributos")
    group_ids, n_grupos = agrupar_vetores_exatos(X)
    print(f"Grupos de vetores X distintos (float32): {n_grupos:,}")

    metricas_globais = []
    metricas_classes = []
    historico_selecao_mdi = []
    cm_acumulada = {}

    # Execução 1: Stratified 10-Fold (77 atributos vs. MDI intra-fold >= 99%)
    print("\n--- Stratified 10-Fold (Baseline 77 atributos vs. MDI intra-fold >= 99%) ---")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    ultimo_dt_mdi = None
    ultimo_rf_mdi = None
    ultimo_features_mdi = None
    ultimo_test_idx_skf = None
    ultimo_y_pred_dt = None
    ultimo_y_pred_rf = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        t_fold = time.time()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Baseline 77 atributos: DT
        t0 = time.perf_counter()
        dt_77 = DecisionTreeClassifier(random_state=42)
        dt_77.fit(X_train, y_train)
        t_tr_dt77 = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_dt77 = dt_77.predict(X_test)
        t_te_dt77 = time.perf_counter() - t0
        f1_dt77 = registrar_metricas(
            "StratifiedKFold", "77_Atributos", "Decision Tree", fold, 77,
            y_test, pred_dt77, t_tr_dt77, t_te_dt77,
            metricas_globais, metricas_classes, cm_acumulada,
        )
        del dt_77

        # Baseline 77 atributos: RF (usado também para extrair MDI apenas do treino)
        t0 = time.perf_counter()
        rf_77 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_77.fit(X_train, y_train)
        t_tr_rf77 = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_rf77 = rf_77.predict(X_test)
        t_te_rf77 = time.perf_counter() - t0
        f1_rf77 = registrar_metricas(
            "StratifiedKFold", "77_Atributos", "Random Forest", fold, 77,
            y_test, pred_rf77, t_tr_rf77, t_te_rf77,
            metricas_globais, metricas_classes, cm_acumulada,
        )

        # Seleção MDI intra-fold (>= 99% acumulada no treino)
        imp_treino = rf_77.feature_importances_.copy()
        del rf_77
        idx_sel, imp_acum_sel = selecionar_mdi_treino(imp_treino, limiar_acumulado=0.99)
        n_sel = len(idx_sel)
        nomes_sel = [features_77[i] for i in idx_sel]

        historico_selecao_mdi.append({
            "Protocolo": "StratifiedKFold",
            "Fold": fold,
            "N_Features_Selecionadas": n_sel,
            "Imp_Acumulada (%)": round(imp_acum_sel * 100, 4),
            "Atributos_Zero": int(np.sum(imp_treino == 0.0)),
            "Features_Selecionadas": ", ".join(nomes_sel),
        })

        X_train_mdi = X_train[:, idx_sel]
        X_test_mdi = X_test[:, idx_sel]

        # MDI intra-fold: DT
        t0 = time.perf_counter()
        dt_mdi = DecisionTreeClassifier(random_state=42)
        dt_mdi.fit(X_train_mdi, y_train)
        t_tr_dt_mdi = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_dt_mdi = dt_mdi.predict(X_test_mdi)
        t_te_dt_mdi = time.perf_counter() - t0
        f1_dt_mdi = registrar_metricas(
            "StratifiedKFold", "MDI_IntraFold_99", "Decision Tree", fold, n_sel,
            y_test, pred_dt_mdi, t_tr_dt_mdi, t_te_dt_mdi,
            metricas_globais, metricas_classes, cm_acumulada,
        )

        # MDI intra-fold: RF
        t0 = time.perf_counter()
        rf_mdi = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_mdi.fit(X_train_mdi, y_train)
        t_tr_rf_mdi = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_rf_mdi = rf_mdi.predict(X_test_mdi)
        t_te_rf_mdi = time.perf_counter() - t0
        f1_rf_mdi = registrar_metricas(
            "StratifiedKFold", "MDI_IntraFold_99", "Random Forest", fold, n_sel,
            y_test, pred_rf_mdi, t_tr_rf_mdi, t_te_rf_mdi,
            metricas_globais, metricas_classes, cm_acumulada,
        )

        if fold == 10:
            ultimo_dt_mdi = dt_mdi
            ultimo_rf_mdi = rf_mdi
            ultimo_features_mdi = nomes_sel
            ultimo_test_idx_skf = test_idx
            ultimo_y_pred_dt = pred_dt_mdi
            ultimo_y_pred_rf = pred_rf_mdi

            ordem_f10 = np.argsort(imp_treino)[::-1]
            df_mdi_f10 = pd.DataFrame({
                "Rank": np.arange(1, len(features_77) + 1),
                "Atributo": [features_77[i] for i in ordem_f10],
                "Importancia_MDI": imp_treino[ordem_f10],
                "Importancia_Perc (%)": (imp_treino[ordem_f10] * 100).round(4),
                "Importancia_Acumulada (%)": (np.cumsum(imp_treino[ordem_f10]) * 100).round(4),
            })
            df_mdi_f10.to_csv(METRICAS_DIR / "ranking_mdi_treino_fold10.csv", index=False)
        else:
            del dt_mdi, rf_mdi

        print(
            f"[SKF Fold {fold:>2}/10 | {n_sel} feats] "
            f"DT: {f1_dt77*100:.4f}% -> {f1_dt_mdi*100:.4f}% | "
            f"RF: {f1_rf77*100:.4f}% -> {f1_rf_mdi*100:.4f}% ({time.time() - t_fold:.1f}s)"
        )

    # Execução 2: StratifiedGroupKFold (sem vetores X idênticos entre treino e teste)
    print("\n--- StratifiedGroupKFold (10 folds | MDI intra-fold >= 99%) ---")
    sgkf = StratifiedGroupKFold(n_splits=10)

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=group_ids), start=1):
        t_fold = time.time()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf_sel = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_sel.fit(X_train, y_train)
        imp_treino_g = rf_sel.feature_importances_.copy()
        del rf_sel

        idx_sel_g, imp_acum_g = selecionar_mdi_treino(imp_treino_g, limiar_acumulado=0.99)
        n_sel_g = len(idx_sel_g)
        nomes_sel_g = [features_77[i] for i in idx_sel_g]

        historico_selecao_mdi.append({
            "Protocolo": "StratifiedGroupKFold",
            "Fold": fold,
            "N_Features_Selecionadas": n_sel_g,
            "Imp_Acumulada (%)": round(imp_acum_g * 100, 4),
            "Atributos_Zero": int(np.sum(imp_treino_g == 0.0)),
            "Features_Selecionadas": ", ".join(nomes_sel_g),
        })

        X_train_mdi = X_train[:, idx_sel_g]
        X_test_mdi = X_test[:, idx_sel_g]

        t0 = time.perf_counter()
        dt_g = DecisionTreeClassifier(random_state=42)
        dt_g.fit(X_train_mdi, y_train)
        t_tr_dt_g = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_dt_g = dt_g.predict(X_test_mdi)
        t_te_dt_g = time.perf_counter() - t0
        f1_dt_g = registrar_metricas(
            "StratifiedGroupKFold", "MDI_IntraFold_99", "Decision Tree", fold, n_sel_g,
            y_test, pred_dt_g, t_tr_dt_g, t_te_dt_g,
            metricas_globais, metricas_classes, cm_acumulada,
        )
        del dt_g

        t0 = time.perf_counter()
        rf_g = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_g.fit(X_train_mdi, y_train)
        t_tr_rf_g = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_rf_g = rf_g.predict(X_test_mdi)
        t_te_rf_g = time.perf_counter() - t0
        f1_rf_g = registrar_metricas(
            "StratifiedGroupKFold", "MDI_IntraFold_99", "Random Forest", fold, n_sel_g,
            y_test, pred_rf_g, t_tr_rf_g, t_te_rf_g,
            metricas_globais, metricas_classes, cm_acumulada,
        )
        del rf_g

        print(
            f"[GroupKFold {fold:>2}/10 | {n_sel_g} feats] "
            f"DT={f1_dt_g*100:.4f}% | RF={f1_rf_g*100:.4f}% ({time.time() - t_fold:.1f}s)"
        )

    # Consolidação e exportação dos resultados
    df_globais = pd.DataFrame(metricas_globais)
    df_classes = pd.DataFrame(metricas_classes)
    df_hist_mdi = pd.DataFrame(historico_selecao_mdi)

    resumo_global = (
        df_globais.groupby(["Protocolo", "Cenario", "Modelo"], as_index=False, sort=False)
        .agg(
            Feats_Mean=("N_Features", "mean"),
            Feats_Min=("N_Features", "min"),
            Feats_Max=("N_Features", "max"),
            Acc_Mean=("Accuracy", "mean"),
            Acc_Std=("Accuracy", "std"),
            Pr_Mean=("Precision_Weighted", "mean"),
            Pr_Std=("Precision_Weighted", "std"),
            Rc_Mean=("Recall_Weighted", "mean"),
            Rc_Std=("Recall_Weighted", "std"),
            F1_Mean=("F1_Weighted", "mean"),
            F1_Std=("F1_Weighted", "std"),
            F1_Macro_Mean=("F1_Macro", "mean"),
            F1_Macro_Std=("F1_Macro", "std"),
            Train_Time_Mean_s=("Train_Time_s", "mean"),
            Test_Time_Mean_s=("Test_Time_s", "mean"),
            Test_Time_Std_s=("Test_Time_s", "std"),
        )
    )

    resumo_classes = (
        df_classes.groupby(["Protocolo", "Cenario", "Modelo", "Classe"], as_index=False, sort=False)
        .agg(
            Pr_Mean=("Precision", "mean"),
            Pr_Std=("Precision", "std"),
            Rc_Mean=("Recall", "mean"),
            Rc_Std=("Recall", "std"),
            F1_Mean=("F1_Score", "mean"),
            F1_Std=("F1_Score", "std"),
        )
    )

    print("\nResumo global (10 folds):")
    print(resumo_global.to_string(index=False))

    for (prot, cen, mod), cm in cm_acumulada.items():
        df_cm = pd.DataFrame(
            cm,
            index=[f"Real_{c}" for c in NOMES_CLASSES],
            columns=[f"Pred_{c}" for c in NOMES_CLASSES],
        )
        nome_base = f"cm_{prot}_{cen}_{mod.lower().replace(' ', '_')}"
        df_cm.to_csv(METRICAS_DIR / f"{nome_base}.csv")
        plotar_matriz_confusao(cm, FIGURAS_DIR / f"{nome_base}.png")

    df_globais.to_csv(METRICAS_DIR / "metricas_10folds_auditadas_brutas.csv", index=False)
    resumo_global.to_csv(METRICAS_DIR / "resumo_desempenho_global_auditado.csv", index=False)
    resumo_classes.to_csv(METRICAS_DIR / "resumo_desempenho_por_classe_auditado.csv", index=False)
    df_hist_mdi.to_csv(METRICAS_DIR / "historico_selecao_mdi_10folds.csv", index=False)
    pd.Series(ultimo_features_mdi, name="Atributo").to_csv(
        METRICAS_DIR / "features_selecionadas_fold10.csv", index=False
    )

    # Salva modelos e amostras de teste do Fold 10 para a etapa de explicabilidade
    joblib.dump(ultimo_dt_mdi, MODELOS_DIR / "modelo_dt_mdi_fold10.joblib")
    joblib.dump(ultimo_rf_mdi, MODELOS_DIR / "modelo_rf_mdi_fold10.joblib")

    idx_cols_f10 = [features_77.index(f) for f in ultimo_features_mdi]
    df_test_f10 = pd.DataFrame(X[ultimo_test_idx_skf][:, idx_cols_f10], columns=ultimo_features_mdi)
    df_test_f10["Label_Num"] = y[ultimo_test_idx_skf]
    df_test_f10["Label"] = df_test_f10["Label_Num"].map(MAPA_CLASSES_INV)
    df_test_f10["Pred_DT"] = ultimo_y_pred_dt
    df_test_f10["Pred_RF"] = ultimo_y_pred_rf

    # Subamostra balanceada (300 fluxos por classe) para comparação pareada no SHAP
    rng = np.random.default_rng(42)
    indices_balanceados = []
    y_test_f10 = y[ultimo_test_idx_skf]
    for c_id in range(len(NOMES_CLASSES)):
        idx_c = np.where(y_test_f10 == c_id)[0]
        escolhidos = rng.choice(idx_c, size=min(len(idx_c), 300), replace=False)
        indices_balanceados.extend(np.sort(escolhidos))

    df_amostra = df_test_f10.iloc[indices_balanceados].reset_index(drop=True)
    df_amostra.to_parquet(MODELOS_DIR / "amostra_balanceada_1200_shap.parquet", index=False)

    mask_erros = (df_test_f10["Pred_DT"] != df_test_f10["Label_Num"]) | (
        df_test_f10["Pred_RF"] != df_test_f10["Label_Num"]
    )
    df_erros = df_test_f10.loc[mask_erros].reset_index(drop=True)
    df_erros.to_parquet(MODELOS_DIR / "erros_teste_fold10_shap.parquet", index=False)

    print(f"\nArtefatos salvos em {METRICAS_DIR}, {FIGURAS_DIR} e {MODELOS_DIR}")
    print(f"Tempo total: {time.time() - t_inicio:.2f}s")


if __name__ == "__main__":
    main()
