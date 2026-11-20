#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstração prática de explicabilidade local com TreeSHAP.

Modos de operação:
- modelo: carrega o Random Forest treinado (modelos/modelo_rf_mdi_fold10.joblib)
  e os fluxos de teste em Parquet, executa a classificação e calcula o TreeSHAP
  em tempo de execução.
- precalculado: consulta os resultados auditados salvos em
  resultados/shap/estudos_de_caso_locais_shap_auditado.csv sem exigir o modelo
  treinado nem o dataset completo.
- auto (padrão): utiliza o modo "modelo" se os artefatos locais existirem em
  modelos/; caso contrário, alterna automaticamente para "precalculado".
"""

import argparse
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
MODELOS_DIR = RAIZ_PROJETO / "modelos"
METRICAS_DIR = RAIZ_PROJETO / "resultados" / "metricas"
SHAP_DIR = RAIZ_PROJETO / "resultados" / "shap"
FIGURAS_DIR = RAIZ_PROJETO / "resultados" / "figuras"

NOMES_CLASSES = ["Benign", "DDoS", "PortScan", "Brute Force"]

CASOS = {
    "bruteforce": {
        "tipo_csv": "Acerto_Brute Force",
        "classe_idx": 3,
        "indice_amostra": 900,
        "eh_erro": False,
        "titulo": "Estudo de Caso 3 — Classificação correta de Brute Force (Amostra 900)",
        "figura": "shap_local_waterfall_bruteforce_rf_auditado.png",
        "interpretacao": (
            "A classificação como Brute Force foi impulsionada pela porta de destino 21 "
            "(padrão do protocolo FTP) combinada a pacotes curtos e baixo volume de bytes "
            "trocados na conexão. Como os dados tabulares não contêm o conteúdo dos pacotes, "
            "esse alerta não comprova sozinho os comandos enviados; um analista deveria auditar "
            "os logs de autenticação do servidor FTP ou a captura de pacotes (PCAP)."
        ),
    },
    "ddos": {
        "tipo_csv": "Acerto_DDoS",
        "classe_idx": 1,
        "indice_amostra": 300,
        "eh_erro": False,
        "titulo": "Estudo de Caso 1 — Classificação correta de DDoS (Amostra 300)",
        "figura": "shap_local_waterfall_ddos_rf_auditado.png",
        "interpretacao": (
            "A classificação como DDoS foi guiada pelo contraste estatístico entre pacotes "
            "curtos enviados pela origem (sentido Fwd) e respostas volumosas e de alta "
            "variação no sentido de retorno (Bwd). Para confirmar o incidente em ambiente real, "
            "o analista deve correlacionar o volume simultâneo de fluxos e os logs do serviço alvo."
        ),
    },
    "portscan": {
        "tipo_csv": "Acerto_PortScan",
        "classe_idx": 2,
        "indice_amostra": 600,
        "eh_erro": False,
        "titulo": "Estudo de Caso 2 — Classificação correta de PortScan (Amostra 600)",
        "figura": "shap_local_waterfall_portscan_rf_auditado.png",
        "interpretacao": (
            "A decisão pela classe PortScan foi motivada pelo volume mínimo de bytes enviados "
            "e recebidos no fluxo combinado à contagem da flag TCP PSH. Isoladamente, essa flag "
            "não define a técnica exata de varredura; a confirmação exige verificar a sequência "
            "de conexões para múltiplas portas ou os pacotes originais (PCAP)."
        ),
    },
    "erro_portscan": {
        "tipo_csv": "Erro_FN_PortScan",
        "classe_idx": 2,
        "indice_amostra": 0,
        "eh_erro": True,
        "titulo": "Estudo de Caso 4 — Erro do modelo: Falso Negativo de PortScan",
        "figura": "shap_local_waterfall_erro_portscan_rf_auditado.png",
        "interpretacao": (
            "Neste fluxo rotulado como PortScan, o modelo previu Benign. Embora o envio de 0 bytes "
            "e a alta taxa de retorno tenham empurrado a probabilidade para PortScan, a ausência da "
            "flag PSH (0), a presença da flag ACK (1) e respostas com 0 bytes puxaram a decisão para "
            "tráfego normal. O caso mostra como o SHAP auxilia a diagnosticar falhas em fluxos "
            "estatisticamente ambíguos."
        ),
    },
}


def formatar_valor(val: float) -> str:
    if abs(val - round(val)) < 1e-5:
        return f"{int(round(val))}"
    return f"{val:.4f}"


def artefatos_modelo_disponiveis() -> bool:
    necessarios = [
        MODELOS_DIR / "modelo_rf_mdi_fold10.joblib",
        MODELOS_DIR / "amostra_balanceada_1200_shap.parquet",
        MODELOS_DIR / "erros_teste_fold10_shap.parquet",
        METRICAS_DIR / "features_selecionadas_fold10.csv",
    ]
    return all(p.exists() for p in necessarios)


def extrair_shap_instancia(shap_vals, n_features: int, n_classes: int) -> np.ndarray:
    """Retorna matriz (n_features, n_classes) para uma única amostra."""
    if isinstance(shap_vals, list):
        return np.stack([np.asarray(v)[0] for v in shap_vals], axis=-1)
    arr = np.asarray(shap_vals)
    if arr.shape == (1, n_features, n_classes):
        return arr[0]
    if arr.shape == (n_classes, 1, n_features):
        return arr[:, 0, :].T
    raise ValueError(f"Formato inesperado retornado por shap_values: {arr.shape}")


def exibir_relatorio(
    titulo: str,
    modo_label: str,
    classe_real: str,
    classe_predita: str,
    classe_explicada: str,
    prob_explicada: float,
    base_val: float,
    top_atributos: list,
    interpretacao: str,
    caminho_figura: Path,
    probs_todas: dict | None = None,
    tempo_s: float | None = None,
) -> None:
    print("=" * 84)
    print(f"ALERTA INTERPRETÁVEL TREESHAP | {modo_label}")
    print("=" * 84)
    print(f"Caso:             {titulo}")
    print(f"Classe real:      {classe_real}")
    print(f"Classe predita:   {classe_predita}")
    if probs_todas:
        resumo_probs = " | ".join(f"{k}: {v*100:.2f}%" for k, v in probs_todas.items())
        print(f"Probabilidades:   {resumo_probs}")
    print(
        f"Classe explicada: {classe_explicada} "
        f"(base E[f(X)] = {base_val:.4f} -> estimativa f(x) = {prob_explicada:.4f})"
    )
    if tempo_s is not None:
        print(f"Tempo de inferência + TreeSHAP: {tempo_s:.2f}s")

    print("\nPrincipais atributos que influenciaram a decisão:")
    print(f"  {'#':<3} {'Atributo':<30} {'Valor no fluxo':<16} {'Contribuição SHAP'}")
    print("  " + "-" * 72)
    for rank, nome, val_str, shap_val in top_atributos:
        direcao = "aumenta" if shap_val >= 0 else "reduz"
        print(f"  {rank:<3} {nome:<30} {val_str:<16} {shap_val:+.4f} ({direcao})")

    print("\nLeitura prática para um analista de segurança:")
    print(f"  {interpretacao}")

    print("\nAviso metodológico:")
    print(
        "  Demonstração offline sobre estatísticas agregadas de fluxo (CSV do CICIDS2017). "
        "O SHAP explica a decisão do classificador e não substitui inspeção de pacotes (PCAP) ou logs."
    )
    if caminho_figura.exists():
        print(f"\nGráfico waterfall de referência: {caminho_figura}")
    print("=" * 84)


def executar_modo_modelo(chave_caso: str, salvar_grafico: bool) -> None:
    import joblib
    import shap

    if not artefatos_modelo_disponiveis():
        raise FileNotFoundError(
            "Artefatos do modelo não encontrados em modelos/. "
            "Execute scripts/03_treinamento_avaliacao.py, baixe o pacote via GitHub Releases "
            "(se disponível) ou rode com --modo precalculado."
        )

    cfg = CASOS[chave_caso]
    t0 = time.perf_counter()

    clf_rf = joblib.load(MODELOS_DIR / "modelo_rf_mdi_fold10.joblib")
    features = pd.read_csv(METRICAS_DIR / "features_selecionadas_fold10.csv")["Atributo"].tolist()

    if cfg["eh_erro"]:
        df_erros = pd.read_parquet(MODELOS_DIR / "erros_teste_fold10_shap.parquet")
        sub = df_erros[(df_erros["Label_Num"] == 2) & (df_erros["Pred_RF"] == 0)].reset_index(drop=True)
        linha = sub.iloc[[0]]
    else:
        df_amostra = pd.read_parquet(MODELOS_DIR / "amostra_balanceada_1200_shap.parquet")
        linha = df_amostra.iloc[[cfg["indice_amostra"]]]

    X_df = linha[features].astype(np.float64)
    X_f32 = linha[features].to_numpy(dtype=np.float32)

    probs = clf_rf.predict_proba(X_f32)[0]
    pred_idx = int(np.argmax(probs))
    real_idx = int(linha["Label_Num"].iloc[0])

    explainer = shap.TreeExplainer(clf_rf)
    sv_inst = extrair_shap_instancia(explainer.shap_values(X_df), len(features), len(NOMES_CLASSES))
    base_vals = np.asarray(explainer.expected_value).flatten()

    idx_exp = cfg["classe_idx"]
    vals_shap = sv_inst[:, idx_exp]
    vals_reais = X_df.iloc[0].to_numpy(dtype=np.float64)
    tempo_exec = time.perf_counter() - t0

    ordem = np.argsort(np.abs(vals_shap))[::-1][:6]
    top_atributos = [
        (rank, features[i], formatar_valor(vals_reais[i]), float(vals_shap[i]))
        for rank, i in enumerate(ordem, start=1)
    ]

    caminho_fig = FIGURAS_DIR / cfg["figura"]
    if salvar_grafico:
        import matplotlib.pyplot as plt

        exp_local = shap.Explanation(
            values=vals_shap,
            base_values=float(base_vals[idx_exp]),
            data=np.round(vals_reais, 2),
            feature_names=features,
        )
        caminho_fig = RAIZ_PROJETO / "exemplo" / f"saida_demo_{chave_caso}.png"
        plt.figure(figsize=(10, 6.5))
        shap.plots.waterfall(exp_local, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(caminho_fig, dpi=300, bbox_inches="tight")
        plt.close()

    exibir_relatorio(
        titulo=cfg["titulo"],
        modo_label="EXECUÇÃO AO VIVO DO MODELO RF + TREESHAP",
        classe_real=NOMES_CLASSES[real_idx],
        classe_predita=NOMES_CLASSES[pred_idx],
        classe_explicada=NOMES_CLASSES[idx_exp],
        prob_explicada=float(probs[idx_exp]),
        base_val=float(base_vals[idx_exp]),
        top_atributos=top_atributos,
        interpretacao=cfg["interpretacao"],
        caminho_figura=caminho_fig,
        probs_todas={NOMES_CLASSES[i]: float(probs[i]) for i in range(len(NOMES_CLASSES))},
        tempo_s=tempo_exec,
    )


def executar_modo_precalculado(chave_caso: str) -> None:
    caminho_csv = SHAP_DIR / "estudos_de_caso_locais_shap_auditado.csv"
    if not caminho_csv.exists():
        raise FileNotFoundError(f"Arquivo de resultados pré-calculados não encontrado: {caminho_csv}")

    cfg = CASOS[chave_caso]
    df = pd.read_csv(caminho_csv)
    sub = df[df["Tipo_Caso"] == cfg["tipo_csv"]].sort_values("Rank_Local").head(6)
    if sub.empty:
        raise ValueError(f"Caso '{cfg['tipo_csv']}' não encontrado em {caminho_csv}")

    primeira = sub.iloc[0]
    classe_real = "PortScan" if cfg["eh_erro"] else str(primeira["Classe_Explicada"])
    classe_predita = "Benign" if cfg["eh_erro"] else classe_real
    classe_explicada = "PortScan" if cfg["eh_erro"] else classe_real

    top_atributos = [
        (
            int(row["Rank_Local"]),
            str(row["Atributo"]),
            formatar_valor(float(row["Valor_Fisico_Real"])),
            float(row["Contribuicao_SHAP"]),
        )
        for _, row in sub.iterrows()
    ]

    exibir_relatorio(
        titulo=cfg["titulo"],
        modo_label="CONSULTA A RESULTADOS AUDITADOS PRÉ-CALCULADOS (SEM MODELO .JOBLIB)",
        classe_real=classe_real,
        classe_predita=classe_predita,
        classe_explicada=classe_explicada,
        prob_explicada=float(primeira["Probabilidade_Predita"]),
        base_val=float(primeira["Valor_Base_Esperado"]),
        top_atributos=top_atributos,
        interpretacao=cfg["interpretacao"],
        caminho_figura=FIGURAS_DIR / cfg["figura"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demonstração de alerta interpretável com Random Forest e TreeSHAP."
    )
    parser.add_argument(
        "--caso",
        choices=list(CASOS.keys()),
        default="bruteforce",
        help="Estudo de caso a exibir (padrão: bruteforce).",
    )
    parser.add_argument(
        "--modo",
        choices=["auto", "modelo", "precalculado"],
        default="auto",
        help="Modo de execução: auto, modelo (requer .joblib) ou precalculado (lê CSV auditado).",
    )
    parser.add_argument(
        "--salvar-grafico",
        action="store_true",
        help="Salva o gráfico waterfall gerado no modo modelo em exemplo/.",
    )
    args = parser.parse_args()

    usar_modelo = (args.modo == "modelo") or (args.modo == "auto" and artefatos_modelo_disponiveis())
    if usar_modelo:
        executar_modo_modelo(args.caso, args.salvar_grafico)
    else:
        executar_modo_precalculado(args.caso)


if __name__ == "__main__":
    main()
