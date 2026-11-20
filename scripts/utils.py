#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Constantes de caminhos, mapeamento de classes e utilitários compartilhados."""

from pathlib import Path
import numpy as np

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

DADOS_DIR = RAIZ_PROJETO / "dados"
DADOS_PROCESSADOS_DIR = DADOS_DIR / "processados"
MODELOS_DIR = RAIZ_PROJETO / "modelos"

RESULTADOS_DIR = RAIZ_PROJETO / "resultados"
AUDITORIA_DIR = RESULTADOS_DIR / "auditoria"
METRICAS_DIR = RESULTADOS_DIR / "metricas"
SHAP_DIR = RESULTADOS_DIR / "shap"
FIGURAS_DIR = RESULTADOS_DIR / "figuras"

NOMES_CLASSES = ["Benign", "DDoS", "PortScan", "Brute Force"]
MAPA_CLASSES = {nome: idx for idx, nome in enumerate(NOMES_CLASSES)}
MAPA_CLASSES_INV = {idx: nome for idx, nome in enumerate(NOMES_CLASSES)}


def garantir_diretorios() -> None:
    """Cria os diretórios de saída caso ainda não existam."""
    for pasta in (
        DADOS_PROCESSADOS_DIR,
        MODELOS_DIR,
        AUDITORIA_DIR,
        METRICAS_DIR,
        SHAP_DIR,
        FIGURAS_DIR,
    ):
        pasta.mkdir(parents=True, exist_ok=True)


def agrupar_vetores_exatos(X: np.ndarray):
    """
    Atribui um identificador inteiro (0 a G-1) para cada linha única de X.
    Padroniza -0.0 para +0.0 antes da comparação por bytes contíguos.
    """
    X_c = np.ascontiguousarray(X)
    if np.issubdtype(X_c.dtype, np.floating):
        X_c = X_c.copy()
        X_c[X_c == 0.0] = 0.0

    void_dt = np.dtype((np.void, X_c.shape[1] * X_c.dtype.itemsize))
    visao = X_c.view(void_dt).ravel()

    ordem = np.argsort(visao, kind="mergesort")
    dif = np.empty(len(visao), dtype=bool)
    dif[0] = True
    dif[1:] = visao[ordem[1:]] != visao[ordem[:-1]]

    ids_ord = np.cumsum(dif, dtype=np.int32) - 1
    group_ids = np.empty(len(visao), dtype=np.int32)
    group_ids[ordem] = ids_ord
    n_grupos = int(ids_ord[-1]) + 1
    return group_ids, n_grupos
