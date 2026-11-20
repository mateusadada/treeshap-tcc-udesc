# Resultados Auditados do Experimento

Esta pasta reúne apenas os artefatos definitivos (tabelas CSV leves e figuras em alta resolução) correspondentes à metodologia final publicada no TCC.

## Organização

- **`auditoria/`** — Relatórios das etapas `01_preprocessamento.py` e `02_auditoria_dados.py`:
  - `resumo_arquivos_csv.csv`: contagem de linhas e classes dos 8 CSVs oficiais.
  - `tabela_limpeza_15_classes.csv` e `tabela_limpeza_4_classes.csv`: volumetria antes e após a limpeza de `NaN`/`Infinity`.
  - `comparacao_float64_vs_float32.csv`, `duplicatas_por_classe_float32.csv`, `conflitos_rotulo_float32.csv` e `distribuicao_stratified_group_kfold.csv`: auditoria de registros duplicados e conflitos de rótulo.
- **`metricas/`** — Relatórios da etapa `03_treinamento_avaliacao.py`:
  - `resumo_desempenho_global_auditado.csv` e `resumo_desempenho_por_classe_auditado.csv`: médias e desvios padrão nos 10 *folds* dos três cenários avaliados.
  - `metricas_10folds_auditadas_brutas.csv` e `historico_selecao_mdi_10folds.csv`: resultados detalhados por partição e estabilidade da seleção MDI intra-fold ($\ge 99\%$).
  - `features_selecionadas_fold10.csv` e `ranking_mdi_treino_fold10.csv`: 54 atributos selecionados no treino do Fold 10.
  - `cm_*.csv`: matrizes de confusão acumuladas (2.571.981 fluxos) para `StratifiedKFold` e `StratifiedGroupKFold`.
- **`shap/`** — Relatórios da etapa `04_explicabilidade_shap.py`:
  - `ranking_shap_global_dt_auditado.csv` e `ranking_shap_global_rf_auditado.csv`: importância global pareada (1.200 fluxos balanceados) para Decision Tree e Random Forest.
  - `estudos_de_caso_locais_shap_auditado.csv`: decomposição local dos 4 estudos de caso (3 acertos + 1 falso negativo de `PortScan`).
- **`figuras/`** — Gráficos finais utilizados no artigo:
  - `fluxograma_metodologia.png`: visão geral das sete etapas do experimento.
  - `top20_atributos_mdi_auditado.png`: top 20 atributos por importância MDI no treino (Fold 10).
  - `cm_StratifiedKFold_MDI_IntraFold_99_decision_tree.png` e `cm_StratifiedKFold_MDI_IntraFold_99_random_forest.png`: matrizes de confusão em 400 DPI.
  - `shap_global_bar_*.png`, `shap_beeswarm_*.png` e `shap_local_waterfall_*.png`: visualizações TreeSHAP globais, direcionais e locais.
