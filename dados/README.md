# Dados do Experimento (CICIDS2017)

Os arquivos originais do *dataset* **CICIDS2017** não são redistribuídos neste repositório devido ao tamanho total e às condições de uso definidas pelos autores da base (*Canadian Institute for Cybersecurity* — UNB).

## Como obter os dados

1. Acesse a página oficial do conjunto de dados:
   - <https://www.unb.ca/cic/datasets/ids-2017.html>
2. Baixe o pacote **`MachineLearningCSV.zip`** (extraído frequentemente como pasta `MachineLearningCVE/`).
3. Extraia os **oito arquivos CSV oficiais** abaixo dentro desta pasta `dados/` (ou em uma subpasta dentro de `dados/`):

- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Monday-WorkingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`

## Dados processados

Ao executar `python3 scripts/01_preprocessamento.py`, o script valida a presença exata dos oito arquivos acima, remove a coluna duplicada (`Fwd Header Length.1`) e os registros com `NaN`/`Infinity`, filtra as quatro classes do estudo (`Benign`, `DDoS`, `PortScan` e `Brute Force`) e grava o arquivo consolidado em:

- `dados/processados/cicids2017_4classes_limpo.parquet` (ignorado pelo Git)
