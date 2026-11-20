# Detecção de DDoS, Port Scan e Brute Force em Redes de Computadores com Machine Learning: Uma Abordagem Interpretável via SHAP

Trabalho de Conclusão de Curso desenvolvido na Universidade do Estado de Santa Catarina (UDESC).

## Sobre o projeto

Este trabalho investiga a detecção de intrusões em redes de computadores por meio de modelos de Aprendizado de Máquina. A solução aborda a classificação multiclasse de tráfego **Benign**, **DDoS**, **Port Scan** e **Brute Force**, comparando os algoritmos **Decision Tree** e **Random Forest** e aplicando o método **SHAP** para interpretar as decisões dos classificadores.

Como extensão ao estudo de referência do grupo GREat/UFC, o projeto utiliza o algoritmo **TreeSHAP**, que permite explicar tanto o comportamento global dos modelos quanto a contribuição dos atributos em classificações individuais.

## Resumo

A detecção de intrusões em redes de computadores utiliza com frequência modelos de Aprendizado de Máquina para identificar padrões de tráfego malicioso. Apesar da alta taxa de acerto, muitos classificadores funcionam como caixas-pretas, dificultando identificar quais características do tráfego motivaram cada alerta. Este trabalho desenvolve e avalia uma abordagem de detecção multiclasse focada nos ataques de DDoS, Port Scan e Brute Force, além do tráfego normal (Benign), utilizando os algoritmos Decision Tree e Random Forest sobre a base de dados CICIDS2017. Como extensão ao estudo de referência do grupo GREat/UFC, integra-se o método SHAP (SHapley Additive exPlanations), por meio do algoritmo TreeSHAP, para fornecer explicabilidade global e local às decisões dos modelos. No desenvolvimento do trabalho, foram realizadas a limpeza de ruídos em 2.571.981 fluxos válidos, a verificação de registros duplicados e atributos redundantes, a seleção de variáveis via Mean Decrease in Impurity (MDI) calculada apenas nos conjuntos de treinamento de cada partição (média de 52,8 atributos) e a avaliação com validação cruzada estratificada (Stratified 10-Fold) e agrupada (StratifiedGroupKFold). Nos testes realizados sobre o CICIDS2017, a Decision Tree e o Random Forest alcançaram F1-Score ponderado de 99,91% e 99,93%, respectivamente, enquanto a análise com o TreeSHAP permitiu identificar como os atributos estatísticos de fluxo influenciaram as classificações corretas e incorretas em cada classe.

## Objetivos

### Objetivo geral

Desenvolver e avaliar uma abordagem interpretável de detecção de ataques de rede baseada em Aprendizado de Máquina.

### Objetivos específicos

- Pré-processar os registros da base CICIDS2017;
- Remover ruídos, registros duplicados e atributos redundantes;
- Selecionar variáveis relevantes sem vazamento de dados entre treinamento e teste;
- Comparar modelos Decision Tree e Random Forest;
- Avaliar os classificadores com validação cruzada estratificada e agrupada;
- Utilizar o TreeSHAP para analisar a importância dos atributos em nível global e local;
- Investigar os atributos associados a acertos e erros de classificação.

## Base de dados

O projeto utiliza a base pública **CICIDS2017**, composta por fluxos de tráfego de rede coletados em um ambiente controlado. O conjunto final analisado contém **2.571.981 fluxos válidos** distribuídos entre tráfego benigno e três tipos de ataques:

| Classe | Descrição |
| --- | --- |
| Benign | Tráfego normal |
| DDoS | Ataque de negação de serviço distribuído |
| Port Scan | Varredura de portas |
| Brute Force | Tentativas de acesso por força bruta |

## Metodologia

O desenvolvimento da pesquisa segue as seguintes etapas:

1. Limpeza e preparação dos dados;
2. Identificação e remoção de duplicatas;
3. Identificação e tratamento de atributos redundantes;
4. Seleção de atributos com Mean Decrease in Impurity (MDI);
5. Particionamento dos dados em conjuntos de treinamento e teste;
6. Treinamento dos modelos Decision Tree e Random Forest;
7. Avaliação por validação cruzada;
8. Interpretação das predições com TreeSHAP.

A seleção de variáveis foi realizada separadamente em cada partição de treinamento, evitando vazamento de informações provenientes dos conjuntos de validação ou teste. Em média, **52,8 atributos** foram utilizados em cada partição.

## Modelos avaliados

| Modelo | Técnica de validação | Métrica principal |
| --- | --- | --- |
| Decision Tree | Stratified 10-Fold e StratifiedGroupKFold | F1-Score ponderado |
| Random Forest | Stratified 10-Fold e StratifiedGroupKFold | F1-Score ponderado |

## Validação

Os modelos foram avaliados utilizando duas estratégias:

- **Stratified 10-Fold:** preserva a proporção das classes em cada divisão;
- **StratifiedGroupKFold:** mantém a separação estratificada considerando agrupamentos dos registros.

## Resultados

No conjunto de testes da base CICIDS2017, os modelos alcançaram os seguintes resultados de **F1-Score ponderado**:

| Modelo | F1-Score ponderado |
| --- | --- |
| Decision Tree | 99,91% |
| Random Forest | 99,93% |

## Explicabilidade com SHAP

O método SHAP foi aplicado aos modelos baseados em árvores por meio do algoritmo **TreeSHAP**. A análise combina explicações globais e locais:

- **Explicabilidade global:** identifica os atributos que mais influenciam o comportamento geral dos classificadores;
- **Explicabilidade local:** quantifica a contribuição de cada atributo para uma predição específica;
- **Análise de erros:** auxilia na identificação dos atributos associados às classificações corretas e incorretas de cada classe.

## Tecnologias

Este projeto utiliza conceitos e ferramentas relacionadas a:

- Python;
- Aprendizado de Máquina;
- Decision Tree;
- Random Forest;
- CICIDS2017;
- SHAP e TreeSHAP;
- Validação cruzada estratificada.

## Estrutura do repositório

```text
.
├── LICENSE
└── README.md
```

## Referência ao estudo de origem

Este trabalho amplia o estudo de referência do grupo **GREat/UFC**, incorporando análise interpretável por meio do SHAP/TreeSHAP às tarefas de detecção e classificação de ataques em redes de computadores.

## Licença

Este projeto está sob os termos da licença definida no arquivo [LICENSE](LICENSE).
