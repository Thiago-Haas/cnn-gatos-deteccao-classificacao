# Checklist de entrega — diagnóstico do projeto

Projeto: **Detecção e classificação de raças de gatos** · Thiago Haas Rausch e Bruna Henning Pereira
Disciplina: Aprendizado Profundo · PPGCA / UNIVALI · Prof. Felipe Viel
Base: `Projeto Final.pdf`, Seções 9, 10, 11, 12 e 14
Fontes verificadas: `cnn-gatos-deteccao-classificacao` (notebook, docs, artifacts) e `mobile-onnx` (app publicado)
Última atualização: 29/09/2026, depois de rodar E1–E3 e a análise SRD

---

## Resumo

| | |
|---|---|
| Cumpridos | 16 de 17 itens |
| **Pendente** | **1 — o relatório técnico** |

O plano de experimentos, que era o maior risco, está fechado: E1, E2 e E3 rodaram
com tabela comparativa e gráfico. A análise da regra SRD, que não era exigida,
virou o resultado mais forte do trabalho. Falta apenas o relatório técnico.

O notebook foi reorganizado para seguir este checklist: cada item tem a sua
seção, e a tabela do topo do notebook mapeia uma coisa na outra.

---

## Checklist de Entrega (Seção 14)

| # | Item | Status | Onde está / o que falta |
|---|---|---|---|
| 1 | Problema claramente definido | ✅ | Detecção + classificação de raça em 22 classes; enquadrado no Problema 1 do enunciado |
| 2 | Motivação apresentada | ✅ | Seção 1 do notebook e slide 2 do deck |
| 3 | Dataset descrito e fonte registrada | ✅ | `docs/DADOS.md`: Oxford-IIIT Pet + Cat Breeds Dataset (CC BY 4.0), commit `56a6905` fixado, citação e licença |
| 4 | Análise exploratória realizada | ✅ | Seção 4: distribuição por raça e fonte, razão de desbalanceamento e uma imagem por classe |
| 5 | Problemas dos dados identificados | ✅ | Ilegíveis, duplicatas RGB exatas, conflito de rótulo, desbalanceamento (27 a 85 no teste), ausência de classe SRD. `dataset_rejections.csv` |
| 6 | Pré-processamento documentado | ✅ | 224×224, normalização ImageNet, augmentation parametrizado no `CONFIG` |
| 7 | Divisão treino/validação/teste definida | ✅ | 70/15/15 estratificado, `random_state=42`, manifests salvos, *assert* de disjunção por SHA-256 — evita data leakage |
| 8 | Arquitetura justificada | ✅ | ResNet50 + transfer learning; YOLOv8n COCO para detecção |
| 9 | Baseline implementado | ✅ | E1 — backbone congelado, só a cabeça: 0,756 de acurácia e 0,725 de F1-macro |
| 10 | Pelo menos duas variações experimentais | ✅ | E2 (layer3+layer4+fc) e E3 (layer4+fc), uma variável alterada por vez · Seção 10 |
| 11 | Histórico de treinamento apresentado | ✅ | Seção 11: curvas de loss e acurácia dos três experimentos, com o gap treino−validação |
| 12 | Métricas apropriadas utilizadas | ✅ | Seção 12: F1-macro como métrica principal, relatório por classe e matriz de confusão normalizada |
| 13 | Resultados comparados em tabela e/ou gráfico | ✅ | Seção 13 · `artifacts/experimentos/comparativo.csv` e `.md` |
| 14 | Limitações discutidas | ✅ | Seções 16 e 17: a limitação SRD é **medida**, não só declarada |
| 15 | Código organizado | ✅ | Módulos ao lado do notebook, docs em português, testes em `tests/`, seed fixa |
| 16 | **Relatório finalizado** | ❌ | **Única pendência.** Todos os números já existem em `artifacts/experimentos/` |
| 17 | Apresentação preparada | ✅ | `docs/apresentacao-final-gatos.pptx`, 14 slides |

---

## Requisitos Mínimos (Seção 12)

Os 15 requisitos espelham o checklist acima. O único ainda não cumprido:

- **#15 — Relatório e apresentação.** A apresentação está pronta; falta o relatório.

---

## Como cada item aparece na apresentação

O enunciado (Seção 10) pede que a apresentação destaque **problema, dados,
arquitetura, experimentos, métricas, resultados e conclusões** em 10 a 15 minutos.
O deck final foi montado com um slide por bloco da rubrica:

| Slide | Cobre |
|---|---|
| 2 · Problema e motivação | Checklist 1, 2 · Rubrica: definição e contextualização (10%) |
| 3 · Dados e análise exploratória | Checklist 3, 4, 5 · Rubrica: análise e preparação dos dados (15%) |
| 4 · Pré-processamento e split | Checklist 6, 7 |
| 5 · Arquitetura e justificativa | Checklist 8 · Rubrica: implementação e compreensão da arquitetura (20%) |
| 6 · Plano de experimentos | Checklist 9, 10 · Rubrica: exploração experimental (20%) |
| 7 · Tabela comparativa | Checklist 13 · Rubrica: avaliação e análise (20%) |
| 8 · Métricas e histórico de treino | Checklist 11, 12 |
| 9 · SRD, calibração e cobertura × acurácia | Checklist 14 · Perguntas de reflexão 5, 6, 7 |
| 10 · Deploy no navegador | Diferencial do trabalho |
| 11 · Limitações e conclusões | Checklist 14 · Perguntas de reflexão 3, 8 |
| 12 · Reprodutibilidade | Rubrica: organização e reprodutibilidade (5%) |

---

## Rubrica de avaliação (Seção 11) — onde o risco está

| Critério | Peso | Situação |
|---|---|---|
| Definição e contextualização do problema | 10% | ✅ Sólido |
| Análise e preparação dos dados | 15% | ✅ Sólido — limpeza documentada é um ponto forte |
| Implementação e compreensão da arquitetura | 20% | ✅ Sólido |
| **Exploração experimental e hiperparâmetros** | **20%** | ❌ **Em risco: sem E2/E3** |
| **Avaliação e análise dos resultados** | **20%** | ⚠️ Métricas sólidas, mas sem comparação entre configurações |
| Relatório e apresentação | 10% | ❌ Relatório ausente |
| Organização e reprodutibilidade do código | 5% | ✅ Sólido |

---

## Perguntas para reflexão (Seção 14.1) — estado das respostas

| # | Pergunta | Resposta disponível? |
|---|---|---|
| 1 | Por que esta arquitetura foi escolhida? | ✅ ResNet50: conexões residuais, pesos ImageNet, custo compatível com a disciplina |
| 2 | Qual característica dos dados mais influencia o problema? | ✅ Similaridade entre raças de pelo curto e o desbalanceamento (27 a 85 no teste) |
| 3 | Qual foi a principal dificuldade? | ✅ Raças visualmente próximas: Devon Rex 0,56 e Cornish Rex 0,58 de F1 |
| 4 | Qual alteração experimental produziu maior impacto? | ❌ **Só responderemos com a tabela E1/E2/E3** |
| 5 | Houve sobreajuste? Como foi identificado? | ⚠️ Evidência existe (treino 0,92 × validação 0,80 no fim), mas **falta dizer isso explicitamente** |
| 6 | A métrica escolhida representa o objetivo? | ⚠️ F1-macro sim para raças; **não** para o uso real, onde um gato SRD não tem rótulo válido. `analise_srd.py` fecha isso |
| 7 | O modelo generaliza para dados não vistos? | ✅ Demonstrado que **não** para SRD: Tom → Bombay 76,4%, Bob → Sphynx 58,8% |
| 8 | O que seria necessário para melhorar? | ✅ Classe SRD treinada, mais dados de gatos comuns, rejeição calibrada |

---

## Achados adicionais (não pedidos, mas que pesam na nota)

**1. A avaliação de `real_photos/` está quebrada.**
As 30 fotos estão soltas na raiz de `real_photos/`, e o notebook usa o nome da
subpasta como rótulo verdadeiro. Resultado: todas recebem o rótulo `real_photos`
e todas as 30 aparecem como **ERRADO** na saída salva — inclusive as corretas.
Um avaliador lendo o notebook vê "0 de 30". Duas saídas:

- organizar por raça (`real_photos/Persian/foto.jpg`) para as que têm raça conhecida; ou
- assumir que são o **conjunto de controle SRD** — que é o uso mais interessante — e
  avaliá-las com `analise_srd.py`, onde o acerto é *rejeitar*, não classificar.

**2. A última célula do notebook está salva com erro.**
A célula 42 (webcam) ficou com `RuntimeError: Camera index out of range` gravado
na saída. Limpe essa saída antes de entregar: o item 14 da rubrica é "código
organizado", e um traceback na última célula é a última coisa que o avaliador lê.

**3. O `LANDMARKS_MODE` está `"off"`.**
Os pontos faciais existem, estão documentados e não aparecem na execução salva.
Ou ligue em uma execução, ou trate como trabalho futuro no relatório — não deixe
o código sem evidência.

**4. A regra SRD ainda não tem números.**
O limiar de 40% está em produção na webcam e no app, mas não há nenhuma medida
que o justifique. `scripts/analise_srd.py` produz: temperatura de calibração,
ECE antes/depois, curva cobertura × acurácia, e a taxa de rejeição no conjunto
de controle. Sem isso, o 40% é um número escolhido a olho — e essa é exatamente
a pergunta que o professor vai fazer.

---

## O que rodar

O notebook faz tudo, na ordem do checklist. Selecione o kernel `.venv/bin/python`
e rode as células em sequência — em GPU o Run All completo leva cerca de 15
minutos, e a Seção 2 avisa se a execução caiu em CPU antes de qualquer treino.

Ele gera, em `artifacts/figuras/`, todas as imagens usadas na apresentação:
distribuição de classes, diagrama do pipeline, histórico de treino de cada
experimento, F1 por classe, matriz de confusão, comparativo dos experimentos e
as curvas da regra SRD.

### Pelo terminal, se preferir

```bash
source .venv/bin/activate      # sem isto, cai em CPU e leva horas

python scripts/experimentos.py --exp E1 E2 E3
python scripts/experimentos.py --exp E4     # opcional: sem augmentation
python scripts/analise_srd.py --checkpoint artifacts/experimentos/E2/classificador.pt
```

As duas linhas de comando usam o mesmo `cat_experimentos.py` que o notebook —
não existem duas implementações que possam divergir.

### Execução em GPU

`preparar_gpu()` liga `cudnn.benchmark`, TF32, precisão mista (`bfloat16` quando
a GPU suporta, senão `float16`), `channels_last`, `pin_memory`, cópia assíncrona
e os *workers* do `DataLoader`. Se cair em CPU, distingue os dois motivos: build
de PyTorch sem CUDA (kernel errado) ou nenhuma GPU visível.

O `NUM_WORKERS = 0` do notebook antigo existia porque um `Dataset` definido em
célula não pode ser enviado aos processos do `DataLoader`. Com a classe no
módulo, esse limite some — é a diferença que mais pesa no tempo por época.

A precisão mista muda os últimos dígitos das métricas. Para reproduzir bit a
bit, use `--sem-amp` no terminal ou `preparar_gpu(amp=False)` no notebook.

## Resultados obtidos

| Exp. | Configuração | Params | Épocas | Acurácia | F1-macro | Gap treino−val |
|---|---|---|---|---|---|---|
| E1 | Baseline: só a cabeça | 0,05 M | 30 | 0,756 | 0,725 | 0,036 |
| **E2** | + layer3, layer4 e fc | 22,11 M | 50 | **0,806** | **0,781** | 0,162 |
| E3 | + layer4 e fc | 15,01 M | 50 | 0,797 | 0,773 | 0,141 |

As três hipóteses se confirmaram. E2 ganha 5,0 pontos de acurácia sobre o
baseline, **mas o gap salta de 0,036 para 0,162** — o ganho vem junto com o
sobreajuste. E3 mostra que 32% menos parâmetros custam apenas 0,8 ponto.

### Regra SRD

| Medida | Valor |
|---|---|
| Temperatura de calibração | T = 0,87 — o modelo estava **sub**confiante |
| ECE antes / depois | 0,043 → 0,025 |
| Limiar em uso (0,40) | cobertura 95,1%, acurácia entre aceitos 83,2% |
| Limiar para 90% de acurácia | 0,65 — cobertura cai para 74,7% |
| Controle SRD em 0,40 | só **23%** das 30 fotos são rejeitadas |
| Controle SRD em 0,65 | 47% — ainda mais da metade erra |

Conclusão: **rejeição por confiança não substitui uma classe SRD treinada.**
Rejeitar 87% dos gatos SRD exigiria limiar 0,90, que derruba a cobertura a 48%.

## O que falta

Só o **relatório técnico** (Seção 10 do enunciado): Introdução, Contextualização,
Definição do problema, Dataset, Análise exploratória, Pré-processamento,
Arquitetura, Metodologia, Experimentos, Resultados, Discussão, Limitações,
Conclusão e Referências. Todos os números já estão em `artifacts/experimentos/`
e as figuras em `artifacts/figuras/`.
