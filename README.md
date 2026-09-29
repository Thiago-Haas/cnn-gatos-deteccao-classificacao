# Detecção e classificação de raças de gatos

Projeto final da disciplina de Aprendizado Profundo: um pipeline que detecta
gatos, sugere suas raças e visualiza as regiões que influenciam a classificação.
Este repositório reúne os dados de configuração, o treinamento, a avaliação e
a exportação dos modelos para uso no navegador.

**[Documentação](docs/README.md)** ·
**[Notebook](cnn_gato_deteccao_classificacao.ipynb)** ·
**[Aplicação web](https://thiago-haas.github.io/mobile-onnx/)**

## Visão geral

- **YOLOv8n** localiza gatos e fornece os recortes para classificação.
- **ResNet50** é ajustada às raças dos datasets selecionados: até 22 classes
  na união de Oxford-IIIT Pet e Cat Breeds Dataset.
- **Grad-CAM** permite inspecionar as regiões relacionadas à previsão.
- **YOLO Pose**, opcional, estima nove pontos faciais dos gatos.
- O projeto inclui avaliação por classe, inferência pela webcam e exportação ONNX.

A webcam aplica um limiar de baixa confiança para exibir **Sem Raça Definida
(SRD)**. Essa regra não é uma classe treinada nem comprovação de raça.

## Comece aqui

Com Python e suporte a `venv`, execute na raiz:

```bash
./setup_venv.sh
source .venv/bin/activate
jupyter lab
```

Abra `cnn_gato_deteccao_classificacao.ipynb`, confira o `CONFIG` e execute as
células na ordem. Datasets e pesos são baixados durante a execução; o treino
é mais rápido com GPU. Veja [Uso](docs/USO.md) para fotos externas e webcam.

## Do treinamento ao celular

O pipeline exporta o classificador e o detector para ONNX por meio de
`./export_web_models.sh`. O repositório
[mobile-onnx](https://github.com/Thiago-Haas/mobile-onnx) distribui esse pacote
e executa a inferência no navegador, com fotos, câmera, WebGPU/CPU e cache offline.
Veja [Exportação para web](docs/EXPORTACAO_WEB.md).

## Explore o projeto

| Guia | Conteúdo |
| --- | --- |
| [Uso](docs/USO.md) | Ambiente, notebook, webcam e SRD |
| [Dados](docs/DADOS.md) | Datasets, classes, limpeza e créditos |
| [Treinamento e avaliação](docs/PROCESSAMENTO.md) | Etapas, métricas e limitações |
| [Pontos faciais](docs/PONTOS_FACIAIS.md) | Etapa opcional de YOLO Pose |
| [Ferramentas](docs/FERRAMENTAS.md) | Tecnologias utilizadas |
| [Desenvolvimento](docs/DESENVOLVIMENTO.md) | Estrutura, testes e reprodução |
| [Materiais do trabalho](docs/README.md#materiais-do-trabalho) | Enunciado e registro visual |

## Limitações e créditos

As previsões são sugestões entre as classes aprendidas. Iluminação, pose e
imagens diferentes dos dados de treino podem reduzir a qualidade; confiança
alta não comprova genealogia. As limitações da limpeza e da avaliação estão
nos guias de [dados](docs/DADOS.md) e [processamento](docs/PROCESSAMENTO.md).
Datasets, modelos e dependências mantêm suas respectivas licenças e créditos.
