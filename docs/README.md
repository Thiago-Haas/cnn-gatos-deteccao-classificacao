# Documentação

[Apresentação do projeto](../README.md)

| Guia | Conteúdo |
| --- | --- |
| [Uso](USO.md) | Ambiente, notebook, fotos, webcam e regra SRD |
| [Dados e rótulos](DADOS.md) | Fontes, classes, limpeza e manifestos |
| [Treinamento e avaliação](PROCESSAMENTO.md) | Pipeline, métricas e limitações |
| [Pontos faciais](PONTOS_FACIAIS.md) | Treino opcional, anotações e resultados |
| [Ferramentas](FERRAMENTAS.md) | Bibliotecas e responsabilidades |
| [Exportação para web](EXPORTACAO_WEB.md) | Pacote ONNX, paridade e atualização do app |
| [Desenvolvimento](DESENVOLVIMENTO.md) | Estrutura, testes e reprodutibilidade |

## Materiais do trabalho

- [Enunciado da atividade](projeto-final.pdf).
- [Captura da webcam](assets/webcam-exemplo.png), registro ilustrativo de uma versão anterior.
- O notebook na raiz reúne a implementação e as saídas da execução registrada.

## Relação entre os repositórios

- [cnn-gatos-deteccao-classificacao](https://github.com/Thiago-Haas/cnn-gatos-deteccao-classificacao):
  dados, treinamento, avaliação e exportação.
- [mobile-onnx](https://github.com/Thiago-Haas/mobile-onnx):
  distribuição dos ONNX, interface web, inferência no aparelho e publicação.
