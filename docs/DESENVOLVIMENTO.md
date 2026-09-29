# Desenvolvimento

[Documentação](README.md) · [Apresentação do projeto](../README.md)

## Estrutura do repositório

```text
cnn_gato_deteccao_classificacao.ipynb  Treinamento, avaliação e visualizações
cat_datasets.py                       Download, rótulos e limpeza dos dados
cat_landmarks.py                      Preparação e inferência de pontos faciais
train_cat_landmarks.py                Treino opcional de YOLO Pose
webcam.py                            Inferência local e regra SRD
scripts/export_web_models.py         Exportação ONNX e validação de paridade
tests/                               Testes dos datasets e pontos faciais
docs/                                Guias, referências e materiais do trabalho
setup_venv.sh, run_webcam.sh           Preparação do ambiente e webcam
export_web_models.sh                 Lançador da exportação para web
requirements*.txt                    Dependências de treino e exportação
```

Os módulos Python permanecem ao lado do notebook para permitir execução local
e no Colab. Execute os comandos a partir da raiz.

## Ambiente e testes

Siga o [guia de uso](USO.md) para criar a `.venv`, depois execute:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

Os testes verificam limpeza de imagens, rótulos, conversão de anotações,
agrupamento dos splits, resultados e integração dos pontos faciais.
Eles não substituem o treinamento e a avaliação do notebook.
Para validar os ONNX, siga [Exportação para web](EXPORTACAO_WEB.md).

## Reprodutibilidade

- A amostragem usada na divisão por raça possui `random_state=42`.
- Os dados baixados, pesos (`*.pt`) e resultados gerados ficam fora do Git por padrão.
- As versões efetivamente usadas em uma execução podem ser registradas com `pip freeze > environment-lock.txt`.
- O treinamento requer mais memória e é consideravelmente mais rápido com GPU.

## Convenções e arquivos locais

- Documentação em português, com apresentação no README e guias em `docs/`.
- `.editorconfig` define UTF-8, LF e indentação; `.gitattributes` distingue texto de binários.
- Dados, checkpoints, ambientes, fotos locais e resultados gerados ficam fora do Git.
- As saídas do notebook são mantidas como registro da execução. Ao apresentar
  resultados, identifique a configuração usada e evite misturar experimentos.
- Materiais acadêmicos ficam em `docs/`; imagens de documentação, em `docs/assets/`.

Não apague `artifacts/`, `data/` ou `real_photos/` como parte de uma limpeza de
código: podem conter treinamentos e imagens sem outra cópia.
