# Detecção e classificação de raças de gatos

Projeto final da disciplina de Aprendizado Profundo. O notebook implementa um pipeline de visão computacional que detecta gatos em imagens, classifica suas raças e explica as previsões com mapas de ativação.

## Visão geral

O projeto usa:

- **Oxford-IIIT Pet** como conjunto de dados;
- **ResNet50** pré-treinada na ImageNet para classificar as raças;
- **YOLOv8 Nano** pré-treinado no COCO para detectar e recortar gatos;
- **Grad-CAM** para visualizar as regiões que influenciam a classificação;
- acurácia, F1-score macro, relatório de classificação e matriz de confusão para avaliação.

O conjunto é filtrado para imagens de gatos e dividido de forma estratificada em 70% para treino, 15% para validação e 15% para teste.

## Estrutura

```text
.
├── cnn_gato_deteccao_classificacao.ipynb  # implementação e análise
├── docs/
│   └── projeto-final.pdf                   # enunciado da atividade
├── .gitignore
├── README.md
├── requirements.txt
└── setup_venv.sh                           # criação do ambiente local
```

## Como executar

Para preparar um ambiente local:

```bash
chmod +x setup_venv.sh
./setup_venv.sh
source .venv/bin/activate
jupyter lab
```

O script cria o ambiente em `.venv/`, atualiza as ferramentas de instalação e baixa todas as dependências declaradas em `requirements.txt`. Esse diretório já está listado no `.gitignore` e não será enviado ao repositório.

Se o executável do Python tiver outro nome ou caminho, informe-o assim:

```bash
PYTHON_BIN=/caminho/para/python ./setup_venv.sh
```

Abra o notebook no VS Code ou JupyterLab, selecione o interpretador `.venv/bin/python` como kernel e execute as células na ordem. Os caminhos usados são relativos à raiz do projeto.

Para testar fotos externas, coloque os arquivos `.jpg`, `.jpeg`, `.png` ou `.webp` no diretório `real_photos/`. A pasta é criada automaticamente pela célula correspondente e não é versionada.

O notebook baixa automaticamente o Oxford-IIIT Pet, os pesos da ResNet50 e o modelo `yolov8n.pt`. Esses artefatos exigem conexão com a internet e não são versionados.

## Pipeline

1. Baixar e explorar o Oxford-IIIT Pet.
2. Filtrar as imagens de gatos e criar os rótulos das raças.
3. Aplicar redimensionamento, normalização e data augmentation.
4. Treinar a camada classificadora da ResNet50.
5. Fazer fine-tuning parcial da última camada residual.
6. Avaliar no conjunto de teste.
7. Detectar gatos em novas imagens com YOLOv8.
8. Classificar o recorte e gerar sua explicação com Grad-CAM.

## Observações de reprodutibilidade

- A amostragem usada na divisão por raça possui `random_state=42`.
- Os dados baixados, pesos (`*.pt`) e resultados gerados ficam fora do Git por padrão.
- As versões efetivamente usadas em uma execução podem ser registradas com `pip freeze > environment-lock.txt`.
- O treinamento requer mais memória e é consideravelmente mais rápido com GPU.

## Limitações

- O classificador reconhece apenas as raças de gato presentes no Oxford-IIIT Pet.
- Gatos sem raça definida ou raças ausentes sempre serão associados a uma classe conhecida.
- Mudanças de iluminação, pose, enquadramento e fundo podem reduzir a qualidade fora do dataset.
- A detecção usa pesos genéricos do COCO e não foi ajustada especificamente para este conjunto.
