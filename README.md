# Detecção e classificação de raças de gatos

Projeto final da disciplina de Aprendizado Profundo. O notebook implementa um pipeline de visão computacional que detecta gatos em imagens, classifica suas raças e explica as previsões com mapas de ativação.

## Visão geral

O projeto usa:

- **Oxford-IIIT Pet** e **Cat Breeds Dataset**, de Atharva Taras, como conjuntos de dados;
- **ResNet50** pré-treinada na ImageNet para classificar as raças;
- **YOLOv8 Nano** pré-treinado no COCO para detectar e recortar gatos;
- **YOLOv8 Pose** opcional, treinado no CAT Dataset para localizar nove pontos faciais antes da raça;
- **Grad-CAM** para visualizar as regiões que influenciam a classificação;
- acurácia, F1-score macro, relatório de classificação e matriz de confusão para avaliação.

As imagens de gatos das duas fontes são combinadas, os rótulos são padronizados e duplicatas RGB exatas são removidas antes da divisão estratificada em 70% para treino, 15% para validação e 15% para teste.

## Estrutura

```text
.
├── cnn_gato_deteccao_classificacao.ipynb  # implementação e análise
├── cat_datasets.py                        # download, rótulos e limpeza dos datasets
├── cat_landmarks.py                       # CAT Dataset e inferência de pontos faciais
├── train_cat_landmarks.py                 # treino separado dos pontos faciais
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

Para testar fotos externas, crie uma subpasta por raça dentro de `real_photos/`, como `real_photos/Siamese/foto.jpg`. O nome da subpasta será usado como rótulo real. A pasta é criada automaticamente e não é versionada.

O notebook baixa automaticamente o Oxford-IIIT Pet, o Cat Breeds Dataset, os pesos da ResNet50 e o modelo `yolov8n.pt`. Esses artefatos exigem conexão com a internet e não são versionados.

## Dataset adicional e rótulos

O [Cat Breeds Dataset](https://github.com/AtharvaTaras/Cat-Breeds-Dataset)
contém 4.835 imagens em 20 raças antes da limpeza (cerca de 303 MB descompactados).
O download está fixado no commit `56a69059119626ed210023bbfc6fdcb8262e8703`
e fica em `data/`. Mantenha `cat_datasets.py` ao lado do notebook, inclusive se
copiar o projeto para o Colab. Selecione a fonte em `CONFIG["DATASET_MODE"]`:

| Valor | Dataset usado | Raças |
|---|---|---|
| `"oxford"` | Apenas Oxford-IIIT Pet | 12 |
| `"cat_breeds"` | Apenas Cat Breeds Dataset, de Atharva Taras | 20 |
| `"both"` | Os dois (padrão) | 22 |

Somente as fontes escolhidas são baixadas e lidas. Ao mudar o valor, reinicie
o kernel e execute todas as células para atualizar os rótulos, splits e modelo.
O checkpoint registra as fontes usadas. Cada treinamento salva no mesmo
`artifacts/cat_breed_classifier.pt`; copie o arquivo antes se quiser preservar
modelos de experimentos anteriores.

As pastas usam nomes como `maine_coon_cat`; a integração remove `_cat` e adota
o padrão do Oxford, como `Maine_Coon`, `British_Shorthair` e `Russian_Blue`.
As dez raças adicionais são `American_Shorthair`, `Cornish_Rex`, `Devon_Rex`,
`Himalayan`, `Manx`, `Norwegian_Forest`, `Oriental_Shorthair`, `Savannah`,
`Scottish_Fold` e `Turkish_Angora`. A união contém **22 classes**; `Bombay` e
`Egyptian_Mau` continuam vindo do Oxford.

Antes do split, a integração remove imagens ilegíveis, mantém uma cópia de
imagens com pixels RGB idênticos e exclui todas as cópias quando os rótulos
conflitam. Os arquivos `artifacts/dataset_manifest.csv`, `dataset_rejections.csv`
e `{train,val,test}_manifest.csv` registram a origem, os rótulos e as exclusões.
Fotos semelhantes, recortadas ou recomprimidas não são detectadas por essa limpeza.

Execute novamente o notebook para treinar e exportar o modelo com as 22 classes.
O `.pt` antigo continua reconhecendo apenas suas raças originais. A webcam lê
o mapeamento salvo no novo checkpoint automaticamente. Os resultados antigos
do notebook foram limpos para evitar confundi-los com uma avaliação das 22 classes.

**Citation: Atharva Taras, 2026. Cat Breeds Dataset.**
[Fonte](https://github.com/AtharvaTaras/Cat-Breeds-Dataset) ·
[Licença CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Alterações realizadas na integração: padronização dos rótulos e exclusão de
imagens ilegíveis, duplicatas exatas e conflitos de rótulo. A licença e o README
originais acompanham o dataset baixado.

## Pipeline

### Pontos faciais antes da raça

A seção **8.5** do notebook adiciona o fluxo: detecção do gato → recorte →
detecção dos pontos faciais → classificação da raça → Grad-CAM.
O [CAT Dataset, disponibilizado por Chris Crawford](https://www.kaggle.com/datasets/crawford/cat-dataset/data)
anota nove pontos: dois olhos, um ponto da boca e três pontos em cada orelha.
Ele não tem rótulos de raça, cor ou pelagem; seu modelo é treinado separadamente
e funciona com qualquer opção de `DATASET_MODE`.

No `CONFIG` do notebook, use:

```python
"LANDMARKS_MODE": "train",  # "train", "load" ou "off" (padrão)
"LANDMARKS_EPOCHS": 30,
"LANDMARKS_BATCH": 16,
"LANDMARKS_IMG_SIZE": 640,
"LANDMARKS_CONF": 0.25,
"LANDMARKS_SHOW_RESULTS": True,
"LANDMARKS_EXAMPLES": 3,
```

`train` baixa/prepara os dados, treina e exporta `artifacts/cat_landmarks.pt`.
Nas execuções seguintes, use `load` para reutilizar esse modelo. `off` mantém
a classificação sem pontos faciais. A célula 8.5 pode ser executada após setup
e CONFIG, sem treinar primeiro o classificador de raças.

A célula de resultados exibe as métricas salvas do teste, gráficos de loss de
treino/validação e mAP de pose, além de exemplos do teste com anotações e previsões
lado a lado. Cada exemplo mostra uma tabela com o nome, coordenadas, confiança
e status dos nove pontos. No pipeline de fotos, há um painel próprio de features
entre o recorte e o Grad-CAM; a webcam do notebook também atualiza a tabela a cada frame.
Com `off`, esses painéis e tabelas são omitidos. Para ocultar apenas o diagnóstico
da seção 8.5, use `LANDMARKS_SHOW_RESULTS=False`; para ocultar seus exemplos, use
`LANDMARKS_EXAMPLES=0`. Os novos treinamentos exportam também `cat_landmarks.csv`
com o histórico. Checkpoints antigos sem histórico ou métricas continuam utilizáveis,
e o notebook informa quais resultados estão indisponíveis.

Também é possível treinar pelo terminal:

```bash
source .venv/bin/activate
python train_cat_landmarks.py --epochs 30
./run_webcam.sh --landmarks artifacts/cat_landmarks.pt
```

Para dados já baixados, use `--raw /caminho/extraido` ou `--archive /caminho/cat.zip`.
`--prepare-only` apenas converte os dados. Ajuste `--batch 8` se faltar memória
na GPU, ou `--device cpu` para usar CPU. Os pesos iniciais `yolov8n-pose.pt`
são baixados no primeiro treinamento; o modelo humano não serve diretamente
para localizar pontos faciais de gatos.

O download da versão 2 ocupa vários GB e fica em `data/cat_landmarks_raw_v2/`.
A conversão em `data/cat_landmarks_pose/` reúne imagens de pixels idênticos e
preserva suas anotações distintas, inclusive quando há várias cabeças. Variantes
com o mesmo identificador de foto são agrupadas antes do split 70/15/15 **por
grupo**, portanto as proporções por imagem podem variar. Os hashes e grupos
estão em `manifest.csv`, e problemas de leitura/anotação em `rejections.csv`.
Na validação local, a preparação resultou em 9.936 imagens únicas e 1.331 grupos.
Fotos semelhantes com identificadores diferentes ainda podem exigir revisão.

Os arquivos `.cat` são convertidos para YOLO Pose com `kpt_shape: [9, 3]`.
A caixa da cabeça é derivada dos pontos com margem de 15%; pontos fora da imagem
ficam sem supervisão. As imagens são regravadas em JPEG sem orientação EXIF para
manter as coordenadas. Espelhamentos são desativados para preservar a ordem dos pontos.
O melhor checkpoint é avaliado no teste separado; as métricas ficam em
`artifacts/cat_landmarks.json`, e o histórico em `artifacts/landmark_runs/`.
O mAP de pose usa OKS uniforme da Ultralytics para nove pontos e não é comparável
à acurácia/F1 de raça. [Formato e métricas de pose](https://docs.ultralytics.com/datasets/pose/).

Na visualização, 1–2 são olhos, 3 é boca, 4–6 são orelha esquerda e 7–9 são orelha
direita, seguindo a nomenclatura do dataset. Só são desenhados pontos com confiança
de pelo menos 0,5. Sem cabeça detectada, a classificação prossegue. O pipeline
retorna `facial_features`, com coordenadas relativas ao recorte e confiança.
Os pontos são estimados antes da raça, mas a ResNet continua recebendo o recorte
RGB original: esta etapa não acrescenta entradas ao classificador nem garante
melhora nas métricas de raça. O modelo de pontos precisa ser treinado antes do uso.

Referência: **Weiwei Zhang, Jian Sun e Xiaoou Tang. Cat Head Detection — How to
Effectively Exploit Shape and Texture Features. ECCV, 2008, pp. 802–816.**
[Artigo original](https://www.microsoft.com/en-us/research/wp-content/uploads/2008/10/ECCV_CAT_PROC.pdf).
O espelho Kaggle declara CC0; a integração transforma as anotações e as imagens
conforme descrito acima.

### Usar o modelo treinado na webcam

Com o ambiente instalado, execute na raiz do projeto:

```bash
./run_webcam.sh
```

O lançador ativa a `.venv` automaticamente e aceita as opções do script Python.

O script carrega `artifacts/cat_breed_classifier.pt` sem executar o notebook nem
treinar novamente. Para usar o treinamento do Colab, baixe
`/content/cat_breed_classifier.pt` e coloque-o nessa pasta, ou informe outro caminho:

```bash
./run_webcam.sh --model ~/Downloads/cat_breed_classifier.pt --camera 0
```

A janela mostra a raça e a confiança para cada gato detectado. Pressione **Q**
ou **Esc** para sair. Se necessário, tente `--camera 1` para outra câmera ou
`--device cpu` para executar sem CUDA. Use `python webcam.py --help` para todas
as opções. É necessário executar em uma sessão gráfica local com acesso à webcam.

São aceitos os checkpoints das versões original e corrigida do notebook. O
pré-processamento usa resolução 224 e normalização ImageNet; se você alterou
`IMG_SIZE` no treino, informe o mesmo valor em `--image-size`. Os pesos YOLO
`yolov8n.pt` são usados separadamente para detectar gatos e baixados se ausentes.
O classificador só reconhece as raças do treinamento, incluindo ao mostrar
resultados para gatos sem raça definida.

### Etapas do notebook

1. Baixar e explorar o Oxford-IIIT Pet e o Cat Breeds Dataset.
2. Filtrar os gatos, unificar rótulos e remover imagens inválidas e duplicadas.
3. Fazer um split estratificado e reprodutível de 70/15/15.
4. Aplicar redimensionamento, normalização e data augmentation configurável.
5. Treinar uma cabeça com Dropout sobre a ResNet50 congelada.
6. Fazer fine-tuning parcial das camadas `layer3`, `layer4` e `fc`.
7. Controlar o treino com label smoothing, weight decay, scheduler e early stopping.
8. Avaliar no conjunto de teste.
9. Detectar gatos em novas imagens com YOLOv8.
10. Estimar os pontos faciais (opcional), classificar o recorte e gerar Grad-CAM.
11. Avaliar um dataset local organizado por raça.
12. Executar detecção e classificação com a webcam local via OpenCV.

### Controle de overfitting

As duas fases de treinamento usam early stopping baseado na perda de validação e restauram automaticamente os pesos da melhor época:

- treinamento da camada classificadora: `HEAD_PATIENCE=5`;
- fine-tuning de `layer3`, `layer4` e `fc`: `FT_PATIENCE=5`;
- melhoria mínima considerada pela implementação: `min_delta=0.0001`.

Os melhores pesos são restaurados ao fim de cada fase. O gráfico reúne loss e acurácia de treino/validação e marca o início do fine-tuning.

## Observações de reprodutibilidade

- A amostragem usada na divisão por raça possui `random_state=42`.
- Os dados baixados, pesos (`*.pt`) e resultados gerados ficam fora do Git por padrão.
- As versões efetivamente usadas em uma execução podem ser registradas com `pip freeze > environment-lock.txt`.
- O treinamento requer mais memória e é consideravelmente mais rápido com GPU.

## Limitações

- O classificador reconhece apenas as raças presentes nos datasets usados no treinamento.
- Gatos sem raça definida ou raças ausentes sempre serão associados a uma classe conhecida.
- Mudanças de iluminação, pose, enquadramento e fundo podem reduzir a qualidade fora do dataset.
- A detecção usa pesos genéricos do COCO e não foi ajustada especificamente para este conjunto.
