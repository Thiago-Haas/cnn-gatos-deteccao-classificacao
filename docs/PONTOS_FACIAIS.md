# Pontos faciais

[Documentação](README.md) · [Apresentação do projeto](../README.md)

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
