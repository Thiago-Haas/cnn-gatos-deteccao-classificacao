# Exportar os modelos para o app web

[Documentação](README.md) · [Projeto de treinamento](../README.md) · [Aplicação mobile-onnx](https://github.com/Thiago-Haas/mobile-onnx)

O notebook treina o classificador e salva `artifacts/cat_breed_classifier.pt`.
O detector é o `yolov8n.pt` pré-treinado no COCO. Este pipeline converte os dois
checkpoints para ONNX e verifica a paridade numérica com PyTorch.

## Gerar o pacote

Depois de concluir o treinamento, execute na raiz deste repositório:

```bash
./export_web_models.sh
```

O script usa/cria `.venv/`, instala as dependências de exportação se estiverem
ausentes e chama `scripts/export_web_models.py`. A saída padrão é:

```text
artifacts/onnx/
  classifier.onnx
  detector.onnx
  metadata.json
```

`artifacts/` permanece ignorado pelo Git deste projeto. Os arquivos são publicados
pelo repositório do app. Para informar checkpoints ou outro destino:

```bash
./export_web_models.sh \
  --classifier /caminho/classificador.pt \
  --detector /caminho/yolov8n.pt \
  --output /caminho/pacote-onnx
```

Também é possível usar diretamente o Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-export.txt
python scripts/export_web_models.py
```

## Atualizar o aplicativo

Copie os **três arquivos juntos** para `models/` na raiz de
[mobile-onnx](https://github.com/Thiago-Haas/mobile-onnx). Se os repositórios
estiverem lado a lado, você pode exportar diretamente para essa pasta:

```bash
./export_web_models.sh --output ../mobile-onnx/models
```

Esse comando substitui o pacote existente depois das verificações de exportação.
Preserve uma cópia se quiser manter o modelo anterior. No repositório do app,
execute `./run.sh` para preparar o site e testar uma foto antes de publicar.
O build verifica os hashes e gera a versão do cache a partir dos pesos e da
configuração; um pacote novo passa a ser baixado pelos usuários após o deploy.

## Contrato e validação

- Classificador ResNet50 com cabeça Linear ou Dropout + Linear.
- Rótulos lidos de `idx_to_breed`, na ordem usada pelo checkpoint.
- Tamanho do classificador lido de `config.IMG_SIZE` (padrão 224).
- Detector COCO com 80 classes, gato no índice 15, entrada 320 × 320 por padrão.
- Exportação FP32/opset 17; `--detector-size` aceita múltiplos de 32.
- Validação dos grafos com `onnx.checker` e comparação PyTorch/ONNX em tensores
  reprodutíveis. Os erros máximos ficam em `metadata.json`.
- Origem em `training`: experimento, SHA-256 do checkpoint e fontes selecionadas.
  Checkpoints antigos sem esses campos continuam aceitos.
- Hash SHA-256 dos dois ONNX em `modelHashes`, para detectar arquivos misturados
  ou modificados e permitir atualização automática do cache no app.

A comparação numérica não substitui avaliação de acurácia em fotos reais.
Landmarks e Grad-CAM não são exportados para a aplicação web.

O checkpoint padrão atual corresponde ao experimento E2. O exportador lê
`config/srd.json`, verifica o SHA-256 do checkpoint e publica o perfil em
`classifierCalibration`. Para outro estudo, use `--calibration /caminho/srd.json`.
Para exportar sem calibração, use `--no-calibration` (T=1, limiar 0,40).
O ONNX mantém logits brutos; o app aplica a temperatura no pós-processamento.
