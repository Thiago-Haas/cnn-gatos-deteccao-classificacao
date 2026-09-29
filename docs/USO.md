# Guia de uso

[Documentação](README.md) · [Apresentação do projeto](../README.md)

Execute os comandos na raiz do repositório.

## Preparar o ambiente

Para usar a câmera do celular com processamento no navegador, veja o projeto
[ONNX Web/PWA mobile-onnx](https://github.com/Thiago-Haas/mobile-onnx). Ele oferece
seleção de fotos, câmera e cache offline, sem servidor de inferência.
A exportação dos modelos fica neste repositório: veja o
[guia de exportação para web](EXPORTACAO_WEB.md).

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

## Usar a webcam

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
O classificador só reconhece as raças presentes no checkpoint. A webcam aplica
a regra de baixa confiança descrita abaixo para exibir SRD.

## Sem Raça Definida (SRD)

Na webcam, uma confiança abaixo de **40%** exibe **Sem Raça Definida** em
laranja. Ajuste esse limite com `./run_webcam.sh --breed-conf 0.40`.
O percentual continua sendo o da raça mais provável; não representa uma
probabilidade de SRD. Essa regra não cria uma classe treinada nem comprova
raça ou genealogia. Um gato SRD ainda pode receber uma sugestão com confiança alta.
A aplicação web usa a mesma regra com um controle ajustável na interface.
