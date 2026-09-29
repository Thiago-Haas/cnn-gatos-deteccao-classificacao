# Ferramentas e tecnologias

[Documentação](README.md) · [Apresentação do projeto](../README.md)

| Ferramenta | Papel no projeto |
| --- | --- |
| Python, JupyterLab e ipykernel | Execução do notebook e experimentos |
| PyTorch e torchvision | ResNet50, treinamento, transformações e checkpoints |
| Ultralytics | YOLOv8n para detecção e YOLO Pose opcional para pontos faciais |
| OpenCV e Pillow | Leitura, recortes, anotações e webcam |
| NumPy e pandas | Tensores auxiliares, tabelas e manifestos dos datasets |
| scikit-learn | Divisão dos dados e métricas de classificação |
| Matplotlib e seaborn | Gráficos e visualizações dos resultados |
| grad-cam | Mapas de ativação do classificador |
| PyYAML | Configuração dos dados de pose |
| ONNX e ONNX Runtime | Exportação e comparação numérica com PyTorch |

As dependências do notebook estão em [requirements.txt](../requirements.txt).
As da exportação estão em [requirements-export.txt](../requirements-export.txt).
A aplicação web tem ambiente e dependências próprios.

## Créditos

Consulte [Dados e rótulos](DADOS.md) para as fontes de classificação e
[Pontos faciais](PONTOS_FACIAIS.md) para o CAT Dataset e sua referência acadêmica.
Datasets, modelos e bibliotecas mantêm suas respectivas licenças; a inclusão
neste projeto não altera as condições de redistribuição de cada componente.
