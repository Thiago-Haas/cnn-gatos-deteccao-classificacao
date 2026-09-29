# Treinamento e avaliação

[Documentação](README.md) · [Apresentação do projeto](../README.md)

## Etapas do notebook

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

## Interpretação dos resultados

Acurácia, F1 macro, relatório por classe e matriz de confusão são calculados
no teste separado. Grad-CAM mostra regiões que influenciam a previsão;
não demonstra causalidade nem comprova a raça do animal.

Os [pontos faciais](PONTOS_FACIAIS.md) são opcionais e não alteram as entradas
RGB da ResNet50. A [regra SRD](USO.md#sem-raça-definida-srd) é aplicada na
webcam e no app web sobre a confiança do classificador.

## Limitações

- O classificador aprende somente as raças presentes nos datasets selecionados.
- Baixa confiança pode indicar SRD, raça ausente ou dificuldade da imagem;
  confiança alta também pode estar errada.
- Iluminação, pose, enquadramento e fundo afetam o resultado em fotos externas.
- O detector usa pesos genéricos COCO, sem ajuste específico neste projeto.
- A remoção de duplicatas exatas não elimina todas as imagens semelhantes.
