# Dados e rótulos

[Documentação](README.md) · [Apresentação do projeto](../README.md)

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
o mapeamento salvo no novo checkpoint automaticamente. As saídas salvas no notebook correspondem à execução que as gerou; confira
a configuração e as classes do checkpoint antes de comparar resultados.

**Citation: Atharva Taras, 2026. Cat Breeds Dataset.**
[Fonte](https://github.com/AtharvaTaras/Cat-Breeds-Dataset) ·
[Licença CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Alterações realizadas na integração: padronização dos rótulos e exclusão de
imagens ilegíveis, duplicatas exatas e conflitos de rótulo. A licença e o README
originais acompanham o dataset baixado.
