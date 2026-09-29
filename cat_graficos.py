"""Gráficos do trabalho, no mesmo estilo visual da apresentação.

Fica ao lado do notebook, como `cat_datasets.py` e `cat_experimentos.py`. Cada
função devolve a figura e, quando recebe `salvar=True`, grava o PNG em
`artifacts/figuras/` — é de lá que a apresentação puxa as imagens, então rodar
o notebook mantém o deck e o relatório sincronizados com a última execução.

Paleta: o azul `#005D98` é o do tema do deck; o laranja `#C8531F` marca o que
merece atenção (classes fracas, a regra SRD, o sobreajuste).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

AZUL = "#005D98"
AZUL_CLARO = "#7FB2D6"
LARANJA = "#C8531F"
CINZA = "#9AA3AA"
TEXTO = "#2B3A44"

FIG_DIR = Path("artifacts") / "figuras"


def definir_saida(caminho) -> Path:
    """Define onde os PNGs são gravados (padrão: artifacts/figuras)."""
    global FIG_DIR
    FIG_DIR = Path(caminho)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR


def _limpar(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(CINZA)
    ax.tick_params(colors=TEXTO, labelsize=9.5, length=0)


def _virgula(x) -> str:
    return f"{x:g}".replace(".", ",")


def _salvar(fig, nome: str, salvar: bool):
    if not salvar:
        return fig
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    destino = FIG_DIR / nome
    fig.savefig(destino, dpi=190, bbox_inches="tight")
    print(f"figura salva em {destino}")
    return fig


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #

def distribuicao_classes(cats_df: pd.DataFrame, salvar: bool = True):
    """Imagens por raça, separadas por fonte. Evidencia o desbalanceamento."""
    tabela = (pd.crosstab(cats_df["breed_name"], cats_df["source"])
              .sort_index(ascending=False))
    fontes = list(tabela.columns)
    rotulos = {"atharva_taras": "Cat Breeds Dataset", "oxford_iiit_pet": "Oxford-IIIT Pet"}
    cores = [AZUL, AZUL_CLARO, CINZA]

    nomes = [n.replace("_", " ") for n in tabela.index]
    y = np.arange(len(nomes))
    total = tabela.sum(axis=1).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    esquerda = np.zeros(len(nomes))
    for i, fonte in enumerate(fontes):
        valores = tabela[fonte].to_numpy()
        ax.barh(y, valores, left=esquerda, height=0.72,
                color=cores[i % len(cores)], label=rotulos.get(fonte, fonte))
        esquerda += valores

    for i, t in enumerate(total):
        ax.text(t + total.max() * 0.015, i, str(int(t)), va="center",
                fontsize=8.5, color=TEXTO)

    ax.set_yticks(y)
    ax.set_yticklabels(nomes)
    ax.set_xlim(0, total.max() * 1.42)
    ax.set_xlabel("imagens após a limpeza", color=TEXTO, fontsize=10)
    ax.set_title(f"{len(nomes)} classes · {int(total.sum()):,} imagens · "
                 f"{int(total.min())} a {int(total.max())} por raça".replace(",", "."),
                 color=TEXTO, fontsize=12.5, fontweight="bold", loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right",
              bbox_to_anchor=(1.0, 0.02), labelcolor=TEXTO)
    _limpar(ax)
    ax.grid(axis="x", alpha=0.18, color=CINZA)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _salvar(fig, "dist_classes.png", salvar)


# --------------------------------------------------------------------------- #
# Arquitetura
# --------------------------------------------------------------------------- #

def pipeline_diagrama(salvar: bool = True):
    """Diagrama das etapas: detecção, recorte, classificação e regra SRD."""
    etapas = [
        ("Foto ou quadro da câmera", "entrada RGB", CINZA, "#F1F4F6"),
        ("YOLOv8n · COCO", "detecta o gato\npesos genéricos, sem treino", AZUL, "#E6EEF5"),
        ("Recorte 224 × 224", "normalização ImageNet", CINZA, "#F1F4F6"),
        ("ResNet50 · transfer learning", "backbone ImageNet\nfine-tuning parcial", AZUL, "#E6EEF5"),
        ("Softmax · 22 raças", "p₁ = confiança da 1ª classe", AZUL, "#E6EEF5"),
        ("Regra SRD · p₁ < limiar", "exibe “Sem Raça Definida”", LARANJA, "#FAEAE2"),
    ]
    altura, gap, topo = 1.55, 0.58, 12.7

    fig, ax = plt.subplots(figsize=(5.4, 6.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 13.2)
    ax.axis("off")

    for i, (titulo, sub, borda, fundo) in enumerate(etapas):
        y0 = topo - i * (altura + gap) - altura
        ax.add_patch(FancyBboxPatch((0.45, y0), 9.1, altura,
                                    boxstyle="round,pad=0.02,rounding_size=0.16",
                                    facecolor=fundo, edgecolor=borda, linewidth=1.6))
        ax.text(5.0, y0 + altura * 0.63, titulo, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=TEXTO)
        ax.text(5.0, y0 + altura * 0.26, sub, ha="center", va="center",
                fontsize=8.6, color=TEXTO, linespacing=1.35)
        if i < len(etapas) - 1:
            ax.add_patch(FancyArrowPatch((5.0, y0 - 0.06), (5.0, y0 - gap + 0.06),
                                         arrowstyle="-|>", mutation_scale=13,
                                         color=CINZA, linewidth=1.4))

    ax.text(5.0, -0.25, "Grad-CAM inspeciona a decisão do classificador",
            ha="center", fontsize=8.6, color=CINZA, style="italic")
    fig.tight_layout()
    return _salvar(fig, "pipeline.png", salvar)


# --------------------------------------------------------------------------- #
# Treinamento
# --------------------------------------------------------------------------- #

def historico(hist: dict, inicio_ft: int = None, nome: str = "historico.png",
              salvar: bool = True):
    """Loss e acurácia por época, com marca no início do fine-tuning."""
    epocas = np.arange(1, len(hist["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.2), sharex=True)

    for ax, kt, kv, titulo in ((ax1, "train_loss", "val_loss", "Loss"),
                               (ax2, "train_acc", "val_acc", "Acurácia")):
        ax.plot(epocas, hist[kt], color=AZUL, linewidth=2, label="treino")
        ax.plot(epocas, hist[kv], color=LARANJA, linewidth=2, linestyle="--",
                label="validação")
        if inicio_ft:
            ax.axvline(inicio_ft + 0.5, color=CINZA, linestyle=":", linewidth=1.4)
        ax.set_title(titulo, color=TEXTO, fontsize=12, fontweight="bold", loc="left")
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([_virgula(t) for t in ax.get_yticks()])
        _limpar(ax)
        ax.grid(axis="y", alpha=0.18, color=CINZA)
        ax.set_axisbelow(True)

    if inicio_ft:
        ax1.text(inicio_ft + 1.2, max(hist["train_loss"]) * 0.92,
                 "início do\nfine-tuning", fontsize=9, color=CINZA, linespacing=1.3)
    for chave, cor, dy in (("train_acc", AZUL, -2), ("val_acc", LARANJA, -4)):
        ax2.annotate(f"{hist[chave][-1]:.2f}".replace(".", ","),
                     (epocas[-1], hist[chave][-1]), xytext=(6, dy),
                     textcoords="offset points", color=cor, fontweight="bold", fontsize=10)

    ax2.set_xlabel("época", color=TEXTO, fontsize=10)
    ax2.set_xlim(1, len(epocas) + 3)
    ax2.legend(frameon=False, loc="lower right", fontsize=9.5, ncols=2)
    fig.tight_layout()
    return _salvar(fig, nome, salvar)


# --------------------------------------------------------------------------- #
# Avaliação
# --------------------------------------------------------------------------- #

def f1_por_classe(metricas: dict, limiar_atencao: float = 0.70,
                  nome: str = "f1_por_classe.png", salvar: bool = True):
    """F1 por raça, ordenado. Em laranja, as classes abaixo do limiar."""
    f1_macro = metricas["f1_macro"]
    itens = sorted(((k.replace("_", " "), v["f1-score"])
                    for k, v in metricas["relatorio"].items() if k[0].isupper()),
                   key=lambda x: x[1])
    nomes = [i[0] for i in itens]
    vals = np.array([i[1] for i in itens])
    fracas = int((vals < limiar_atencao).sum())

    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    y = np.arange(len(nomes))
    ax.barh(y, vals, height=0.72,
            color=[LARANJA if v < limiar_atencao else AZUL for v in vals])
    ax.axvline(f1_macro, color=TEXTO, linestyle="--", linewidth=1.2)
    ax.text(f1_macro + 0.012, len(nomes) - 0.4,
            f"F1-macro {f1_macro:.2f}".replace(".", ","),
            fontsize=9.5, color=TEXTO, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(v - 0.018, i, f"{v:.2f}".replace(".", ","), va="center", ha="right",
                fontsize=8.5, color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(nomes)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0,2", "0,4", "0,6", "0,8", "1,0"])
    ax.set_xlabel("F1-score no conjunto de teste", color=TEXTO, fontsize=10)
    ax.set_title(f"{fracas} das {len(nomes)} classes ficam abaixo de "
                 f"{_virgula(limiar_atencao)} de F1",
                 color=TEXTO, fontsize=12, fontweight="bold", loc="left", pad=12)
    _limpar(ax)
    ax.grid(axis="x", alpha=0.18, color=CINZA)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _salvar(fig, nome, salvar)


def matriz_confusao(labels, preds, breed_names, nome: str = "matriz_confusao.png",
                    salvar: bool = True, normalizar: bool = True):
    """Matriz de confusão. Normalizada por linha, para comparar classes de
    tamanhos diferentes — sem isso as raças grandes dominam a escala de cor."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds).astype(float)
    if normalizar:
        soma = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, soma, out=np.zeros_like(cm), where=soma > 0)

    nomes = [n.replace("_", " ") for n in breed_names]
    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalizar else None)

    ax.set_xticks(np.arange(len(nomes)))
    ax.set_yticks(np.arange(len(nomes)))
    ax.set_xticklabels(nomes, rotation=90, fontsize=8.5)
    ax.set_yticklabels(nomes, fontsize=8.5)
    ax.set_xlabel("previsto", color=TEXTO, fontsize=10.5)
    ax.set_ylabel("real", color=TEXTO, fontsize=10.5)
    ax.set_title("Matriz de confusão — proporção por classe real",
                 color=TEXTO, fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.tick_params(colors=TEXTO, length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)

    # Destaca em vermelho as confusões fora da diagonal acima de 15%.
    for i in range(len(nomes)):
        for j in range(len(nomes)):
            if i != j and cm[i, j] >= 0.15:
                ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center",
                        fontsize=7.5, color=LARANJA, fontweight="bold")

    barra = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    barra.ax.tick_params(colors=TEXTO, labelsize=9, length=0)
    barra.outline.set_visible(False)
    fig.tight_layout()
    return _salvar(fig, nome, salvar)


# --------------------------------------------------------------------------- #
# Experimentos
# --------------------------------------------------------------------------- #

def comparativo_experimentos(tabela: pd.DataFrame, salvar: bool = True):
    """Acurácia e F1 por experimento, ao lado do gap treino−validação."""
    tab = tabela.dropna(subset=["Acurácia (teste)"])
    if tab.empty:
        print("Nenhum experimento com métrica de teste para plotar.")
        return None

    x = np.arange(len(tab))
    larg = 0.36
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.0),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    axA.bar(x - larg / 2, tab["Acurácia (teste)"], larg, color=AZUL, label="Acurácia")
    axA.bar(x + larg / 2, tab["F1-macro (teste)"], larg, color=AZUL_CLARO, label="F1-macro")
    for i, (a, f) in enumerate(zip(tab["Acurácia (teste)"], tab["F1-macro (teste)"])):
        axA.text(i - larg / 2, a + 0.015, f"{a:.3f}".replace(".", ","),
                 ha="center", fontsize=9.5, color=TEXTO, fontweight="bold")
        axA.text(i + larg / 2, f + 0.015, f"{f:.3f}".replace(".", ","),
                 ha="center", fontsize=9.5, color=TEXTO)
    axA.set_ylim(0, 1.06)
    axA.set_title("Desempenho no teste", color=TEXTO, fontsize=12,
                  fontweight="bold", loc="left")
    axA.legend(frameon=False, fontsize=9.5, loc="upper right", ncols=2)

    axB.bar(x, tab["Gap treino−val"], 0.46, color=LARANJA)
    for i, g in enumerate(tab["Gap treino−val"]):
        axB.text(i, g + 0.005, f"{g:.3f}".replace(".", ","),
                 ha="center", fontsize=9.5, color=TEXTO, fontweight="bold")
    axB.set_ylim(0, max(tab["Gap treino−val"]) * 1.28)
    axB.set_title("Gap treino − validação", color=TEXTO, fontsize=12,
                  fontweight="bold", loc="left")

    for ax in (axA, axB):
        ax.set_xticks(x)
        ax.set_xticklabels(tab["Exp."], fontsize=11)
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([_virgula(t) for t in ax.get_yticks()])
        _limpar(ax)
        ax.grid(axis="y", alpha=0.18, color=CINZA)
        ax.set_axisbelow(True)

    fig.tight_layout()
    return _salvar(fig, "comparativo_experimentos.png", salvar)


# --------------------------------------------------------------------------- #
# Regra SRD
# --------------------------------------------------------------------------- #

def srd_calibracao(analise: dict, salvar: bool = True):
    """Diagrama de confiabilidade antes e depois da calibração."""
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], color=CINZA, linestyle=":", label="calibração perfeita")
    for bins, cor, rotulo in ((analise["bins_antes"], AZUL, "antes"),
                              (analise["bins_depois"], LARANJA,
                               f"depois (T = {analise['temperatura']:.2f})".replace(".", ","))):
        ax.plot([b["confianca"] for b in bins], [b["acuracia"] for b in bins],
                marker="o", markersize=4, color=cor, label=rotulo)
    ax.set_xlabel("confiança média do softmax", color=TEXTO, fontsize=10)
    ax.set_ylabel("acurácia observada", color=TEXTO, fontsize=10)
    ax.set_title("Diagrama de confiabilidade", color=TEXTO, fontsize=12,
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9.5)
    _limpar(ax)
    ax.grid(alpha=0.18, color=CINZA)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _salvar(fig, "srd_calibracao.png", salvar)


def srd_limiar(analise: dict, limiar_atual: float = 0.40, salvar: bool = True):
    """Cobertura e acurácia no teste, e rejeição no conjunto de controle SRD."""
    v = analise["varreduras"]
    conf = v[v["criterio"] == "confianca"]
    ctrl = analise.get("controle")
    recom = (analise["resumo"].get("limiar_recomendado_confianca") or {}).get("limiar")

    marcas = [(limiar_atual, LARANJA, f"{_virgula(limiar_atual)} em uso")]
    if recom:
        alvo = analise["resumo"]["alvo_acuracia"]
        marcas.append((recom, AZUL, f"{_virgula(recom)} alvo {alvo:.0%}"))

    n_paineis = 2 if ctrl is not None else 1
    fig, eixos = plt.subplots(n_paineis, 1, figsize=(6.0, 3.3 * n_paineis))
    eixos = np.atleast_1d(eixos)
    axA = eixos[0]

    axA.plot(conf["limiar"], conf["cobertura"], color=AZUL, linewidth=2.4,
             marker="o", markersize=4, label="cobertura (recebem raça)")
    axA.plot(conf["limiar"], conf["acuracia_aceitos"], color=LARANJA, linewidth=2.4,
             linestyle="--", marker="o", markersize=4, label="acurácia entre os aceitos")
    axA.set_title("Conjunto de teste", color=TEXTO, fontsize=12,
                  fontweight="bold", loc="left")
    axA.legend(frameon=False, fontsize=9, loc="lower left")

    if ctrl is not None:
        axB = eixos[1]
        axB.plot(ctrl["limiar"], ctrl["taxa_rejeicao_confianca"], color=AZUL,
                 linewidth=2.4, marker="o", markersize=4)
        n = analise["resumo"].get("controle_srd", {}).get("n_fotos", "")
        axB.set_title(f"Controle: {n} gatos SRD · acertar é rejeitar", color=TEXTO,
                      fontsize=12, fontweight="bold", loc="left")
        axB.set_ylabel("rejeitadas como SRD", color=TEXTO, fontsize=9.5)

    for ax in eixos:
        for lim, cor, rot in marcas:
            ax.axvline(lim, color=cor, linestyle=":", linewidth=1.5)
            ax.text(lim + 0.014, 1.02, rot, fontsize=8.4, color=cor, fontweight="bold")
        ax.set_xlabel("limiar de confiança", color=TEXTO, fontsize=10)
        ax.set_xlim(0, 0.96)
        ax.set_ylim(0, 1.04)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xticklabels(["0", "0,2", "0,4", "0,6", "0,8"])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
        _limpar(ax)
        ax.grid(alpha=0.18, color=CINZA)
        ax.set_axisbelow(True)

    fig.tight_layout()
    return _salvar(fig, "srd_limiar.png", salvar)


def todas_as_figuras() -> list[Path]:
    """Lista os PNGs gerados — os mesmos que a apresentação usa."""
    if not FIG_DIR.exists():
        return []
    return sorted(FIG_DIR.glob("*.png"))
