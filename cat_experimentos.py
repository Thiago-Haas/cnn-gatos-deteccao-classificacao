"""Plano de experimentos e análise da regra SRD, com execução em GPU.

Fica ao lado do notebook, como `cat_datasets.py` e `cat_landmarks.py`, por dois
motivos: o notebook importa daqui em vez de repetir o código, e as classes de
`Dataset` definidas em um módulo podem ser enviadas aos processos do DataLoader
— classes definidas em célula de notebook não podem, e é por isso que o notebook
original usa `NUM_WORKERS = 0`.

Aceleração aplicada quando há GPU:

- `torch.amp` (bfloat16 quando a GPU suporta, senão float16) no treino e na
  avaliação, com `GradScaler` apenas onde float16 exige;
- `channels_last`, o formato de memória que as convoluções da ResNet preferem
  nos núcleos tensoriais;
- `cudnn.benchmark`, que escolhe o melhor algoritmo para um tamanho de entrada
  fixo — o nosso caso, sempre 224 × 224;
- TF32 nas multiplicações de matriz;
- DataLoader com vários processos, `pin_memory`, `persistent_workers` e cópia
  assíncrona para a GPU (`non_blocking`), para a decodificação das imagens não
  deixar a GPU ociosa.

`preparar_gpu()` imprime um diagnóstico e explica o motivo quando cai em CPU.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

AZUL = "#005D98"
AZUL_CLARO = "#7FB2D6"
LARANJA = "#C8531F"
CINZA = "#9AA3AA"
TEXTO = "#2B3A44"


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #

class Acelerador:
    """Reúne device, precisão mista e parâmetros de DataLoader em um objeto só."""

    def __init__(self, device: torch.device, amp_dtype=None, num_workers: int = 0):
        self.device = device
        self.amp_dtype = amp_dtype
        self.num_workers = num_workers
        self.na_gpu = device.type == "cuda"
        self.usa_amp = amp_dtype is not None
        # GradScaler só faz sentido em float16; bfloat16 tem alcance suficiente.
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.usa_amp and amp_dtype == torch.float16)
        )

    def autocast(self):
        if not self.usa_amp:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def para_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device, non_blocking=self.na_gpu)

    def kwargs_loader(self) -> dict:
        kwargs = {"num_workers": self.num_workers, "pin_memory": self.na_gpu}
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return kwargs

    def __repr__(self) -> str:
        precisao = {None: "float32",
                    torch.bfloat16: "bfloat16 (AMP)",
                    torch.float16: "float16 (AMP)"}[self.amp_dtype]
        return (f"Acelerador(device={self.device}, precisão={precisao}, "
                f"workers={self.num_workers})")


def preparar_gpu(device=None, *, amp: bool = True, num_workers=None,
                 verbose: bool = True) -> Acelerador:
    """Configura a GPU e devolve o Acelerador. Explica o motivo se cair em CPU."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if verbose:
        print(f"torch {torch.__version__}  ·  CUDA da build: {torch.version.cuda or 'nenhuma'}")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                'Foi pedido "cuda", mas torch.cuda.is_available() é False. '
                "Veja o diagnóstico abaixo rodando preparar_gpu() sem argumentos."
            )
        props = torch.cuda.get_device_properties(0)
        torch.backends.cudnn.benchmark = True        # entrada de tamanho fixo
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if verbose:
            print(f"GPU: {props.name}  ·  {props.total_memory / 1e9:.1f} GB  "
                  f"·  capability {props.major}.{props.minor}")
    elif verbose:
        # Distingue os dois motivos: build sem CUDA × build com CUDA e sem GPU visível.
        if torch.version.cuda is None:
            print("ATENÇÃO — este Python tem uma build de PyTorch SEM CUDA.\n"
                  "  Provavelmente o kernel ou o interpretador não é o .venv do projeto.\n"
                  "  No terminal:  source .venv/bin/activate\n"
                  "  No notebook:  selecione o kernel .venv/bin/python")
        else:
            print("ATENÇÃO — a build tem CUDA, mas nenhuma GPU foi encontrada.\n"
                  "  Verifique `nvidia-smi` e a variável CUDA_VISIBLE_DEVICES.")
        print("Em CPU cada experimento leva horas.")

    amp_dtype = None
    if amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if num_workers is None:
        num_workers = min(8, (os.cpu_count() or 2)) if device.type == "cuda" else 0

    acel = Acelerador(device, amp_dtype, num_workers)
    if verbose:
        print(acel)
    return acel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Configuração e catálogo de experimentos
# --------------------------------------------------------------------------- #

CONFIG_PADRAO = {
    "IMG_SIZE": 224,
    "BATCH_SIZE": 32,
    "VAL_FRAC": 0.15,
    "TEST_FRAC": 0.15,
    "SEED": 42,

    "AUG_ENABLED": True,
    "AUG_HFLIP_PROB": 0.5,
    "AUG_ROTATION_DEG": 15,
    "AUG_BRIGHTNESS": 0.2,
    "AUG_CONTRAST": 0.2,
    "AUG_SATURATION": 0.2,
    "AUG_HUE": 0.03,
    "AUG_BLUR_PROB": 0.2,
    "AUG_BLUR_KERNEL": 3,
    "AUG_BLUR_SIGMA": (0.1, 1.2),
    "AUG_ERASING_PROB": 0.25,
    "AUG_ERASING_SCALE": (0.02, 0.10),
    "AUG_ERASING_RATIO": (0.3, 3.3),

    "DROPOUT": 0.5,

    "HEAD_LR": 1e-3,
    "HEAD_WEIGHT_DECAY": 3e-4,
    "HEAD_EPOCHS": 30,
    "HEAD_PATIENCE": 5,

    "FT_UNFREEZE_LAYERS": ["layer3", "layer4", "fc"],
    "FT_LR": 1e-5,
    "FT_WEIGHT_DECAY": 3e-4,
    "FT_EPOCHS": 20,
    "FT_PATIENCE": 5,

    "LABEL_SMOOTHING": 0.05,
    "LR_SCHEDULER_FACTOR": 0.5,
    "LR_SCHEDULER_PATIENCE": 2,
}

EXPERIMENTOS = {
    "E1": {
        "nome": "Baseline — somente a cabeça",
        "variavel": "—",
        "hipotese": "Estabelecer a referência: quanto a ResNet50 da ImageNet já "
                    "resolve sem ajustar nenhuma camada convolucional.",
        "fine_tuning": False,
        "overrides": {},
    },
    "E2": {
        "nome": "Fine-tuning de layer3 + layer4 + fc",
        "variavel": "Camadas descongeladas (arquitetura)",
        "hipotese": "Ajustar os dois últimos blocos residuais especializa as features "
                    "da ImageNet em textura e formato de pelagem, elevando acurácia "
                    "e F1-macro.",
        "fine_tuning": True,
        "overrides": {"FT_UNFREEZE_LAYERS": ["layer3", "layer4", "fc"]},
    },
    "E3": {
        "nome": "Fine-tuning de layer4 + fc",
        "variavel": "Profundidade do descongelamento (arquitetura)",
        "hipotese": "Descongelar só o último bloco reduz os parâmetros treináveis e o "
                    "sobreajuste, com perda pequena de acurácia em relação a E2.",
        "fine_tuning": True,
        "overrides": {"FT_UNFREEZE_LAYERS": ["layer4", "fc"]},
    },
    "E4": {
        "nome": "E2 sem data augmentation",
        "variavel": "Data augmentation (regularização)",
        "hipotese": "Sem augmentation o modelo decora o treino: o gap entre acurácia "
                    "de treino e de validação cresce e a métrica de teste cai.",
        "fine_tuning": True,
        "overrides": {"AUG_ENABLED": False},
    },
}


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #

class CatBreedDataset(Dataset):
    """Como no notebook, mas em módulo — o que permite workers no DataLoader."""

    def __init__(self, dataframe: pd.DataFrame, transform=None, crops: dict | None = None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.crops = crops or {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        bbox = self.crops.get(row["filepath"])
        if bbox is not None:
            img = img.crop(bbox)
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"]) if "label" in row else -1


def completar_config(cfg: dict = None) -> dict:
    """Completa uma configuração parcial com os padrões.

    O CONFIG do notebook não precisa conhecer as chaves que só os experimentos
    usam (`AUG_ENABLED`, por exemplo). Tudo que lê configuração passa por aqui,
    então uma chave ausente vira o padrão em vez de um KeyError no meio do treino.
    """
    return {**CONFIG_PADRAO, **(cfg or {})}


def montar_transforms(cfg: dict):
    cfg = completar_config(cfg)
    size = cfg["IMG_SIZE"]
    eval_tfms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    if not cfg.get("AUG_ENABLED", True):
        return eval_tfms, eval_tfms

    train_tfms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=cfg["AUG_HFLIP_PROB"]),
        transforms.RandomApply([transforms.RandomRotation(cfg["AUG_ROTATION_DEG"])], p=0.5),
        transforms.ColorJitter(
            brightness=cfg["AUG_BRIGHTNESS"], contrast=cfg["AUG_CONTRAST"],
            saturation=cfg["AUG_SATURATION"], hue=cfg["AUG_HUE"],
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=cfg["AUG_BLUR_KERNEL"],
                                     sigma=cfg["AUG_BLUR_SIGMA"])],
            p=cfg["AUG_BLUR_PROB"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(
            p=cfg["AUG_ERASING_PROB"], scale=cfg["AUG_ERASING_SCALE"],
            ratio=cfg["AUG_ERASING_RATIO"], value="random",
        ),
    ])
    return train_tfms, eval_tfms


def montar_loaders(frames: dict, cfg: dict, acel: Acelerador) -> dict:
    cfg = completar_config(cfg)
    train_tfms, eval_tfms = montar_transforms(cfg)
    kwargs = acel.kwargs_loader()
    return {
        "train": DataLoader(CatBreedDataset(frames["train"], train_tfms),
                            batch_size=cfg["BATCH_SIZE"], shuffle=True,
                            drop_last=False, **kwargs),
        "val": DataLoader(CatBreedDataset(frames["val"], eval_tfms),
                          batch_size=cfg["BATCH_SIZE"], shuffle=False, **kwargs),
        "test": DataLoader(CatBreedDataset(frames["test"], eval_tfms),
                           batch_size=cfg["BATCH_SIZE"], shuffle=False, **kwargs),
    }


def carregar_splits(artifacts_dir: Path, cfg: dict = None):
    """Reutiliza os manifests do notebook, para o split ser exatamente o mesmo."""
    cfg = completar_config(cfg)
    artifacts_dir = Path(artifacts_dir)
    manifests = {s: artifacts_dir / f"{s}_manifest.csv" for s in ("train", "val", "test")}

    if all(p.exists() for p in manifests.values()):
        frames = {s: pd.read_csv(p) for s, p in manifests.items()}
        print("Splits carregados de artifacts/*_manifest.csv — mesmo split do notebook.")
    else:
        completo = artifacts_dir / "dataset_manifest.csv"
        if not completo.exists():
            raise FileNotFoundError(
                f"Não encontrei {completo}. Rode as seções 1 a 3 do notebook uma vez: "
                "elas baixam, limpam e registram os dados."
            )
        cats_df = pd.read_csv(completo)
        nomes = sorted(cats_df["breed_name"].unique())
        cats_df["label"] = cats_df["breed_name"].map({b: i for i, b in enumerate(nomes)})
        train_df, temp_df = train_test_split(
            cats_df, test_size=cfg["VAL_FRAC"] + cfg["TEST_FRAC"],
            stratify=cats_df["label"], random_state=cfg["SEED"])
        rel = cfg["TEST_FRAC"] / (cfg["VAL_FRAC"] + cfg["TEST_FRAC"])
        val_df, test_df = train_test_split(
            temp_df, test_size=rel, stratify=temp_df["label"], random_state=cfg["SEED"])
        frames = {"train": train_df, "val": val_df, "test": test_df}
        print("Manifests de split ausentes; split refeito com a semente 42.")

    frames = {k: v.reset_index(drop=True) for k, v in frames.items()}
    breed_names = sorted(pd.concat(frames.values())["breed_name"].unique())
    breed_to_idx = {b: i for i, b in enumerate(breed_names)}
    for f in frames.values():
        f["label"] = f["breed_name"].map(breed_to_idx)

    print(f"Treino: {len(frames['train'])} | Validação: {len(frames['val'])} "
          f"| Teste: {len(frames['test'])} | Classes: {len(breed_names)}")
    return frames, breed_names


# --------------------------------------------------------------------------- #
# Modelo e treino
# --------------------------------------------------------------------------- #

def build_model(num_classes: int, dropout: float, acel: Acelerador) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Dropout(p=dropout),
                             nn.Linear(model.fc.in_features, num_classes))
    model = model.to(acel.device)
    if acel.na_gpu:
        # channels_last é o layout que as convoluções usam nos núcleos tensoriais.
        model = model.to(memory_format=torch.channels_last)
    return model


def contar_parametros(model: nn.Module) -> tuple[int, int]:
    return (sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters()))


class EarlyStopping:
    """Igual ao do notebook: para quando val_loss estagna e restaura o melhor estado."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        improved = self.best_score is None or val_loss < (self.best_score - self.min_delta)
        if improved:
            self.best_score = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def run_epoch(model, loader, criterion, optimizer, acel: Acelerador, train: bool):
    model.train() if train else model.eval()
    total_loss = total_correct = total_samples = 0

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs = acel.para_device(imgs)
            labels = acel.para_device(labels)
            if acel.na_gpu:
                imgs = imgs.contiguous(memory_format=torch.channels_last)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with acel.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            if train:
                if acel.scaler.is_enabled():
                    acel.scaler.scale(loss).backward()
                    acel.scaler.step(optimizer)
                    acel.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += imgs.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_phase(model, loaders, criterion, optimizer, scheduler, acel: Acelerador,
                epochs: int, patience: int, phase_name: str) -> dict:
    stopper = EarlyStopping(patience=patience)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, acel, True)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, optimizer, acel, False)

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = stopper.step(val_loss, model)
        flag = "melhor val_loss" if improved else f"sem melhora ({stopper.counter}/{patience})"
        print(f"[{phase_name}] Epoch {epoch + 1}/{epochs} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
              f"{time.time() - t0:.0f}s | {flag}")

        if stopper.should_stop:
            print(f"Early stopping na época {epoch + 1}.")
            break

    stopper.restore_best(model)
    print(f"[{phase_name}] Melhores pesos restaurados (val_loss={stopper.best_score:.4f}).")
    return history


@torch.no_grad()
def coletar_logits(model, loader, acel: Acelerador):
    model.eval()
    logits, labels = [], []
    for imgs, y in loader:
        imgs = acel.para_device(imgs)
        if acel.na_gpu:
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        with acel.autocast():
            out = model(imgs)
        logits.append(out.float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def avaliar(model, loader, acel: Acelerador, breed_names: list[str]) -> dict:
    logits, labels = coletar_logits(model, loader, acel)
    preds = logits.argmax(axis=1)
    return {
        "acuracia": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "relatorio": classification_report(labels, preds, target_names=breed_names,
                                           output_dict=True, zero_division=0),
        "relatorio_texto": classification_report(labels, preds, target_names=breed_names,
                                                 zero_division=0),
    }


# --------------------------------------------------------------------------- #
# Execução de um experimento
# --------------------------------------------------------------------------- #

def rodar_experimento(exp_id: str, frames: dict, breed_names: list[str],
                      acel: Acelerador, out_dir: Path, *,
                      config: dict = None, cache_head: dict = None) -> dict:
    spec = EXPERIMENTOS[exp_id]
    cfg = {**completar_config(config), **spec["overrides"]}
    cache_head = cache_head if cache_head is not None else {}
    set_seed(cfg["SEED"])

    loaders = montar_loaders(frames, cfg, acel)

    print("\n" + "=" * 72)
    print(f"{exp_id} — {spec['nome']}")
    print(f"Variável alterada: {spec['variavel']}")
    print(f"Hipótese: {spec['hipotese']}")
    print("=" * 72)

    inicio = time.time()
    model = build_model(len(breed_names), cfg["DROPOUT"], acel)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["LABEL_SMOOTHING"])

    # ---- Fase 1: cabeça, backbone congelado --------------------------------
    # Idêntica sempre que augmentation e hiperparâmetros da cabeça coincidem,
    # então é treinada uma vez e reaproveitada pelos demais experimentos.
    chave = (cfg["AUG_ENABLED"], cfg["DROPOUT"], cfg["HEAD_LR"],
             cfg["HEAD_WEIGHT_DECAY"], cfg["HEAD_EPOCHS"], cfg["HEAD_PATIENCE"],
             cfg["BATCH_SIZE"])
    if chave in cache_head:
        estado, history_head = cache_head[chave]
        model.load_state_dict({k: v.clone() for k, v in estado.items()})
        print("[Head] Fase 1 reaproveitada de um experimento anterior.")
    else:
        optimizer = optim.Adam(model.fc.parameters(), lr=cfg["HEAD_LR"],
                               weight_decay=cfg["HEAD_WEIGHT_DECAY"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg["LR_SCHEDULER_FACTOR"],
            patience=cfg["LR_SCHEDULER_PATIENCE"])
        history_head = train_phase(model, loaders, criterion, optimizer, scheduler,
                                   acel, cfg["HEAD_EPOCHS"], cfg["HEAD_PATIENCE"], "Head")
        cache_head[chave] = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                             history_head)

    history = {k: list(v) for k, v in history_head.items()}
    inicio_ft = len(history_head["train_loss"])

    # ---- Fase 2: fine-tuning ------------------------------------------------
    if spec["fine_tuning"]:
        for name, param in model.named_parameters():
            if any(layer in name for layer in cfg["FT_UNFREEZE_LAYERS"]):
                param.requires_grad = True

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=cfg["FT_LR"], weight_decay=cfg["FT_WEIGHT_DECAY"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg["LR_SCHEDULER_FACTOR"],
            patience=cfg["LR_SCHEDULER_PATIENCE"])
        history_ft = train_phase(model, loaders, criterion, optimizer, scheduler,
                                 acel, cfg["FT_EPOCHS"], cfg["FT_PATIENCE"], "Fine-tune")
        for k in history:
            history[k] += history_ft[k]

    treinaveis, _ = contar_parametros(model)
    metricas = avaliar(model, loaders["test"], acel, breed_names)
    duracao = time.time() - inicio

    print(f"\n{exp_id}: acurácia {metricas['acuracia']:.4f} | "
          f"F1-macro {metricas['f1_macro']:.4f} | "
          f"{treinaveis / 1e6:.2f} M parâmetros treináveis | "
          f"{len(history['train_loss'])} épocas | {duracao / 60:.1f} min")
    print(metricas["relatorio_texto"])

    out_dir = Path(out_dir)
    exp_dir = out_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    breed_to_idx = {b: i for i, b in enumerate(breed_names)}
    torch.save({
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "breed_to_idx": breed_to_idx,
        "idx_to_breed": {i: b for b, i in breed_to_idx.items()},
        "config": cfg,
        "experimento": exp_id,
    }, exp_dir / "classificador.pt")
    (exp_dir / "historico.json").write_text(
        json.dumps({"historico": history, "inicio_fine_tuning": inicio_ft},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (exp_dir / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "Exp.": exp_id,
        "Configuração": spec["nome"],
        "Variável alterada": spec["variavel"],
        "Parâmetros treináveis (M)": round(treinaveis / 1e6, 2),
        "Épocas": len(history["train_loss"]),
        "Acurácia (teste)": round(metricas["acuracia"], 4),
        "F1-macro (teste)": round(metricas["f1_macro"], 4),
        "Acurácia treino (última)": round(history["train_acc"][-1], 4),
        "Acurácia val. (última)": round(history["val_acc"][-1], 4),
        "Gap treino−val": round(history["train_acc"][-1] - history["val_acc"][-1], 4),
        "Tempo (min)": round(duracao / 60, 1),
        "Hipótese": spec["hipotese"],
    }


COLUNAS_COMPARATIVO = ["Exp.", "Configuração", "Variável alterada",
                       "Parâmetros treináveis (M)", "Épocas", "Acurácia (teste)",
                       "F1-macro (teste)", "Gap treino−val"]


def _markdown(tabela: pd.DataFrame) -> str:
    """Tabela em Markdown sem depender de `tabulate`.

    `DataFrame.to_markdown` exige o pacote opcional `tabulate`, que não faz
    parte do requirements.txt do projeto. Como aqui a tabela é pequena e de
    formato conhecido, montá-la à mão sai mais barato do que uma dependência.
    """
    cols = [c for c in COLUNAS_COMPARATIVO if c in tabela.columns]
    linhas = ["| " + " | ".join(cols) + " |",
              "|" + "|".join("---" for _ in cols) + "|"]
    for _, linha in tabela.iterrows():
        valores = []
        for c in cols:
            v = linha[c]
            valores.append("—" if pd.isna(v) else
                           (f"{v:g}" if isinstance(v, (int, float)) else str(v)))
        linhas.append("| " + " | ".join(valores) + " |")
    return "\n".join(linhas)


def salvar_comparativo(tabela: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Grava comparativo.csv e comparativo.md. Pode ser chamada sobre um CSV já
    existente, para refazer os relatórios sem repetir o treino."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(out_dir / "comparativo.csv", index=False)

    texto = "# Tabela comparativa dos experimentos\n\n" + _markdown(tabela)
    if "Hipótese" in tabela.columns:
        texto += "\n\n## Hipóteses\n\n" + "\n".join(
            f"- **{r['Exp.']}** — {r['Hipótese']}" for _, r in tabela.iterrows())
    (out_dir / "comparativo.md").write_text(texto + "\n", encoding="utf-8")

    print(f"\nTabela salva em {out_dir}/comparativo.csv e comparativo.md")
    return tabela


def rodar_experimentos(exp_ids, frames, breed_names, acel, out_dir,
                       config: dict = None) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_head: dict = {}
    linhas = [rodar_experimento(e, frames, breed_names, acel, out_dir,
                                config=config, cache_head=cache_head) for e in exp_ids]

    tabela = pd.DataFrame(linhas)
    # O CSV vem primeiro e sozinho: é o resultado do treino e não pode se perder
    # por causa de um relatório. O resto é cosmético e não derruba a execução.
    tabela.to_csv(out_dir / "comparativo.csv", index=False)
    try:
        salvar_comparativo(tabela, out_dir)
    except Exception as erro:                                    # noqa: BLE001
        print(f"\ncomparativo.md não gerado ({erro}). O comparativo.csv está salvo "
              f"em {out_dir}; refaça os relatórios com salvar_comparativo().")
    return tabela


def grafico_comparativo(tabela: pd.DataFrame, destino: Path = None):
    """Barras de acurácia e F1-macro por experimento, com o gap treino−val."""
    import matplotlib.pyplot as plt

    executados = tabela.dropna(subset=["Acurácia (teste)"])
    if executados.empty:
        print("Nenhum experimento com métrica de teste para plotar.")
        return None

    x = np.arange(len(executados))
    largura = 0.38
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                  gridspec_kw={"width_ratios": [1.6, 1]})

    ax.bar(x - largura / 2, executados["Acurácia (teste)"], largura,
           color=AZUL, label="Acurácia")
    ax.bar(x + largura / 2, executados["F1-macro (teste)"], largura,
           color=AZUL_CLARO, label="F1-macro")
    for i, (a, f) in enumerate(zip(executados["Acurácia (teste)"],
                                   executados["F1-macro (teste)"])):
        ax.text(i - largura / 2, a + 0.012, f"{a:.3f}".replace(".", ","),
                ha="center", fontsize=9, color=TEXTO)
        ax.text(i + largura / 2, f + 0.012, f"{f:.3f}".replace(".", ","),
                ha="center", fontsize=9, color=TEXTO)
    ax.set_xticks(x)
    ax.set_xticklabels(executados["Exp."])
    ax.set_ylim(0, 1.05)
    ax.set_title("Desempenho no conjunto de teste", color=TEXTO,
                 fontweight="bold", loc="left")
    ax.legend(frameon=False)

    ax2.bar(x, executados["Gap treino−val"], 0.5, color=LARANJA)
    for i, g in enumerate(executados["Gap treino−val"]):
        ax2.text(i, g + 0.004, f"{g:.3f}".replace(".", ","),
                 ha="center", fontsize=9, color=TEXTO)
    ax2.set_xticks(x)
    ax2.set_xticklabels(executados["Exp."])
    ax2.set_title("Gap treino − validação (sobreajuste)", color=TEXTO,
                  fontweight="bold", loc="left")

    for eixo in (ax, ax2):
        eixo.spines[["top", "right"]].set_visible(False)
        eixo.grid(axis="y", alpha=0.2)
        eixo.set_axisbelow(True)

    fig.tight_layout()
    if destino:
        fig.savefig(destino, dpi=170)
        print("Gráfico salvo em", destino)
    return fig


# --------------------------------------------------------------------------- #
# Análise da regra SRD
# --------------------------------------------------------------------------- #

def carregar_checkpoint(caminho: Path, acel: Acelerador):
    ckpt = torch.load(caminho, map_location=acel.device, weights_only=False)
    idx_to_breed = {int(k): v for k, v in ckpt["idx_to_breed"].items()}
    breed_names = [idx_to_breed[i] for i in sorted(idx_to_breed)]
    cfg = ckpt.get("config", CONFIG_PADRAO)

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=cfg.get("DROPOUT", 0.5)),
                             nn.Linear(model.fc.in_features, len(breed_names)))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(acel.device).eval()
    if acel.na_gpu:
        model = model.to(memory_format=torch.channels_last)
    print(f"Checkpoint: {caminho} ({len(breed_names)} classes, "
          f"experimento {ckpt.get('experimento', '—')})")
    return model, breed_names, cfg


def softmax(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ajustar_temperatura(logits: np.ndarray, labels: np.ndarray) -> float:
    """Otimiza um escalar T minimizando a NLL na validação (Guo et al., 2017)."""
    x = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)   # T = exp(log_t) > 0
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Expected Calibration Error e os pontos do diagrama de confiabilidade."""
    conf = probs.max(axis=1)
    acerto = (probs.argmax(axis=1) == labels).astype(float)
    limites = np.linspace(0.0, 1.0, n_bins + 1)
    erro, bins = 0.0, []
    for lo, hi in zip(limites[:-1], limites[1:]):
        dentro = (conf > lo) & (conf <= hi)
        if dentro.sum() == 0:
            continue
        acc_bin, conf_bin, peso = acerto[dentro].mean(), conf[dentro].mean(), dentro.mean()
        erro += peso * abs(acc_bin - conf_bin)
        bins.append({"centro": float((lo + hi) / 2), "acuracia": float(acc_bin),
                     "confianca": float(conf_bin), "fracao": float(peso)})
    return float(erro), bins


def varrer(probs: np.ndarray, labels: np.ndarray, criterio: str) -> pd.DataFrame:
    ordenado = np.sort(probs, axis=1)
    p1, p2 = ordenado[:, -1], ordenado[:, -2]
    score = p1 if criterio == "confianca" else (p1 - p2)
    acerto = probs.argmax(axis=1) == labels

    linhas = []
    for limiar in np.round(np.arange(0.0, 0.96, 0.05), 2):
        aceito = score >= limiar
        linhas.append({
            "criterio": criterio,
            "limiar": float(limiar),
            "cobertura": round(float(aceito.mean()), 4),
            "acuracia_aceitos": round(float(acerto[aceito].mean()), 4) if aceito.any() else None,
            "n_aceitos": int(aceito.sum()),
            "erros_aceitos": int((aceito & ~acerto).sum()),
        })
    return pd.DataFrame(linhas)


def escolher_limiar(tabela: pd.DataFrame, alvo: float) -> dict | None:
    """Menor limiar que atinge a acurácia alvo — ou seja, o de maior cobertura."""
    cand = tabela.dropna(subset=["acuracia_aceitos"])
    cand = cand[cand["acuracia_aceitos"] >= alvo]
    return None if cand.empty else cand.sort_values("limiar").iloc[0].to_dict()


def detectar_crops(filepaths: list[str], projeto: Path, conf: float = 0.25) -> dict:
    """Recorta o gato com YOLOv8, como no pipeline do notebook e do app."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics indisponível: usando a imagem inteira, sem recorte.")
        return {}

    pesos = Path(projeto) / "yolov8n.pt"
    yolo = YOLO(str(pesos) if pesos.exists() else "yolov8n.pt")
    crops, sem_gato = {}, 0
    for path in filepaths:
        resultado = yolo(path, verbose=False)[0]
        melhor, melhor_conf = None, 0.0
        for box in resultado.boxes:
            c = float(box.conf)
            if int(box.cls) == 15 and c >= conf and c > melhor_conf:   # 15 = cat no COCO
                melhor_conf, melhor = c, tuple(int(v) for v in box.xyxy[0].tolist())
        if melhor:
            crops[path] = melhor
        else:
            sem_gato += 1
    print(f"YOLOv8: {len(crops)} recortes; {sem_gato} imagens sem gato detectado.")
    return crops


def logits_de_pasta(model, pasta: Path, acel: Acelerador, cfg: dict,
                    projeto: Path, usar_yolo: bool = True):
    arquivos = sorted(str(p) for p in Path(pasta).rglob("*")
                      if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)
    if not arquivos:
        return None, []
    crops = detectar_crops(arquivos, projeto) if usar_yolo else {}
    df = pd.DataFrame({"filepath": arquivos, "label": -1})
    _, eval_tfms = montar_transforms({**cfg, "AUG_ENABLED": False})
    loader = DataLoader(CatBreedDataset(df, eval_tfms, crops),
                        batch_size=cfg.get("BATCH_SIZE", 32), shuffle=False,
                        **acel.kwargs_loader())
    logits, _ = coletar_logits(model, loader, acel)
    return logits, arquivos


def analisar_srd(model, breed_names, cfg, frames, acel: Acelerador, *,
                 out_dir: Path, projeto: Path, srd_dir: Path = None,
                 alvo_acuracia: float = 0.90, limiar_atual: float = 0.40,
                 usar_yolo: bool = True) -> dict:
    """Calibra, varre limiares e mede o conjunto de controle SRD."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = montar_loaders(frames, {**cfg, "AUG_ENABLED": False}, acel)

    # 1. Calibração por temperatura na validação
    logits_val, y_val = coletar_logits(model, loaders["val"], acel)
    t = ajustar_temperatura(logits_val, y_val)
    ece_antes, bins_antes = ece(softmax(logits_val, 1.0), y_val)
    ece_depois, bins_depois = ece(softmax(logits_val, t), y_val)
    (out_dir / "calibracao.json").write_text(json.dumps({
        "temperatura": t, "ece_antes": ece_antes, "ece_depois": ece_depois,
        "bins_antes": bins_antes, "bins_depois": bins_depois,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nCalibração por temperatura: T = {t:.3f}")
    print(f"ECE antes: {ece_antes:.4f}  |  depois: {ece_depois:.4f}")
    if t > 1.02:
        print(f"T = {t:.2f} > 1 confirma superconfiança: o softmax dizia mais do que acertava.")
    elif t < 0.98:
        print(f"T = {t:.2f} < 1: o modelo estava subconfiante na validação.")
    else:
        print("T ≈ 1: o softmax já estava razoavelmente calibrado na validação.")

    # 2. Varredura de limiar no teste
    logits_test, y_test = coletar_logits(model, loaders["test"], acel)
    probs_test = softmax(logits_test, t)
    varreduras = pd.concat([varrer(probs_test, y_test, "confianca"),
                            varrer(probs_test, y_test, "margem")], ignore_index=True)
    varreduras.to_csv(out_dir / "varredura.csv", index=False)

    conf_tab = varreduras[varreduras["criterio"] == "confianca"]
    atual = conf_tab.iloc[(conf_tab["limiar"] - limiar_atual).abs().argmin()].to_dict()
    escolhido = escolher_limiar(conf_tab, alvo_acuracia)
    escolhido_margem = escolher_limiar(varreduras[varreduras["criterio"] == "margem"],
                                       alvo_acuracia)

    if atual["acuracia_aceitos"] is None or pd.isna(atual["acuracia_aceitos"]):
        print(f"\nLimiar atual ({limiar_atual:.2f}): nenhuma foto passa do limiar — "
              "cobertura 0%. O limiar está alto demais para este modelo.")
    else:
        print(f"\nLimiar atual ({limiar_atual:.2f}): cobertura {atual['cobertura']:.1%}, "
              f"acurácia entre aceitos {atual['acuracia_aceitos']:.1%}")
    if escolhido:
        print(f"Para {alvo_acuracia:.0%} de acurácia entre aceitos: limiar "
              f"{escolhido['limiar']:.2f}, cobertura {escolhido['cobertura']:.1%}")
    else:
        print(f"Nenhum limiar alcança {alvo_acuracia:.0%} de acurácia entre os aceitos "
              "— resultado negativo, e é isso que deve ser reportado.")

    # 3. Conjunto de controle SRD: acertar é rejeitar
    resumo = {
        "temperatura": round(t, 3),
        "ece_antes": round(ece_antes, 4),
        "ece_depois": round(ece_depois, 4),
        "limiar_atual": {"limiar": limiar_atual, "cobertura": atual["cobertura"],
                         "acuracia_aceitos": atual["acuracia_aceitos"],
                         "erros_aceitos": atual["erros_aceitos"]},
        "limiar_recomendado_confianca": escolhido,
        "limiar_recomendado_margem": escolhido_margem,
        "alvo_acuracia": alvo_acuracia,
    }
    controle = None

    if srd_dir and Path(srd_dir).is_dir():
        logits_srd, arquivos = logits_de_pasta(model, srd_dir, acel, cfg, projeto, usar_yolo)
        if logits_srd is not None:
            probs_srd = softmax(logits_srd, t)
            ordenado = np.sort(probs_srd, axis=1)
            p1, margem = ordenado[:, -1], ordenado[:, -1] - ordenado[:, -2]
            controle = pd.DataFrame([{
                "limiar": float(l),
                "taxa_rejeicao_confianca": round(float((p1 < l).mean()), 4),
                "taxa_rejeicao_margem": round(float((margem < l).mean()), 4),
            } for l in np.round(np.arange(0.0, 0.96, 0.05), 2)])
            controle.to_csv(out_dir / "controle_srd.csv", index=False)

            linha = controle.iloc[(controle["limiar"] - limiar_atual).abs().argmin()]
            print(f"\nConjunto de controle SRD: {len(arquivos)} fotos")
            print(f"No limiar {limiar_atual:.2f}, "
                  f"{linha['taxa_rejeicao_confianca']:.1%} são corretamente marcadas "
                  "como SRD — as demais recebem uma raça que o gato não tem.")
            top = pd.Series([breed_names[i] for i in probs_srd.argmax(axis=1)]).value_counts()
            print("Raças mais atribuídas às fotos SRD:",
                  ", ".join(f"{b} ({n})" for b, n in top.head(3).items()))

            resumo["controle_srd"] = {
                "n_fotos": len(arquivos),
                "taxa_rejeicao_limiar_atual": float(linha["taxa_rejeicao_confianca"]),
            }
            if escolhido:
                rec = controle.iloc[(controle["limiar"] - escolhido["limiar"]).abs().argmin()]
                resumo["controle_srd"]["taxa_rejeicao_limiar_recomendado"] = \
                    float(rec["taxa_rejeicao_confianca"])
    else:
        print(f"\n{srd_dir} não encontrado; conjunto de controle SRD ignorado.")

    (out_dir / "resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResumo salvo em {out_dir / 'resumo.json'}")

    return {"resumo": resumo, "varreduras": varreduras, "controle": controle,
            "bins_antes": bins_antes, "bins_depois": bins_depois, "temperatura": t}


def graficos_srd(analise: dict, out_dir: Path = None):
    """Diagrama de confiabilidade e curva cobertura × acurácia."""
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax.plot([0, 1], [0, 1], color=CINZA, linestyle=":", label="calibração perfeita")
    for bins, cor, rotulo in ((analise["bins_antes"], AZUL, "antes"),
                              (analise["bins_depois"], LARANJA,
                               f"depois (T = {analise['temperatura']:.2f})")):
        ax.plot([b["confianca"] for b in bins], [b["acuracia"] for b in bins],
                marker="o", color=cor, label=rotulo)
    ax.set_xlabel("confiança média do softmax")
    ax.set_ylabel("acurácia observada")
    ax.set_title("Diagrama de confiabilidade", color=TEXTO, fontweight="bold", loc="left")
    ax.legend(frameon=False)

    for criterio, cor, rotulo in (("confianca", AZUL, "limiar de confiança (p₁)"),
                                  ("margem", LARANJA, "limiar de margem (p₁ − p₂)")):
        sub = analise["varreduras"]
        sub = sub[sub["criterio"] == criterio].dropna(subset=["acuracia_aceitos"])
        ax2.plot(sub["cobertura"], sub["acuracia_aceitos"], marker="o", color=cor, label=rotulo)
    ax2.set_xlabel("cobertura — fração das fotos que recebem uma raça")
    ax2.set_ylabel("acurácia entre as fotos aceitas")
    ax2.set_title("Cobertura × acurácia no teste", color=TEXTO, fontweight="bold", loc="left")
    ax2.legend(frameon=False)

    for eixo in (ax, ax2):
        eixo.spines[["top", "right"]].set_visible(False)
        eixo.grid(alpha=0.2)
        eixo.set_axisbelow(True)

    fig.tight_layout()
    if out_dir:
        destino = Path(out_dir) / "srd_calibracao_cobertura.png"
        fig.savefig(destino, dpi=170)
        print("Gráficos salvos em", destino)
    return fig
