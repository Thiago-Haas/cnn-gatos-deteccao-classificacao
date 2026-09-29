#!/usr/bin/env python3
"""Análise da regra SRD: calibração, limiar de rejeição e cobertura × acurácia.

A lógica fica em `cat_experimentos.py`, na raiz do projeto — o mesmo módulo que
a Seção 13 do notebook usa. Aqui só há a linha de comando.

A regra "Sem Raça Definida" hoje é um limiar fixo de 40% sobre o softmax. Este
script mede se esse limiar se sustenta e responde às perguntas 5, 6 e 7 da
Seção 14.1 do enunciado.

1. Calibração por temperatura na **validação**: um escalar T ajustado por NLL.
   Softmax de rede profunda costuma ser superconfiante; T > 1 achata a
   distribuição e aproxima a confiança da taxa de acerto. ECE antes e depois.
2. Varredura de limiar no **teste**, por confiança máxima (p₁) e por margem
   entre a primeira e a segunda classe (p₁ − p₂): cobertura e acurácia entre os
   aceitos, em cada limiar.
3. Conjunto de controle SRD — as fotos de `real_photos/`. São gatos sem raça
   definida, então toda predição aceita ali é um falso aceite: a métrica é a
   taxa de rejeição, e quanto maior, melhor.

**Rode com o Python do projeto**, senão cai em CPU:

    source .venv/bin/activate
    python scripts/analise_srd.py                     # usa o modelo de E2
    python scripts/analise_srd.py --checkpoint artifacts/cat_breed_classifier.pt
    python scripts/analise_srd.py --sem-yolo          # sem recorte, imagem inteira

Saídas em `artifacts/experimentos/srd/`: `varredura.csv`, `controle_srd.csv`,
`resumo.json` e `srd_calibracao_cobertura.png`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import cat_experimentos as ce  # noqa: E402  (precisa do sys.path acima)

ARTIFACTS_DIR = PROJECT_DIR / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        default=ARTIFACTS_DIR / "experimentos" / "E2" / "classificador.pt")
    parser.add_argument("--srd-dir", type=Path, default=PROJECT_DIR / "real_photos",
                        help="pasta com fotos de gatos SRD (conjunto de controle)")
    parser.add_argument("--alvo-acuracia", type=float, default=0.90,
                        help="acurácia desejada entre os aceitos (padrão: 0,90)")
    parser.add_argument("--limiar-atual", type=float, default=0.40,
                        help="limiar em uso hoje na webcam e no app")
    parser.add_argument("--sem-yolo", action="store_true",
                        help="não recortar com YOLOv8; classificar a imagem inteira")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sem-amp", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=ARTIFACTS_DIR / "experimentos" / "srd")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        alternativa = ARTIFACTS_DIR / "cat_breed_classifier.pt"
        if not alternativa.exists():
            raise SystemExit(
                f"Checkpoint não encontrado: {args.checkpoint}\n"
                "Rode scripts/experimentos.py antes, ou aponte --checkpoint para um .pt."
            )
        print(f"{args.checkpoint} não existe; usando {alternativa}.")
        args.checkpoint = alternativa

    acel = ce.preparar_gpu(args.device, amp=not args.sem_amp)
    model, breed_names, cfg = ce.carregar_checkpoint(args.checkpoint, acel)

    cfg = dict(cfg)
    cfg["BATCH_SIZE"] = args.batch_size or (64 if acel.na_gpu else 32)

    frames, _ = ce.carregar_splits(ARTIFACTS_DIR, cfg)

    analise = ce.analisar_srd(
        model, breed_names, cfg, frames, acel,
        out_dir=args.out, projeto=PROJECT_DIR, srd_dir=args.srd_dir,
        alvo_acuracia=args.alvo_acuracia, limiar_atual=args.limiar_atual,
        usar_yolo=not args.sem_yolo,
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        ce.graficos_srd(analise, args.out)
    except Exception as erro:                                    # noqa: BLE001
        print(f"Gráficos não gerados: {erro}")


if __name__ == "__main__":
    main()
