#!/usr/bin/env python3
"""Plano de experimentos do trabalho final (Seção 9 do enunciado), fora do notebook.

A lógica fica em `cat_experimentos.py`, na raiz do projeto — o mesmo módulo que
a Seção 12 do notebook usa. Aqui só há a linha de comando.

    E1  Baseline      backbone congelado, apenas a cabeça treinada
    E2  Arquitetura   E1 + fine-tuning de layer3 + layer4 + fc
    E3  Arquitetura   E1 + fine-tuning de layer4 + fc
    E4  Regularização E2 sem data augmentation (extra, para a discussão
                      de sobreajuste)

Todos compartilham dados, split, semente, otimizador, scheduler e critério de
parada. A fase 1 (cabeça) é idêntica em E1/E2/E3 e por isso é treinada uma única
vez e reaproveitada.

**Rode com o Python do projeto**, senão cai em CPU e leva horas:

    source .venv/bin/activate
    python scripts/experimentos.py                    # E1, E2 e E3
    python scripts/experimentos.py --exp E1 E2 E3 E4
    python scripts/experimentos.py --batch-size 96    # GPU com mais memória
    python scripts/experimentos.py --sem-amp          # float32, para reproduzir

Saídas em `artifacts/experimentos/`: um diretório por experimento com
`historico.json`, `metricas.json` e `classificador.pt`, mais `comparativo.csv`
e `comparativo.md` com a tabela exigida pelo enunciado.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import cat_experimentos as ce  # noqa: E402  (precisa do sys.path acima)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", nargs="+", default=["E1", "E2", "E3"],
                        choices=list(ce.EXPERIMENTOS), help="experimentos a rodar")
    parser.add_argument("--device", default=None, help='"cuda" ou "cpu" (padrão: automático)')
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="processos do DataLoader (padrão: até 8 em GPU)")
    parser.add_argument("--sem-amp", action="store_true",
                        help="desliga a precisão mista; mais lento, numericamente estável")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_DIR / "artifacts" / "experimentos")
    args = parser.parse_args()

    acel = ce.preparar_gpu(args.device, amp=not args.sem_amp, num_workers=args.num_workers)

    config = dict(ce.CONFIG_PADRAO)
    config["BATCH_SIZE"] = args.batch_size or (64 if acel.na_gpu else 32)
    print(f"Batch size: {config['BATCH_SIZE']}\n")

    frames, breed_names = ce.carregar_splits(PROJECT_DIR / "artifacts", config)
    tabela = ce.rodar_experimentos(args.exp, frames, breed_names, acel, args.out,
                                   config=config)

    colunas = ["Exp.", "Configuração", "Variável alterada", "Parâmetros treináveis (M)",
               "Épocas", "Acurácia (teste)", "F1-macro (teste)", "Gap treino−val"]
    print("\n" + "=" * 72)
    print("TABELA COMPARATIVA")
    print("=" * 72)
    print(tabela[colunas].to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        ce.grafico_comparativo(tabela, args.out / "comparativo.png")
    except Exception as erro:                                    # noqa: BLE001
        print(f"Gráfico comparativo não gerado: {erro}")


if __name__ == "__main__":
    main()
