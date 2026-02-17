#!/usr/bin/env python3

import sys, os, json, torch

sys.path.insert(0, 'src')

from dataset import create_datasets, VOCAB_SIZE, VOCAB
from models import create_model
from train import run_stability_sweep, run_generalization_test, evaluate, check_success_criteria
from losses import create_loss_fn
from torch.utils.data import DataLoader


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, val_ds, test_ds = create_datasets(max_seq_len=64)

    os.makedirs('results', exist_ok=True)

    # Stability sweep
    print('=== STABILITY SWEEP ===')
    stability = {}
    for g in ['B', 'D']:
        stability[g] = run_stability_sweep(g, train_ds, val_ds, device, 'results', epochs=10)

    with open('results/stability_results.json', 'w') as f:
        json.dump(stability, f, indent=2, default=str)

    # Generalization tests
    print('\n=== GENERALIZATION TESTS ===')
    gen = {}
    for g in ['A', 'B', 'C', 'D']:
        model = create_model(g, VOCAB_SIZE, device=device)
        model.load_state_dict(torch.load(
            f'results/group_{g}_model.pt',
            map_location=device,
            weights_only=True
        ))

        print(f'Group {g}:')
        gen[g] = run_generalization_test(model, g, device)

    with open('results/generalization_results.json', 'w') as f:
        json.dump(gen, f, indent=2, default=str)

    # Check success criteria
    all_results = {}
    for g in ['A', 'B', 'C', 'D']:
        with open(f'results/group_{g}_results.json') as f:
            all_results[g] = json.load(f)

    all_results['generalization'] = gen
    check_success_criteria(all_results)

    print('\nDone!')


if __name__ == "__main__":
    main()
