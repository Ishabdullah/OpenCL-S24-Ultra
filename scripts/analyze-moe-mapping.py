#!/usr/bin/env python3
"""Analyze captured routing without changing model behavior.

Usage: PYTHONPATH=.work-npu/llama.cpp/gguf-py python scripts/analyze-moe-mapping.py TRACE --train-trials 6
"""
import argparse
import json
import re
from pathlib import Path


def analyze(records, train_trials, tensor_sizes):
    config = records[0]
    assert config['type'] == 'config'
    trials = records[1:]
    assert len(trials) == config['trials'], 'incomplete run'
    assert 0 < train_trials < len(trials), 'need training and held-out trials'
    layers = set(trials[0]['prefill'])
    expert_count = config['experts']
    used = config['experts_used']
    union = {layer: set() for layer in layers}
    training = None
    curve = []
    heldout = {'prefill': [0, 0], 'decode': [0, 0]}
    heldout_new = {layer: set() for layer in layers}
    continuations = set()
    for index, trial in enumerate(trials, 1):
        assert trial['trial'] == index
        assert set(trial['prefill']) == layers, 'missing prefill layers'
        # EOG can terminate at the first sample; an empty decode is then valid.
        assert set(trial['decode']) == (layers if trial['generated_tokens'] > 1 else set())
        before = sum(map(len, union.values()))
        for phase in ('prefill', 'decode'):
            for layer, counts in trial[phase].items():
                assert all(0 <= int(expert) < expert_count and count > 0 for expert, count in counts.items())
                expected = config['prompt_tokens'] if phase == 'prefill' else trial['generated_tokens'] - 1
                assert sum(counts.values()) == expected * used, 'router capture count mismatch'
                ids = set(map(int, counts))
                union[layer].update(ids)
                if index > train_trials:
                    hits = sum(count for expert, count in counts.items() if int(expert) in training[layer])
                    heldout[phase][0] += hits
                    heldout[phase][1] += sum(counts.values())
                    heldout_new[layer].update(ids - training[layer])
        total = sum(map(len, union.values()))
        assert total - before == trial['new_experts_total']
        curve.append({'trial': index, 'mean_union': total / len(layers),
                      'new_total': total - before,
                      'min_union': min(map(len, union.values())),
                      'max_union': max(map(len, union.values()))})
        continuations.add(trial['text'])
        if index == train_trials:
            training = {layer: ids.copy() for layer, ids in union.items()}
    per_expert = {layer: 0 for layer in layers}
    weight_tensors = {layer: 0 for layer in layers}
    full_routed_bytes = 0
    fixed_bytes = 0
    for name, shape, size in tensor_sizes:
        match = re.fullmatch(r'blk\.(\d+)\.ffn_(?:gate|up|down)_exps\.weight', name)
        if match and match.group(1) in layers:
            layer = match.group(1)
            weight_tensors[layer] += 1
            assert len(shape) == 3 and shape[-1] == expert_count
            assert size % expert_count == 0
            per_expert[layer] += size // expert_count
            full_routed_bytes += size
        else:
            fixed_bytes += size
    assert all(count == 3 for count in weight_tensors.values()), 'missing expert weights'
    training_bytes = sum(len(ids) * per_expert[layer] for layer, ids in training.items())
    final_bytes = sum(len(ids) * per_expert[layer] for layer, ids in union.items())
    return dict(config=config, captured_layers=len(layers), unique_continuations=len(continuations),
                training_trials=train_trials, heldout_trials=len(trials)-train_trials, curve=curve,
                heldout_coverage={phase: {'hits': hits, 'selections': total,
                                        'fraction': hits / total if total else None}
                                  for phase, (hits, total) in heldout.items()},
                heldout_new_distinct_total=sum(map(len, heldout_new.values())),
                weights=dict(full_routed_bytes=full_routed_bytes, fixed_tensor_bytes=fixed_bytes,
                             training_expert_bytes=training_bytes, final_expert_bytes=final_bytes,
                             training_plus_fixed_bytes=training_bytes+fixed_bytes,
                             final_plus_fixed_bytes=final_bytes+fixed_bytes,
                             # Uniform slots are a different policy from the variable per-layer union.
                             uniform_32_experts_bytes=sum(per_expert.values())*32,
                             uniform_64_experts_bytes=sum(per_expert.values())*64),
                storage_read_bytes=sum(t['storage_read_bytes'] for t in trials),
                elapsed_trial_s=sum(t['elapsed_s'] for t in trials))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--train-trials', type=int, default=6)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tensor-sizes', type=Path, help='saved GGUF tensor metadata; avoids opening model again')
    args = parser.parse_args()
    records = [json.loads(line) for line in args.trace.read_text().splitlines()]
    if args.tensor_sizes:
        saved = json.loads(args.tensor_sizes.read_text())
        if saved['model'] != records[0]['model']:
            raise ValueError('tensor metadata belongs to another model')
        sizes = [(t['name'], t['shape'], t['bytes']) for t in saved['tensors']]
    else:
        from gguf import GGUFReader
        reader = GGUFReader(records[0]['model'])
        sizes = [(t.name, list(map(int, t.shape)), int(t.n_bytes)) for t in reader.tensors]
    result = analyze(records, args.train_trials, sizes)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
