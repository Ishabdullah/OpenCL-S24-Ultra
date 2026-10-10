#!/usr/bin/env python3
"""Summarize cache/thread throughput screening with exact-logit verification."""
import argparse
import json
from pathlib import Path
import re
import statistics


def summarize(records, run, stats, thermal, cooldowns=None):
    assert run['returncode'] == 0 and run['stop_reason'] is None, 'run failed or stopped'
    config = next(row for row in records if row['type'] == 'config')
    reference = next(row for row in records if row['type'] == 'reference')
    passes = [row for row in records if row['type'] == 'pass']
    complete = [row for row in records if row['type'] == 'complete']
    plan = config['plan']
    assert len(passes) == len(plan) and len(complete) == 1, 'incomplete sweep'
    assert len(plan) >= 2
    steps = reference['steps']
    assert steps > 1 and len(reference['tokens']) == steps
    assert complete[0] == dict(type='complete', passes=len(plan), reference_positions=steps,
                              verified_positions=steps*(len(plan)-1))
    assert [row['pass'] for row in passes] == list(range(1,len(plan)+1))
    expected_cache_passes = [row['pass'] for row in passes if row['slots']]
    assert [row['pass'] for row in stats] == expected_cache_passes, 'missing cache statistics'
    for row, expected in zip(passes, plan):
        assert all(row[key] == expected[key] for key in ('slots','threads','cache_mib')), 'plan mismatch'
        assert row['evaluated_positions'] == steps and row['decode_evaluations'] == steps-1
        assert row['bit_identical_positions'] == (0 if row['pass'] == 1 else steps), 'logit verification failed'
        assert row['prefill_s'] > 0 and row['decode_s'] > 0
        assert abs(row['evaluation_s']-row['prefill_s']-row['decode_s']) < 0.00001
        assert abs(row['decode_tok_s']-(steps-1)/row['decode_s']) < 0.000001
        assert row['storage_bytes'] == row['prefill_storage_bytes']+row['decode_storage_bytes']
        cache = next((item for item in stats if item['pass'] == row['pass']), None)
        if cache:
            assert cache['slots'] == row['slots']
            assert cache['allocated_bytes'] <= row['cache_mib']*2**20
            row = row.copy()
            row['cache_stats'] = cache
        samples = [item for item in thermal if item.get('pass_number') == row['pass']
                   and item.get('stage') == 'evaluation']
        if samples:
            row = row.copy()
            row['temperature'] = {key: dict(first=values[0], last=values[-1], peak=max(values))
                for key in ('battery_c','cpu_max_c','htp_max_c')
                if (values := [item[key] for item in samples if item.get(key) is not None])}
        passes[row['pass']-1] = row
    groups = []
    for slots, threads, mib in sorted({(row['slots'],row['threads'],row['cache_mib']) for row in passes}):
        group = [row for row in passes if (row['slots'],row['threads'],row['cache_mib']) == (slots,threads,mib)]
        groups.append(dict(slots=slots, threads=threads, cache_mib=mib, repetitions=len(group),
            pooled_decode_tok_s=sum(row['decode_evaluations'] for row in group)/sum(row['decode_s'] for row in group),
            mean_prefill_s=statistics.mean(row['prefill_s'] for row in group),
            mean_evaluation_s=statistics.mean(row['evaluation_s'] for row in group),
            mean_decode_storage_bytes=statistics.mean(row['decode_storage_bytes'] for row in group)))
    groups.sort(key=lambda row:row['pooled_decode_tok_s'], reverse=True)
    balanced_pairs=[]
    if len(plan)>=4 and plan[0]==plan[-1] and plan[1]==plan[-2]:
        for control,candidate in ((passes[0],passes[1]),(passes[-1],passes[-2])):
            balanced_pairs.append(dict(control_pass=control['pass'],candidate_pass=candidate['pass'],
                decode_speedup=control['decode_s']/candidate['decode_s'],
                evaluation_speedup=control['evaluation_s']/candidate['evaluation_s']))
    fastest = max(passes,key=lambda row:row['decode_tok_s'])
    return dict(config=config, reference=reference, passes=passes, ranked_configurations=groups,
                fastest_pass=fastest, cache_stats=stats, run=run, cooldowns=cooldowns or [], balanced_pairs=balanced_pairs,
                limitations=['Single prompt/trajectory/device/session; most settings have one trial.',
                             (f'Recorded 3–5 minute cooling breaks begin after pass {cooldowns[0]["after_pass"]}; inspect chronological comparisons accordingly.'
                              if cooldowns else 'Temperatures observed; fixed configured rests, no thermal admission.'),
                             'Page cache, scheduling and memory pressure are not reset.',
                             'Screening rates are observations; confirm the selected setting in both orders.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    folder = args.directory
    records = [json.loads(line) for line in (folder/'benchmark.jsonl').read_text().splitlines()]
    run = json.loads((folder/'run.json').read_text())
    thermal = [json.loads(line) for line in (folder/'thermal.jsonl').read_text().splitlines()]
    if (folder/'benchmark.log').exists():
        stats=[]
        completed=None
        for line in (folder/'benchmark.log').read_text().splitlines():
            match=re.match(r'CACHE_BENCH complete pass=(\d+)',line)
            if match: completed=int(match.group(1))
            if line.startswith('MOE_CACHE slots='):
                assert completed is not None
                row={key:int(value) for key,value in re.findall(r'(\w+)=(\d+)',line)}
                row['pass']=completed
                stats.append(row)
        (folder/'cache-stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    else:
        stats=json.loads((folder/'cache-stats.json').read_text())
    cooldowns=[json.loads(line) for line in (folder/'cooldown.jsonl').read_text().splitlines()] if (folder/'cooldown.jsonl').exists() else []
    result=summarize(records,run,stats,thermal,cooldowns)
    (folder/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'fastest_pass':{key:result['fastest_pass'][key] for key in ('pass','slots','threads','decode_tok_s')},
                      'ranked_configurations':result['ranked_configurations']}))

if __name__=='__main__':
    main()
