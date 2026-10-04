"""Classify controlled comparisons without altering original raw measurements."""
import datetime,json
def live_process_sample(sample):
    value=sample.get('process',{}).get('stat') or ''
    fields=value[value.rfind(')')+2:].split() if value else []
    # Linux removes zombie tasks from their Android cgroups. That is exit
    # cleanup after computation, not a scheduling change during measurement.
    return not fields or fields[0] not in ['Z','X','x']
def benchmark_priority_sample(sample):
    if not live_process_sample(sample):return False
    value=sample.get('process',{}).get('stat') or ''
    # Popen may be sampled before nice/taskset exec the benchmark. Their
    # inherited priority is not the inference worker's execution priority.
    # Preserve the raw launch sample, but inspect priority after that launcher.
    start=value.find('(');end=value.rfind(')')
    comm=value[start+1:end] if start>=0 and end>start else None
    return comm not in ['nice','taskset']
def issues(result, directory):
    found = []
    stderr = directory / 'stderr.txt'
    text = stderr.read_text(errors='replace') if stderr.exists() else ''
    if 'failed to set affinity' in text:
        found.append('Requested CPU affinity was rejected by Android')
    c=result['config']
    intervention_file=directory.parent.parent/'npu-investigation/thermal-probe-intervention.json'
    if intervention_file.exists() and not c.get('validation_timing_not_comparable') and result.get('before',{}).get('UTC') and result.get('after',{}).get('UTC'):
        intervention=json.loads(intervention_file.read_text())
        start=datetime.datetime.fromisoformat(result['before']['UTC'])
        end=datetime.datetime.fromisoformat(result['after']['UTC'])
        if start<=datetime.datetime.fromisoformat(intervention['end_UTC']) and end>=datetime.datetime.fromisoformat(intervention['start_UTC']):
            found.append('Auxiliary thermal capability compilation/probe overlapped measurement; exclude and repeat')
    if c.get('execution_policy')=='nice10-ui-headroom01' and not c.get('validation_timing_not_comparable'):
        observed={w['nice'] for s in result.get('samples',[]) if benchmark_priority_sample(s) for w in s.get('process',{}).get('workers',{}).values() if 'nice' in w}
        if observed and observed!={10}:
            found.append('Actual benchmark worker priority differs from the nice10 execution policy: '+str(sorted(observed)))
    if c.get('mode') not in ['memory','logits','ops','ops-perf'] and c.get('threads')==1 and c.get('mask') and not c.get('process_mask'):
        found.append('OpenMP single-thread path does not apply the requested threadpool affinity; needs process-level affinity')
    masks = {tuple(s['harness_allowed_CPUs']) for s in result.get('samples', []) if 'harness_allowed_CPUs' in s}
    groups = {s['harness_cgroup'] for s in result.get('samples', []) if s.get('harness_cgroup')}
    if len(masks) > 1 or len(groups) > 1:
        found.append('CPU availability or Android cgroup changed during measurement')
    child_groups={s['process']['cgroup'] for s in result.get('samples',[]) if live_process_sample(s) and s.get('process',{}).get('cgroup')}
    if len(child_groups)>1:found.append('Benchmark process Android cgroup changed during measurement')
    if c.get('process_mask') and not c.get('mask'):
        wanted={i for i in range(8) if int(c['process_mask'],16)&(1<<i)}
        for s in result.get('samples',[]):
            if not live_process_sample(s):continue
            actual=set()
            for part in (s.get('process',{}).get('allowed_CPUs') or '').split(','):
                if not part:continue
                ends=part.split('-');actual.update(range(int(ends[0]),int(ends[-1])+1))
            if actual and not wanted<=actual:
                found.append('Benchmark process lost required cores despite process-level affinity')
                break
    if result.get('exit_code') != 0 or result.get('parse_error') or result.get('abort_reason'):
        found.append('Run failed, was aborted, or did not parse')
    condition_file=directory.parent.parent/'conditions.json'
    if condition_file.exists() and result.get('before',{}).get('UTC') and result.get('after',{}).get('UTC'):
        start=datetime.datetime.fromisoformat(result['before']['UTC'])
        end=datetime.datetime.fromisoformat(result['after']['UTC'])
        for event in json.loads(condition_file.read_text()).get('history',[]):
            transition=any(word in event.get('event','').lower() for word in ['unplugging charger','charging epoch','charger reconnected','battery api observed power transition','hold demanding tests','battery recovered'])
            if transition and event.get('UTC'):
                detected=datetime.datetime.fromisoformat(event['UTC'])
                earliest=datetime.datetime.fromisoformat(event.get('uncertain_since_UTC',event['UTC']))
                if start<=detected and end>=earliest:
                    found.append('Power transition observed or possible within measurement interval')
    return found
