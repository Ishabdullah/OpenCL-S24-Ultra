"""Balanced independent confirmations; one child workload at a time."""
import copy, datetime, json, pathlib, statistics, sys
import measure
from validation import issues
B=pathlib.Path(__file__).resolve().parent
R=B/'npu-investigation'
epoch=measure.conditions()['power_epoch']
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
plan_file=pathlib.Path(sys.argv[1])
plan=json.loads(plan_file.read_text())
state_file=plan_file.with_suffix('.state.json')
records=[]
def publish(stage,**extra):
 state={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':stage,'power_epoch':epoch,'records':records,**extra}
 state_file.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps({'stage':stage,**extra}),flush=True)
try:
 for c in plan:
  if measure.conditions()['power_epoch']!=epoch:raise RuntimeError('Power epoch changed; preserve and plan a new matched set')
  publish('confirming',current=c['id'])
  d=measure.run(c);bad=issues(d,B/'runs'/c['id']);m=d.get('latency_metrics') or {}
  if d.get('status')=='guarded_skip':bad.append('Memory admission skip; no performance measurement')
  elif not(m.get('all_logits_finite') and m.get('same_model_and_cache') and m.get('positions_preserved')):bad.append('Phase state/finite checks failed')
  if d.get('exit_code')==0 and 'PHASE cleanup complete' not in (B/'runs'/c['id']/'stderr.txt').read_text():bad.append('Owned context/model cleanup marker missing')
  records.append({'id':c['id'],'model':c['model'],'round':c['confirmation_round'],'strategy':c['strategy'],'valid':not bad,'issues':bad,'metrics':m})
  publish('recorded',current=c['id'],metrics={k:m.get(k) for k in ['prefill_tok_s','decode_tok_s','handoff_ms','total_response_s']})
  if bad:raise RuntimeError('Excluded run: '+'; '.join(bad))
 summary=[]
 for model in dict.fromkeys(c['model'] for c in plan):
  by={s:[x['metrics']['total_response_s'] for x in records if x['model']==model and x['strategy']==s and x['valid']] for s in ['cpu','gpu','hybrid']}
  assert all(len(v)==3 for v in by.values())
  summary.append({'model':model,'total_response_s':{s:{'values':v,'mean':statistics.mean(v),'between_process_SD':statistics.stdev(v)} for s,v in by.items()},'roundwise_CPU_over_NPU':[a/b for a,b in zip(by['cpu'],by['gpu'])],'roundwise_CPU_over_hybrid':[a/b for a,b in zip(by['cpu'],by['hybrid'])],'scope':'Selected t6 maskFC FAon, synthetic2048-token prompt/128outputs/c2560/b512/ub256, warm model;3 rotated independent processes per mode. NPU mode labeled gpu by prototype. No per-step profiling. Does not establish sustained heat/energy, cold start or universal superiority.'})
 summary_path=plan_file.with_suffix('.summary.json');summary_path.write_text(json.dumps(summary,indent=2)+'\n')
 publish('complete',summary=str(summary_path))
except (RuntimeError,KeyboardInterrupt) as e:
 publish('stopped-with-evidence',reason=str(e));raise SystemExit(str(e))
