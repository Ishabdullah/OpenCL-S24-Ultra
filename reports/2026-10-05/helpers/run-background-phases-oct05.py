"""Keep deliberately restricted background diagnostics separate from foreground timings."""
import datetime,json,pathlib,sys
import measure
from validation import issues
B=pathlib.Path(__file__).resolve().parent
plan=pathlib.Path(sys.argv[1]);epoch=measure.conditions()['power_epoch'];rows=[]
for c in json.loads(plan.read_text()):
 if measure.conditions()['power_epoch']!=epoch:raise SystemExit('Power epoch changed; stop background set')
 fragments=c['required_harness_cgroup_fragments']
 if not all(x in (measure.system()['harness_cgroup'] or '') for x in fragments):raise SystemExit('Expected restricted background state no longer present')
 r=measure.run(c);bad=issues(r,B/'runs'/c['id']);m=r.get('latency_metrics') or {}
 if not all(x in (r.get('before',{}).get('harness_cgroup') or '') for x in fragments):bad.append('Run admitted outside declared background state')
 if not all(m.get(k) for k in ['all_logits_finite','same_model_and_cache','positions_preserved']):bad.append('Finite/state gate missing')
 rows.append({'id':c['id'],'valid':not bad,'issues':bad,'metrics':m})
 plan.with_suffix('.state.json').write_text(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':rows},indent=2)+'\n')
 print(c['id'],{k:m.get(k) for k in ['prefill_tok_s','decode_tok_s','total_response_s']},'issues',bad,flush=True)
 if bad:raise SystemExit('Excluded restricted background timing')
