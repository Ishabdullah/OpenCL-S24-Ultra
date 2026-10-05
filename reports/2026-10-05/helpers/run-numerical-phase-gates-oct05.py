"""Numerical gates retain timing exclusions and never yield performance claims."""
import datetime,json,pathlib,sys
import measure
from validation import issues
B=pathlib.Path(__file__).resolve().parent
p=pathlib.Path(sys.argv[1]);epoch=measure.conditions()['power_epoch'];rows=[]
for c in json.loads(p.read_text()):
 assert c.get('validation_timing_not_comparable') and (c.get('reference_in') or c.get('reference_out'))
 if measure.conditions()['power_epoch']!=epoch:raise SystemExit('Power epoch changed; preserve gate records')
 r=measure.run(c);m=r.get('latency_metrics') or {};bad=[]
 if r.get('exit_code')!=0 or r.get('abort_reason') or r.get('parse_error'):bad.append('Process failed/aborted or output did not parse')
 if not all(m.get(k) for k in ['all_logits_finite','same_model_and_cache','positions_preserved','validation_run']):bad.append('Numerical finite/state flags absent')
 if c.get('reference_in')!='-' and (m.get('top1_agreements')!=c['generation'] or m.get('mean_KL',float('inf'))>.001):bad.append('Forced-prefix predictions/KL gate failed')
 if 'PHASE cleanup complete' not in (B/'runs'/c['id']/'stderr.txt').read_text():bad.append('Owned cleanup marker missing')
 rows.append({'id':c['id'],'numerical_gate_pass':not bad,'numerical_issues':bad,'timing_issues_preserved':issues(r,B/'runs'/c['id']),'scope':'Forced-prefix numerical gate only; ALL timings excluded, base safety/memory/library guards unchanged. Required-core timing exclusions preserved, not removed.','metrics':m})
 p.with_name(p.stem+'.numerical-summary.json').write_text(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':rows},indent=2)+'\n')
 print(c['id'],{k:m.get(k) for k in ['top1_agreements','mean_KL','all_logits_finite']},bad,flush=True)
 if bad:raise SystemExit('Numerical gate failed; performance withheld')
