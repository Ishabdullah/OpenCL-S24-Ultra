"""Measure cumulative FastRPC counters at completed request phase boundaries."""
import datetime,json,pathlib,sys
from validation import issues
B=pathlib.Path(__file__).resolve().parent
plan=pathlib.Path(sys.argv[1]);records=[]
for c in json.loads(plan.read_text()):
 out=B/'runs'/c['id'];path=out/'result.json'
 if not path.exists():continue
 d=json.loads(path.read_text())
 if d.get('status')=='guarded_skip':records.append({'id':c['id'],'status':'guarded_skip','reason':d.get('abort_reason')});continue
 rows=[]
 for line in (out/'stdout.txt').read_text().splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if r.get('iteration',-1)>=0:rows.append(r)
 counters=[]
 for line in (out/'stderr.txt').read_text().splitlines():
  if not line.startswith('REGPROFILE '):continue
  r={k:float(v) for k,v in (field.split('=',1) for field in line.split()[1:])};counters.append(r)
 groups=[]
 for request in rows:
  start=request['monotonic_s'];split=start+request['prefill_s'];end=start+request['total_response_s']
  def preceding(t):
   eligible=[r for r in counters if r['mono']<=t]
   assert eligible,'Missing counter before phase boundary'
   return eligible[-1]
  for name,lo,hi,seconds,steps in [('prefill',start,split,request['prefill_s'],request['prompt']),('decode',split,end,request['decode_s'],request['decode_steps'])]:
   a,b=preceding(lo),preceding(hi);delta={k:b[k]-a[k] for k in a if k!='mono'}
   groups.append({'phase':name,'iteration':request['iteration'],'wall_s':seconds,'tokens_or_steps':steps,'counter_delta':delta,'maps_per_token_or_step':delta['maps']/steps,'map_wall_fraction':delta['map_ms']/1000/seconds,'unmap_wall_fraction':delta['unmap_ms']/1000/seconds,'eviction_flush_wall_fraction':delta['eviction_flush_ms']/1000/seconds,'final_sync_wall_fraction':delta['final_sync_ms']/1000/seconds,'boundary_counter_lag_s':[lo-a['mono'],hi-b['mono']]})
 records.append({'id':c['id'],'window_MiB':int(c['environment']['GGML_HEXAGON_REGISTRATION_WINDOW']),'controlled_issues':issues(d,out),'requests':rows,'phase_groups':groups,'raw_counters':counters,'allocations':d.get('allocations'),'library_paths':d.get('llama_library_paths')})
summary={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Profiled diagnostics, not final throughput. Counter differences use last synchronized cumulative snapshot before each measured phase boundary. Mapping/unmapping is host/vendor wall time, may overlap DSP work. Eviction and final-sync waits include DSP execution; fractions are not disjoint or pure-overhead attribution. Worker-placement guards remain required.','records':records}
output=plan.with_name(plan.stem+'.profile-summary.json');output.write_text(json.dumps(summary,indent=2)+'\n')
for r in records:
 print(r['id'],r.get('controlled_issues',r.get('reason')))
 for g in r.get('phase_groups',[]):print(g['phase'],g['wall_s'],'maps/step',g['maps_per_token_or_step'],'map/wait fractions',g['map_wall_fraction'],g['eviction_flush_wall_fraction'])
print(output)
