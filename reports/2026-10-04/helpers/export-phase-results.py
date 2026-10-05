"""Read-only export of phase measurements; preserve source and exclusions."""
import argparse,csv,json,pathlib,fcntl,os
from validation import issues
B=pathlib.Path(__file__).resolve().parent
lock=(B/'resource.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX)
os.setpriority(os.PRIO_PROCESS,0,max(10,os.getpriority(os.PRIO_PROCESS,0)))
os.sched_setaffinity(0,{0,1})
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--epoch',help='Export a recorded power epoch without changing current guards')
args=parser.parse_args()
E=args.epoch or json.loads((B/'conditions.json').read_text())['power_epoch'];out=B/'npu-investigation'/('results-'+E);out.mkdir(exist_ok=True)
models={m['id']:m for m in json.loads((B/'models.json').read_text())}
rows=[];records=[]
for p in (B/'runs').glob('*/result.json'):
 d=json.loads(p.read_text());m=d.get('latency_metrics');c=d.get('config',{})
 if not m or d.get('power_conditions',{}).get('power_epoch')!=E:continue
 bad=issues(d,p.parent)
 row={'id':c.get('id'),'model':c.get('model'),'device':c.get('device'),'strategy':c.get('strategy'),'quality_only':bool(c.get('validation_timing_not_comparable')),'valid':not bad,'issues':'; '.join(bad),'prompt':c.get('prompt'),'generation_requested':c.get('generation'),'context':c.get('context'),'threads':c.get('threads'),'process_mask':c.get('process_mask'),'attention':c.get('attention'),'batch':c.get('batch'),'ubatch':c.get('ubatch'),'binary':c.get('binary')}
 row.update(power_epoch=E,source_variant=c.get('source_variant'),ngl=c.get('ngl'),CPU_repack=c.get('repack',1),profiling=c.get('environment',{}).get('GGML_HEXAGON_PROFILE'),registration_window_MiB=c.get('environment',{}).get('GGML_HEXAGON_REGISTRATION_WINDOW'),DSP_mapping_budget_MiB=c.get('environment',{}).get('GGML_HEXAGON_VMEM'))
 metadata=models[c['model']]
 for k in ['architecture','quantization','tensor_parameter_count','file_bytes','weight_data_bytes','blocks','attention_heads','KV_heads','trained_context']:row[k]=metadata.get(k)
 for k in ['prefill_s','prefill_tok_s','ttft_s','decode_s','decode_tok_s','first_decode_s','handoff_ms','handoff_weights_ms','handoff_kv_ms','handoff_scheduler_ms','total_response_s','kv_copy_bytes','weight_copy_bytes','cpu_shadow_weight_bytes','gpu_weight_bytes_released','all_logits_finite','top1_agreements','mean_KL','logit_RMSE','max_abs_logit_error','postload_admission_s','request_start_CPU_max_C','request_start_NPU_max_C','request_start_GPU_max_C','request_start_battery_C']:row[k]=m.get(k)
 row['allocations_json']=json.dumps(d.get('allocations'),separators=(',',':'));row['raw_result']=str(p)
 for label,k in [('max_RSS_MiB','max_process_RSS_kB'),('max_PSS_MiB','max_process_PSS_kB'),('min_MemAvailable_MiB','min_system_MemAvailable_kB')]:row[label]=d.get(k,0)/1024
 for k in ['actual_context','same_model_and_cache','positions_preserved']:row[k]=m.get(k)
 row['handoff_plus_first_decode_s']=(m.get('handoff_ms',0)/1000+m.get('first_decode_s',0)) if c.get('strategy')=='hybrid' else None
 command_path=p.with_name('command.json');command=json.loads(command_path.read_text()) if command_path.exists() else {}
 records.append({'id':row['id'],'config':c,'command_provenance':command,'metrics':m,'issues':bad,'exit_code':d.get('exit_code'),'allocations':d.get('allocations'),'llama_library_paths':d.get('llama_library_paths'),'memory':{k:row[k] for k in ['max_RSS_MiB','max_PSS_MiB','min_MemAvailable_MiB']},'raw_result':str(p)})
 rows.append(row)
rows.sort(key=lambda x:x['id'])
if rows:
 with (out/'phase-responses.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'phase-responses.json').write_text(json.dumps(rows,indent=2)+'\n')
 (out/'phase-records.json').write_text(json.dumps(records,indent=2)+'\n')
print('exported',len(rows),'phase records into',out)
