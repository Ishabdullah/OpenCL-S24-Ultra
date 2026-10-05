// Isolate host/DSP mapping lifetime with tiny validated operations, no model.
#include "llama.h"
#include "ggml-backend.h"
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
static std::atomic<int> stop_signal{0};
static_assert(std::atomic<int>::is_always_lock_free,"signal atomics");
static void stop(int n){stop_signal.store(n,std::memory_order_relaxed);}
static size_t available_kb(){std::ifstream f("/proc/meminfo");std::string line;size_t x=0;while(std::getline(f,line))if(sscanf(line.c_str(),"MemAvailable: %zu kB",&x)==1)return x;return 0;}
struct Resources {
 ggml_backend_t backend=nullptr;ggml_context *ctx=nullptr;
 ggml_backend_buffer_t output=nullptr;std::vector<ggml_backend_buffer_t> inputs;
 ~Resources(){if(ctx)ggml_free(ctx);if(output)ggml_backend_buffer_free(output);for(auto b:inputs)ggml_backend_buffer_free(b);if(backend)ggml_backend_free(backend);llama_backend_free();fprintf(stderr,"MMAP_PROBE cleanup complete\n");}
};
int main(int argc,char **argv){
 if(argc!=4)return 2;
 std::signal(SIGTERM,stop);std::signal(SIGINT,stop);Resources r;
 try{
  std::string mode=argv[1];int chunk=std::stoi(argv[2]),count=std::stoi(argv[3]);
  if((mode!="retain"&&mode!="free"&&mode!="unregister")||chunk<1||chunk>1024||count<1||count>16)return 2;
  llama_backend_init();auto dev=ggml_backend_dev_by_name("HTP0");if(!dev)throw std::runtime_error("HTP0 unavailable");
  r.backend=ggml_backend_dev_init(dev,nullptr);if(!r.backend)throw std::runtime_error("HTP0 initialization failed");
  using unmap_fn=int(*)(ggml_backend_t,ggml_backend_buffer_t);
  auto unmap=reinterpret_cast<unmap_fn>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),"ggml_hexagon_probe_unmap"));
  if(mode=="unregister"&&!unmap)throw std::runtime_error("isolated diagnostic unmap function missing");
  size_t bytes=size_t(chunk)*1024*1024;
  for(int i=0;i<count;i++){
   if(stop_signal.load())throw std::runtime_error("probe interrupted");
   size_t avail=available_kb();
   if(avail<(1536ULL+128+chunk)*1024){printf("{\"status\":\"guarded_skip\",\"index\":%d,\"MemAvailable_kB\":%zu,\"physical_reserve_MiB\":1536}\n",i,avail);fflush(stdout);return 5;}
   ggml_init_params params{ggml_graph_overhead_custom(16,false)+4*ggml_tensor_overhead()+4096,nullptr,true};
   r.ctx=ggml_init(params);if(!r.ctx)throw std::runtime_error("metadata allocation failed");
   auto *x=ggml_new_tensor_1d(r.ctx,GGML_TYPE_F32,32);ggml_set_name(x,"probe_input");
   auto in=ggml_backend_alloc_buffer(r.backend,bytes);if(!in)throw std::runtime_error("RPC buffer allocation failed");r.inputs.push_back(in);
   ggml_backend_buffer_set_usage(in,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
   if(ggml_backend_tensor_alloc(in,x,ggml_backend_buffer_get_base(in))!=GGML_STATUS_SUCCESS)throw std::runtime_error("input binding failed");
   float values[32];for(int j=0;j<32;j++)values[j]=float(j)-7.f;ggml_backend_tensor_set(x,values,0,sizeof(values));
   auto *y=ggml_scale(r.ctx,x,2.f);ggml_set_name(y,"probe_output");
   if(!ggml_backend_supports_op(r.backend,y))throw std::runtime_error("HTP SCALE shape unsupported; do not infer mapping behavior");
   r.output=ggml_backend_alloc_buffer(r.backend,4096);if(!r.output)throw std::runtime_error("output allocation failed");
   if(ggml_backend_tensor_alloc(r.output,y,ggml_backend_buffer_get_base(r.output))!=GGML_STATUS_SUCCESS)throw std::runtime_error("output binding failed");
   printf("{\"status\":\"before_graph\",\"index\":%d,\"mode\":\"%s\",\"chunk_MiB\":%d,\"retained_inputs\":%zu,\"MemAvailable_kB\":%zu}\n",i,mode.c_str(),chunk,r.inputs.size(),available_kb());fflush(stdout);
   auto *g=ggml_new_graph_custom(r.ctx,16,false);ggml_build_forward_expand(g,y);
   if(ggml_backend_graph_compute(r.backend,g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("graph failed");ggml_backend_synchronize(r.backend);
   float actual[32];ggml_backend_tensor_get(y,actual,0,sizeof(actual));
   double max_error=0;for(int j=0;j<32;j++){double expected=double(values[j])*2.0;double error=std::abs(double(actual[j])-expected);max_error=std::max(max_error,error);if(!std::isfinite(actual[j])||error>1e-5+1e-6*std::abs(expected)){fprintf(stderr,"MMAP_PROBE mismatch index=%d expected=%.9g actual=%.9g scale_params0=%d\n",j,expected,double(actual[j]),y->op_params[0]);throw std::runtime_error("SCALE numerical mismatch");}}
   printf("{\"status\":\"graph_passed\",\"index\":%d,\"all_values_finite\":true,\"within_tolerance\":true,\"max_abs_error\":%.9g}\n",i,max_error);fflush(stdout);
   if(mode=="unregister"){
    void *base=ggml_backend_buffer_get_base(in);
    for(int repeat=0;repeat<2;repeat++){
     int err=unmap(r.backend,in);if(err!=0)throw std::runtime_error("unregister returned "+std::to_string(err));
     if(ggml_backend_buffer_get_base(in)!=base)throw std::runtime_error("physical buffer base changed");
     float retained[32];ggml_backend_tensor_get(x,retained,0,sizeof(retained));
     for(int j=0;j<32;j++)if(retained[j]!=values[j])throw std::runtime_error("retained input changed");
     printf("{\"status\":\"registration_released\",\"index\":%d,\"repeat\":%d,\"storage_preserved\":true}\n",i,repeat);fflush(stdout);
     if(repeat==0){
      if(ggml_backend_graph_compute(r.backend,g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("remapped graph failed");ggml_backend_synchronize(r.backend);
      ggml_backend_tensor_get(y,actual,0,sizeof(actual));
      for(int j=0;j<32;j++){double expected=double(values[j])*2.0;double error=std::abs(double(actual[j])-expected);if(!std::isfinite(actual[j])||error>1e-5+1e-6*std::abs(expected))throw std::runtime_error("remapped SCALE mismatch");}
      printf("{\"status\":\"remapped_graph_passed\",\"index\":%d}\n",i);fflush(stdout);
     }
    }
   }
   ggml_free(r.ctx);r.ctx=nullptr;ggml_backend_buffer_free(r.output);r.output=nullptr;
   if(mode=="free"){ggml_backend_buffer_free(r.inputs.back());r.inputs.pop_back();}
  }
  printf("{\"status\":\"complete\",\"mode\":\"%s\",\"count\":%d,\"no_model\":true}\n",mode.c_str(),count);fflush(stdout);return 0;
 }catch(const std::exception &e){fprintf(stderr,"MMAP_PROBE failure: %s\n",e.what());int n=stop_signal.load();return n?128+n:4;}
}
