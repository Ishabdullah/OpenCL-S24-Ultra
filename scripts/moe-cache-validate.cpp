// Full-model teacher-forced equivalence gate. Both passes share one model;
// contexts are sequential. Cache pass uses the exact baseline input tokens.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
static unsigned long long read_bytes() {
    std::ifstream f("/proc/self/io"); std::string k; unsigned long long n;
    while(f>>k>>n) if(k=="read_bytes:") return n;
    return 0;
}
static void evaluate(llama_context * ctx, llama_batch_ext * batch, const llama_token * tokens, int count, int pos) {
    llama_batch_ext_clear(batch);
    for(int i=0;i<count;++i) {
        int index=llama_batch_ext_add_token(batch,0,tokens[i]); llama_pos p=pos+i;
        if(!llama_batch_ext_set_pos(batch,index,&p)) throw std::runtime_error("position failed");
    }
    llama_batch_ext_set_output_logits(batch,count-1,true);
    if(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch)) throw std::runtime_error("decode failed");
    llama_synchronize(ctx);
}
int main(int argc,char ** argv) {
    if(argc!=3) { fprintf(stderr,"usage: %s MODEL STEPS\n",argv[0]); return 2; }
    try {
        const int steps=std::stoi(argv[2]);
        if(steps<2 || steps>128) throw std::runtime_error("steps must be 2..128");
        unsetenv("GGML_CPU_MOE_CACHE_SLOTS");
        llama_backend_init();
        auto mp=llama_model_default_params();
        ggml_backend_dev_t devices[]={ggml_backend_dev_by_name("HTP0"),nullptr};
        if(!devices[0]) throw std::runtime_error("HTP0 missing");
        mp.devices=devices;mp.n_gpu_layers=99;mp.load_mode=LLAMA_LOAD_MODE_MMAP;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
        if(!model) throw std::runtime_error("model load failed");
        const std::string text="Write a Python function to reverse a string:";
        auto * vocab=llama_model_get_vocab(model.get());const int nv=llama_vocab_n_tokens(vocab);
        int nt=-llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);
        if(nt<=0) throw std::runtime_error("tokenization failed");
        std::vector<llama_token> prompt(nt), teacher;
        if(llama_tokenize(vocab,text.data(),text.size(),prompt.data(),nt,true,true)!=nt) throw std::runtime_error("tokenization changed");
        std::vector<std::vector<float>> expected;
        double elapsed[2]={};unsigned long long reads[2]={};size_t identical=0;double max_error=0;
        for(int pass=0;pass<2;++pass) {
            if(pass) { setenv("GGML_CPU_MOE_CACHE_SLOTS","32",1);setenv("GGML_CPU_MOE_CACHE_MIB","1280",1); }
            auto cp=llama_context_default_params();
            cp.n_ctx=nt+steps+16;cp.n_batch=cp.n_ubatch=128;cp.n_threads=cp.n_threads_batch=6;
            cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=true;
            std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
            if(!ctx) throw std::runtime_error("context creation failed");
            std::unique_ptr<llama_batch_ext,decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()),llama_batch_ext_free);
            auto start=Clock::now();auto before=read_bytes();
            for(int step=0;step<steps;++step) {
                if(step==0) evaluate(ctx.get(),batch.get(),prompt.data(),nt,0);
                else evaluate(ctx.get(),batch.get(),&teacher[step-1],1,nt+step-1);
                const float * logits=llama_get_logits_ith(ctx.get(),-1);
                for(int i=0;i<nv;++i) if(!std::isfinite(logits[i])) throw std::runtime_error("non-finite logits");
                if(!pass) {
                    expected.emplace_back(logits,logits+nv);
                    teacher.push_back(std::max_element(logits,logits+nv)-logits);
                } else {
                    bool same=memcmp(logits,expected[step].data(),nv*sizeof(float))==0;
                    if(same) ++identical;
                    for(int i=0;i<nv;++i) max_error=std::max(max_error,double(std::abs(logits[i]-expected[step][i])));
                    if(!same) throw std::runtime_error("full-vocabulary logits differ at step "+std::to_string(step));
                }
                fprintf(stderr,"CACHE_GATE pass=%s step=%d/%d token=%d\n",pass?"cache32":"baseline",step+1,steps,teacher[step]);
            }
            elapsed[pass]=std::chrono::duration<double>(Clock::now()-start).count();reads[pass]=read_bytes()-before;
            fprintf(stderr,"CACHE_GATE pass=%s seconds=%.3f storage_bytes=%llu\n",pass?"cache32":"baseline",elapsed[pass],reads[pass]);
        }
        unsetenv("GGML_CPU_MOE_CACHE_SLOTS");unsetenv("GGML_CPU_MOE_CACHE_MIB");
        model.reset();llama_backend_free();
        printf("{\"steps\":%d,\"prompt_tokens\":%d,\"vocab_size\":%d,\"bit_identical_steps\":%zu,\"maximum_logit_error\":%.9g,\"baseline_s\":%.6f,\"cached_s\":%.6f,\"baseline_storage_bytes\":%llu,\"cached_storage_bytes\":%llu}\n",steps,nt,nv,identical,max_error,elapsed[0],elapsed[1],reads[0],reads[1]);
        return 0;
    } catch(const std::exception & e) { fprintf(stderr,"CACHE_GATE FAILED: %s\n",e.what());return 1; }
}
