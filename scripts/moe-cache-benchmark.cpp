// Default ABBA experiment or explicit cache/thread sweep, one loaded model.
// Only forward evaluation is timed. Reference sampling and verification are
// outside timing. Every later pass replays identical sampled input tokens.
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
#include <thread>
#include <vector>
using Clock = std::chrono::steady_clock;
struct configuration { int slots, threads, mib; };
static std::vector<configuration> plan() {
    const char * setting=getenv("MOE_BENCH_PLAN");
    if(!setting) return {{0,6,0},{32,6,1280},{32,6,1280},{0,6,0}};
    std::vector<configuration> result;
    std::string value(setting);size_t start=0;
    while(start<value.size()) {
        const size_t end=value.find(',',start);
        const std::string item=value.substr(start,end==std::string::npos?end:end-start);
        configuration config;char extra;
        if(sscanf(item.c_str(),"%d:%d:%d%c",&config.slots,&config.threads,&config.mib,&extra)!=3 ||
           config.slots<0 || config.slots>256 || config.threads<1 || config.threads>8 ||
           config.mib<0 || config.mib>8192 || (config.slots && config.mib<64))
            throw std::runtime_error("invalid MOE_BENCH_PLAN; use slots:threads:MiB,...");
        result.push_back(config);
        if(end==std::string::npos) break;
        start=end+1;
        if(start==value.size()) throw std::runtime_error("trailing plan separator");
    }
    if(result.size()<2 || result.size()>20) throw std::runtime_error("plan requires 2..20 passes");
    return result;
}
static unsigned long long read_bytes() {
    std::ifstream f("/proc/self/io");std::string key;unsigned long long n;
    while(f>>key>>n) if(key=="read_bytes:") return n;
    throw std::runtime_error("process storage counter unavailable");
}
static long field(const char * file,const char * key) {
    std::ifstream f(file);std::string line;
    while(std::getline(f,line)) if(line.rfind(key,0)==0) {
        return std::stol(line.substr(strlen(key)));
    }
    return -1;
}
// Optional external thermal gate, after context setup and outside timing.
// A distinct approval file per pass prevents an earlier release being reused.
static void await_admission(int pass,int slots) {
    const char * directory=getenv("MOE_BENCH_ADMISSION_DIR");
    if(!directory) return;
    const std::string base=std::string(directory)+"/";
    std::ofstream request(base+"request.json");
    request<<"{\"pass\":"<<pass<<",\"slots\":"<<slots<<"}\n";
    request.close();
    if(!request) throw std::runtime_error("cannot write admission request");
    fprintf(stderr,"CACHE_BENCH waiting pass=%d slots=%d\n",pass,slots);
    const auto deadline=Clock::now()+std::chrono::seconds(930);
    while(Clock::now()<deadline) {
        std::ifstream approval(base+"approve-"+std::to_string(pass));
        int approved=0;
        if(approval>>approved && approved==pass) {
            fprintf(stderr,"CACHE_BENCH admitted pass=%d slots=%d\n",pass,slots);
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("thermal admission timed out");
}
static std::string quote(const std::string & value) {
    std::string out="\"";
    for(unsigned char c:value) {
        if(c=='"'||c=='\\') {out+='\\';out+=c;}
        else if(c<32) {char b[7];snprintf(b,sizeof(b),"\\u%04x",c);out+=b;}
        else out+=c;
    }
    return out+'"';
}
static void evaluate(llama_context * ctx,llama_batch_ext * batch,const llama_token * tokens,int count,int pos) {
    llama_batch_ext_clear(batch);
    for(int i=0;i<count;++i) {
        int index=llama_batch_ext_add_token(batch,0,tokens[i]);llama_pos p=pos+i;
        if(!llama_batch_ext_set_pos(batch,index,&p)) throw std::runtime_error("position failed");
    }
    llama_batch_ext_set_output_logits(batch,count-1,true);
    if(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch)) throw std::runtime_error("decode failed");
    llama_synchronize(ctx);
}
int main(int argc,char ** argv) {
    if(argc!=3) {fprintf(stderr,"usage: %s MODEL MAX_OUTPUT_TOKENS\n",argv[0]);return 2;}
    try {
        size_t consumed=0;const int limit=std::stoi(argv[2],&consumed);
        if(consumed!=strlen(argv[2])||limit<2||limit>128) throw std::runtime_error("token limit must be 2..128");
        const auto configurations=plan();
        unsetenv("GGML_CPU_MOE_CACHE_SLOTS");unsetenv("GGML_CPU_MOE_CACHE_MIB");
        llama_backend_init();
        auto mp=llama_model_default_params();
        ggml_backend_dev_t devices[]={ggml_backend_dev_by_name("HTP0"),nullptr};
        if(!devices[0]) throw std::runtime_error("HTP0 missing");
        mp.devices=devices;mp.n_gpu_layers=99;mp.load_mode=LLAMA_LOAD_MODE_MMAP;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
        if(!model) throw std::runtime_error("model load failed");
        const std::string text="Write a Python binary search tree implementation with insert, search, delete, and unit tests. Explain the edge cases:";
        auto * vocab=llama_model_get_vocab(model.get());const int nv=llama_vocab_n_tokens(vocab);
        int nt=-llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);
        if(nt<=0||nt>128) throw std::runtime_error("invalid prompt token count");
        std::vector<llama_token> prompt(nt),teacher;
        if(llama_tokenize(vocab,text.data(),text.size(),prompt.data(),nt,true,true)!=nt) throw std::runtime_error("tokenization failed");
        std::vector<std::vector<float>> expected;
        std::string generated;bool eog=false;
        std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)> sampler(
            llama_sampler_chain_init(llama_sampler_chain_default_params()),llama_sampler_free);
        llama_sampler_chain_add(sampler.get(),llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler.get(),llama_sampler_init_penalties(nv,64,1.1f,0,0));
        llama_sampler_chain_add(sampler.get(),llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(sampler.get(),llama_sampler_init_dist(20261010u));
        for(auto token:prompt) llama_sampler_accept(sampler.get(),token);
        printf("{\"type\":\"config\",\"model\":%s,\"prompt\":%s,\"prompt_tokens\":%d,\"maximum_output_tokens\":%d,\"vocab_size\":%d,\"slots_order\":[",quote(argv[1]).c_str(),quote(text).c_str(),nt,limit,nv);
        for(size_t i=0;i<configurations.size();++i) printf("%s%d",i?",":"",configurations[i].slots);
        printf("],\"plan\":[");
        for(size_t i=0;i<configurations.size();++i) printf("%s{\"slots\":%d,\"threads\":%d,\"cache_mib\":%d}",i?",":"",configurations[i].slots,configurations[i].threads,configurations[i].mib);
        printf("],\"reference_threads\":%d,\"batch\":128,\"ubatch\":128,\"rest_s\":30,\"seed\":20261010,\"top_k\":40,\"temperature\":0.8,\"repeat_penalty\":1.1,\"timing\":\"synchronized forward calls only\"}\n",configurations.front().threads);fflush(stdout);
        std::vector<double> prefill(configurations.size()),decode(configurations.size());
        std::vector<unsigned long long> storage(configurations.size());
        for(size_t pass=0;pass<configurations.size();++pass) {
            const auto & config=configurations[pass];
            if(pass) std::this_thread::sleep_for(std::chrono::seconds(30));
            if(config.slots) {
                setenv("GGML_CPU_MOE_CACHE_SLOTS",std::to_string(config.slots).c_str(),1);
                setenv("GGML_CPU_MOE_CACHE_MIB",std::to_string(config.mib).c_str(),1);
            } else {unsetenv("GGML_CPU_MOE_CACHE_SLOTS");unsetenv("GGML_CPU_MOE_CACHE_MIB");}
            auto cp=llama_context_default_params();
            cp.n_ctx=nt+limit+16;cp.n_batch=cp.n_ubatch=128;cp.n_threads=cp.n_threads_batch=config.threads;
            cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=true;
            std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
            if(!ctx) throw std::runtime_error("context creation failed");
            std::unique_ptr<llama_batch_ext,decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()),llama_batch_ext_free);
            if(!batch) throw std::runtime_error("batch allocation failed");
            await_admission(int(pass)+1,config.slots);
            fprintf(stderr,"CACHE_BENCH start pass=%zu slots=%d threads=%d mib=%d\n",pass+1,config.slots,config.threads,config.mib);
            const long available_start=field("/proc/meminfo","MemAvailable:");
            const long rss_start=field("/proc/self/status","VmRSS:");
            const int steps=pass?int(expected.size()):limit;
            size_t comparisons=0;unsigned long long prefill_read=0,decode_read=0;
            for(int step=0;step<steps;++step) {
                const auto before=read_bytes();const auto start=Clock::now();
                if(step==0) evaluate(ctx.get(),batch.get(),prompt.data(),nt,0);
                else evaluate(ctx.get(),batch.get(),&teacher[step-1],1,nt+step-1);
                const double duration=std::chrono::duration<double>(Clock::now()-start).count();
                const auto bytes=read_bytes()-before;
                if(step==0) {prefill[pass]=duration;prefill_read=bytes;}
                else {decode[pass]+=duration;decode_read+=bytes;}
                const float * logits=llama_get_logits_ith(ctx.get(),-1);
                if(pass==0) {
                    for(int i=0;i<nv;++i) if(!std::isfinite(logits[i])) throw std::runtime_error("non-finite reference logits");
                    expected.emplace_back(logits,logits+nv);
                    auto token=llama_sampler_sample(sampler.get(),ctx.get(),-1);
                    llama_sampler_accept(sampler.get(),token);teacher.push_back(token);
                    eog=llama_vocab_is_eog(vocab,token);
                    if(!eog) {
                        std::vector<char> piece(128);int size=llama_token_to_piece(vocab,token,piece.data(),piece.size(),0,true);
                        if(size<0) {piece.resize(-size);size=llama_token_to_piece(vocab,token,piece.data(),piece.size(),0,true);}
                        if(size<0) throw std::runtime_error("token piece conversion failed");
                        generated.append(piece.data(),size);
                    }
                } else {
                    if(memcmp(logits,expected[step].data(),nv*sizeof(float))) {
                        double error=0;size_t different=0;
                        for(int i=0;i<nv;++i) if(logits[i]!=expected[step][i]) {++different;error=std::max(error,double(std::abs(logits[i]-expected[step][i])));}
                        fprintf(stderr,"BENCH_MISMATCH pass=%zu slots=%d step=%d different=%zu max_error=%.9g\n",pass+1,config.slots,step,different,error);
                        throw std::runtime_error("reference logit mismatch");
                    }
                    ++comparisons;
                }
                if(step==0||(step+1)%8==0) fprintf(stderr,"CACHE_BENCH pass=%zu/%zu slots=%d threads=%d step=%d/%d decode_s=%.3f\n",pass+1,configurations.size(),config.slots,config.threads,step+1,steps,decode[pass]);
                if(pass==0&&eog) break;
            }
            storage[pass]=prefill_read+decode_read;
            const int actual_steps=int(expected.size());
            if(pass==0) {
                printf("{\"type\":\"reference\",\"steps\":%d,\"eog\":%s,\"text\":%s,\"tokens\":[",actual_steps,eog?"true":"false",quote(generated).c_str());
                for(size_t i=0;i<teacher.size();++i) printf("%s%d",i?",":"",teacher[i]);printf("]}\n");
            }
            printf("{\"type\":\"pass\",\"pass\":%zu,\"slots\":%d,\"threads\":%d,\"cache_mib\":%d,\"evaluated_positions\":%d,\"decode_evaluations\":%d,\"bit_identical_positions\":%zu,\"prefill_s\":%.6f,\"decode_s\":%.6f,\"evaluation_s\":%.6f,\"decode_tok_s\":%.6f,\"prefill_storage_bytes\":%llu,\"decode_storage_bytes\":%llu,\"storage_bytes\":%llu,\"available_start_kib\":%ld,\"available_end_kib\":%ld,\"rss_start_kib\":%ld,\"rss_end_kib\":%ld}\n",pass+1,config.slots,config.threads,config.mib,actual_steps,actual_steps-1,comparisons,prefill[pass],decode[pass],prefill[pass]+decode[pass],decode[pass]>0?(actual_steps-1)/decode[pass]:0,prefill_read,decode_read,storage[pass],available_start,field("/proc/meminfo","MemAvailable:"),rss_start,field("/proc/self/status","VmRSS:"));fflush(stdout);
            fprintf(stderr,"CACHE_BENCH complete pass=%zu slots=%d eval_s=%.3f decode_s=%.3f storage_bytes=%llu\n",pass+1,config.slots,prefill[pass]+decode[pass],decode[pass],storage[pass]);
        }
        unsetenv("GGML_CPU_MOE_CACHE_SLOTS");unsetenv("GGML_CPU_MOE_CACHE_MIB");
        model.reset();sampler.reset();llama_backend_free();
        printf("{\"type\":\"complete\",\"passes\":%zu,\"reference_positions\":%zu,\"verified_positions\":%zu}\n",configurations.size(),expected.size(),expected.size()*(configurations.size()-1));fflush(stdout);
        return 0;
    } catch(const std::exception & e) {fprintf(stderr,"CACHE_BENCH FAILED: %s\n",e.what());return 1;}
}
