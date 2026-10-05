#include "llama.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <atomic>
#include <csignal>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using clock_type = std::chrono::steady_clock;
static std::atomic<int> stop_signal{0};
static_assert(std::atomic<int>::is_always_lock_free, "Signal handling requires lock-free atomics");
static void request_stop(int sig) { stop_signal.store(sig, std::memory_order_relaxed); }
static void check_stop() {
    if (stop_signal.load(std::memory_order_relaxed)) throw std::runtime_error("phase measurement interrupted");
}
struct RuntimeState {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    ggml_threadpool *pool = nullptr;
    llama_batch_ext *batch = nullptr;
    RuntimeState() { llama_backend_init(); }
    ~RuntimeState() {
        if (batch) llama_batch_ext_free(batch);
        if (ctx) llama_detach_threadpool(ctx);
        if (pool) ggml_threadpool_free(pool);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
        fprintf(stderr, "PHASE cleanup complete\n");
    }
};
static double seconds(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}
static double monotonic_s() {
    return std::chrono::duration<double>(clock_type::now().time_since_epoch()).count();
}
static void memory_snapshot(const char * stage) {
    fprintf(stderr,"HYBRID_MEMORY: stage=%s monotonic_s=%.9f\n",stage,monotonic_s());
    for (const char * path : {"/proc/self/status","/proc/self/smaps_rollup","/proc/meminfo"}) {
        std::ifstream file(path);std::string line;
        while (std::getline(file,line)) {
            if (line.find("VmRSS:")==0 || line.find("VmHWM:")==0 || line.find("Pss:")==0 ||
                    line.find("Pss_Anon:")==0 || line.find("Pss_File:")==0 || line.find("MemAvailable:")==0 ||
                    line.find("GpuTotal:")==0 || line.find("SwapPss:")==0) fprintf(stderr,"HYBRID_MEMORY: %s %s\n",stage,line.c_str());
        }
    }
    fflush(stderr);
}
static void evaluate(llama_context * ctx, llama_batch_ext * batch,
        const llama_token * tokens, int count, int position, bool logits) {
    check_stop();
    llama_batch_ext_clear(batch);
    for (int i = 0; i < count; ++i) {
        int idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        llama_pos pos = position + i;
        if (!llama_batch_ext_set_pos(batch, idx, &pos)) throw std::runtime_error("batch position failed");
    }
    if (logits) llama_batch_ext_set_output_logits(batch, count-1, true);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) throw std::runtime_error("evaluation failed");
    llama_synchronize(ctx);
}
int main(int argc, char ** argv) {
    // MODEL MODE PROMPT GEN CONTEXT THREADS BATCH UBATCH NGL SHADOW DROP [REFERENCE_IN REFERENCE_OUT]
    if (argc != 12 && argc != 14) return 2;
    std::signal(SIGTERM, request_stop);
    std::signal(SIGINT, request_stop);
    try {
        std::string mode=argv[2], shadow=argv[10];
        const int prompt=std::stoi(argv[3]), generation=std::stoi(argv[4]), context=std::stoi(argv[5]);
        const int threads=std::stoi(argv[6]), batch_size=std::stoi(argv[7]), ubatch=std::stoi(argv[8]), ngl=std::stoi(argv[9]);
        const bool gpu=mode != "cpu", hybrid=mode == "hybrid", drop=std::stoi(argv[11]);
        if ((mode!="cpu" && mode!="gpu" && mode!="hybrid") || prompt<1 || generation<2 || context<prompt+generation || batch_size<1) return 2;
        const bool validation=argc==14;
        std::ifstream reference_in;
        std::ofstream reference_out;
        if (validation && std::string(argv[12])!="-") reference_in.open(argv[12], std::ios::binary);
        if (validation && std::string(argv[13])!="-") reference_out.open(argv[13], std::ios::binary);
        if (validation && std::string(argv[12])!="-" && !reference_in) throw std::runtime_error("reference input unavailable");
        if (validation && std::string(argv[13])!="-" && !reference_out) throw std::runtime_error("reference output unavailable");
        if (hybrid && shadow=="mmap") setenv("GGML_PERF_HYBRID_KEEP_MMAP","1",1);
        else unsetenv("GGML_PERF_HYBRID_KEEP_MMAP");
        auto initialization_start=clock_type::now();
        RuntimeState resources;
        const char * device_env = std::getenv("GGML_PERF_PHASE_DEVICE");
        const std::string device = device_env ? device_env : "HTP0";
        const char * fa_env = std::getenv("GGML_PERF_PHASE_FA");
        const std::string attention = fa_env ? fa_env : "off";
        if ((device != "HTP0" && device != "GPUOpenCL") ||
            (attention != "off" && attention != "on")) throw std::runtime_error("invalid phase driver backend/attention");
        auto mp=llama_model_default_params();
        ggml_backend_dev_t no_devices[]={nullptr};
        ggml_backend_dev_t gpu_devices[]={ggml_backend_dev_by_name(device.c_str()),nullptr};
        if (gpu && !gpu_devices[0]) throw std::runtime_error("requested accelerator not detected");
        mp.devices=gpu ? gpu_devices : no_devices;
        mp.n_gpu_layers=gpu ? ngl : 0;
        // All three modes use the same explicit mmap loading policy.
        mp.load_mode=LLAMA_LOAD_MODE_MMAP;
        auto * model=llama_model_load_from_file(argv[1],mp);
        resources.model = model;
        if (!model) throw std::runtime_error("model load failed");
        auto cp=llama_context_default_params();
        cp.n_ctx=context; cp.n_batch=batch_size; cp.n_ubatch=ubatch;
        cp.n_threads=cp.n_threads_batch=threads;
        cp.flash_attn_type = attention == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k=cp.type_v=GGML_TYPE_F16;
        cp.op_offload=gpu; cp.offload_kqv=gpu;
        auto * ctx=llama_init_from_model(model,cp);
        resources.ctx = ctx;
        if (!ctx) throw std::runtime_error("context allocation failed");
        auto tp=ggml_threadpool_params_default(threads);
        tp.poll=0;
        auto * pool=ggml_threadpool_new(&tp);
        resources.pool = pool;
        if (!pool) throw std::runtime_error("threadpool allocation failed");
        llama_attach_threadpool(ctx,pool,pool);
        const double initialization_s=seconds(initialization_start);
        memory_snapshot("initialized");
        auto * vocab=llama_model_get_vocab(model);
        const int nv=llama_vocab_n_tokens(vocab);
        std::string text="Write clear Python code and explain its behavior.\n";
        while (int(text.size())<prompt*16) text+="def add(a, b):\n    return a + b\n# Explain the result carefully.\n";
        int nt=-llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);
        std::vector<llama_token> tokens(nt);
        nt=llama_tokenize(vocab,text.data(),text.size(),tokens.data(),nt,true,true);
        if (nt<prompt) throw std::runtime_error("insufficient tokenized prompt");
        tokens.resize(prompt);
        uint64_t token_hash=1469598103934665603ULL;
        for (auto token:tokens) { token_hash^=uint32_t(token); token_hash*=1099511628211ULL; }
        auto * batch=llama_batch_ext_init(ctx);
        resources.batch = batch;
        const int warm_count=std::min(8,prompt);
        evaluate(ctx,batch,tokens.data(),warm_count,0,true);
        llama_token warm=std::max_element(llama_get_logits_ith(ctx,-1),llama_get_logits_ith(ctx,-1)+nv)-llama_get_logits_ith(ctx,-1);
        evaluate(ctx,batch,&warm,1,warm_count,true);
        llama_memory_clear(llama_get_memory(ctx),true);
        llama_synchronize(ctx);
        for (int i = 0; i < 100; ++i) {
            check_stop();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        memory_snapshot("before-request");
        auto request_start=clock_type::now();
        const double request_monotonic=monotonic_s();
        fprintf(stderr,"HYBRID_MEASURE: request start mode=%s prompt=%d generation=%d\n",mode.c_str(),prompt,generation);
        for (int pos=0; pos<prompt; pos+=batch_size) {
            int n=std::min(batch_size,prompt-pos);
            evaluate(ctx,batch,tokens.data()+pos,n,pos,pos+n==prompt);
        }
        const double prefill_s=seconds(request_start);
        std::vector<llama_token> output;
        std::vector<float> reference(nv);
        int agreements=0; size_t compared=0; double square_error=0,max_error=0,kl_sum=0;
        auto choose=[&]() {
            const float * logits=llama_get_logits_ith(ctx,-1);
            for (int i=0;i<nv;i++) if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits");
            int best=std::max_element(logits,logits+nv)-logits;
            if (reference_out.is_open()) reference_out.write((const char*)logits,nv*sizeof(float));
            if (reference_in.is_open()) {
                if (!reference_in.read((char*)reference.data(),nv*sizeof(float))) throw std::runtime_error("short reference logits");
                int ref_best=std::max_element(reference.begin(),reference.end())-reference.begin();
                agreements+=best==ref_best;
                double max_a=reference[ref_best],max_b=logits[best],sum_a=0,sum_b=0;
                for (int i=0;i<nv;i++) {
                    double diff=double(logits[i])-reference[i];square_error+=diff*diff;max_error=std::max(max_error,std::abs(diff));
                    sum_a+=std::exp(double(reference[i])-max_a);sum_b+=std::exp(double(logits[i])-max_b);
                }
                double lse_a=max_a+std::log(sum_a),lse_b=max_b+std::log(sum_b);
                for (int i=0;i<nv;i++) kl_sum+=std::exp(double(reference[i])-lse_a)*(reference[i]-lse_a-logits[i]+lse_b);
                compared+=nv;
                best=ref_best; // Forced common trajectory in correctness runs only.
            }
            output.push_back(best);
            return llama_token(best);
        };
        llama_token selected=choose();
        const double ttft_s=seconds(request_start);
        fprintf(stderr,"HYBRID_MEASURE: first token emitted at %.6f s\n",ttft_s);
        memory_snapshot("after-prefill");
        auto * memory_before=llama_get_memory(ctx);
        auto * model_before=llama_get_model(ctx);
        const auto position_before=llama_memory_seq_pos_max(memory_before,0);
        llama_perf_handoff_stats stats{};
        const double switch_monotonic=monotonic_s();
        check_stop();
        if (hybrid && llama_perf_switch_to_cpu(ctx,drop,&stats)) throw std::runtime_error("CPU handoff failed");
        if (memory_before!=llama_get_memory(ctx) || model_before!=llama_get_model(ctx) ||
                position_before!=llama_memory_seq_pos_max(memory_before,0)) throw std::runtime_error("handoff changed model/cache identity or positions");
        memory_snapshot("after-handoff");
        auto decode_start=clock_type::now();
        double first_decode_s=0;
        for (int step=1;step<generation;step++) {
            evaluate(ctx,batch,&selected,1,prompt+step-1,true);
            selected=choose();
            if (step==1) first_decode_s=seconds(decode_start);
        }
        const double decode_s=seconds(decode_start), total_s=seconds(request_start);
        memory_snapshot("after-generation");
        printf("{\"device\":\"%s\",\"attention\":\"%s\",",device.c_str(),attention.c_str());
        printf("\"request_start_monotonic_s\":%.9f,\"switch_start_monotonic_s\":%.9f,",request_monotonic,switch_monotonic);
        printf("\"mode\":\"%s\",\"shadow\":\"%s\",\"prompt\":%d,\"generation\":%d,\"decode_steps\":%d,\"context\":%d,\"actual_context\":%u,\"threads\":%d,\"batch\":%d,\"ubatch\":%d,\"prefill_ngl\":%d,\"prompt_token_hash\":\"%llu\",\"initialization_s\":%.9f,\"prefill_s\":%.9f,\"prefill_tok_s\":%.9f,\"ttft_s\":%.9f,\"handoff_ms\":%.9f,\"handoff_weights_ms\":%.9f,\"handoff_kv_ms\":%.9f,\"handoff_scheduler_ms\":%.9f,\"first_decode_s\":%.9f,\"decode_s\":%.9f,\"decode_tok_s\":%.9f,\"total_response_s\":%.9f,\"weight_copy_bytes\":%llu,\"cpu_shadow_weight_bytes\":%llu,\"kv_copy_bytes\":%llu,\"gpu_weight_bytes_released\":%llu,\"same_model_and_cache\":true,\"positions_preserved\":true,\"all_logits_finite\":true,\"validation_run\":%s,\"top1_agreements\":%d,\"logit_RMSE\":%.9f,\"max_abs_logit_error\":%.9f,\"mean_KL\":%.9f,\"output_tokens\":[",
            mode.c_str(),shadow.c_str(),prompt,generation,generation-1,context,llama_n_ctx(ctx),threads,batch_size,ubatch,gpu?ngl:0,
            (unsigned long long)token_hash,initialization_s,prefill_s,prompt/prefill_s,ttft_s,stats.total_ms,stats.weights_ms,stats.kv_ms,stats.scheduler_ms,
            first_decode_s,decode_s,(generation-1)/decode_s,total_s,(unsigned long long)stats.weight_copy_bytes,(unsigned long long)stats.cpu_shadow_weight_bytes,
            (unsigned long long)stats.kv_copy_bytes,(unsigned long long)stats.gpu_weight_bytes_released,validation?"true":"false",agreements,
            compared?std::sqrt(square_error/compared):0,max_error,compared?kl_sum/generation:0);
        for (size_t i=0;i<output.size();i++) printf("%s%d",i?",":"",output[i]);
        printf("]}\n");fflush(stdout);
        // RuntimeState releases all backend resources on success and cancellation.
        return 0;
    } catch (const std::exception & e) { fprintf(stderr,"HYBRID_MEASURE failed: %s\n",e.what()); int sig = stop_signal.load(std::memory_order_relaxed); return sig ? 128 + sig : 3; }
}
