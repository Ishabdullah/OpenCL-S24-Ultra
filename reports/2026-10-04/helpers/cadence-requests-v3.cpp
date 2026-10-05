#include "llama.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <glob.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using steady_clock = std::chrono::steady_clock;
static std::atomic<int> stop_signal{0};
static_assert(std::atomic<int>::is_always_lock_free, "Signal cancellation requires lock-free atomics");
static void request_stop(int signal) { stop_signal.store(signal, std::memory_order_relaxed); }
static void check_stop() {
    if (stop_signal.load(std::memory_order_relaxed)) throw std::runtime_error("measurement interrupted; releasing backend resources");
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
        fprintf(stderr, "SUSTAINED cleanup complete\n");
    }
};
static double elapsed(steady_clock::time_point start) {
    return std::chrono::duration<double>(steady_clock::now() - start).count();
}
static double monotonic_seconds() {
    return std::chrono::duration<double>(steady_clock::now().time_since_epoch()).count();
}
struct ThermalAdmission { double waited_s=0,cpu=0,npu=0,gpu=0,battery=0; };
static ThermalAdmission postload_admission() {
    const auto start=steady_clock::now();
    double ready_since=-1;
    glob_t paths{};
    if (glob("/sys/class/thermal/thermal_zone*/type",0,nullptr,&paths))
        throw std::runtime_error("post-load thermal sensor discovery failed");
    std::vector<std::pair<std::string,std::string>> sensors;
    for (size_t i=0;i<paths.gl_pathc;i++) {
        std::string path=paths.gl_pathv[i],name;
        std::ifstream f(path); std::getline(f,name);
        if (name.rfind("cpu-",0)==0 || name.rfind("nsphmx-",0)==0 || name.rfind("nsphvx-",0)==0 ||
                name.rfind("gpuss-",0)==0 || name=="battery")
            sensors.emplace_back(name,path.substr(0,path.size()-4)+"temp");
    }
    globfree(&paths);
    for (;;) {
        check_stop();
        ThermalAdmission r; int nc=0,nn=0,ng=0,nb=0;
        for (const auto & sensor:sensors) {
            int value=0;std::ifstream f(sensor.second);
            if (!(f>>value) || value < -100000 || value > 200000)
                throw std::runtime_error("post-load thermal sensor unavailable: "+sensor.first);
            const double t=value/1000.0;
            if (sensor.first.rfind("cpu-",0)==0) {r.cpu=std::max(r.cpu,t);nc++;}
            else if (sensor.first.rfind("nsp",0)==0) {r.npu=std::max(r.npu,t);nn++;}
            else if (sensor.first.rfind("gpuss-",0)==0) {r.gpu=std::max(r.gpu,t);ng++;}
            else {r.battery=t;nb++;}
        }
        if (!nc || !nn || !ng || nb!=1) throw std::runtime_error("post-load thermal sensors incomplete");
        std::ifstream mem("/proc/meminfo");std::string line;unsigned long long avail=0;
        while (std::getline(mem,line)) if (sscanf(line.c_str(),"MemAvailable: %llu kB",&avail)==1) break;
        if (avail < 1536ULL*1024) throw std::runtime_error("post-load memory reserve reached");
        r.waited_s=elapsed(start);
        const bool ready=r.cpu<=45 && r.npu<=43 && r.gpu<=43 && r.battery<=36;
        if (ready) {if (ready_since<0) ready_since=r.waited_s;}
        else ready_since=-1;
        fprintf(stderr,"PHASE_POSTLOAD_ADMISSION: waited=%.3f CPU=%.1f NPU=%.1f GPU=%.1f battery=%.1f ready=%d\n",
                r.waited_s,r.cpu,r.npu,r.gpu,r.battery,int(ready));fflush(stderr);
        if (ready_since>=0 && r.waited_s-ready_since>=10) return r;
        if (r.waited_s>600) throw std::runtime_error("post-load cold admission timeout; no timed response");
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}
static void evaluate(llama_context *ctx, llama_batch_ext *batch, const llama_token *tokens,
                     int count, int position, bool output) {
    check_stop();
    llama_batch_ext_clear(batch);
    for (int i = 0; i < count; ++i) {
        int idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        llama_pos pos = position + i;
        if (!llama_batch_ext_set_pos(batch, idx, &pos)) throw std::runtime_error("position assignment failed");
    }
    if (output) llama_batch_ext_set_output_logits(batch, count - 1, true);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) throw std::runtime_error("evaluation failed");
    llama_synchronize(ctx);
}
int main(int argc, char **argv) {
    // MODEL DEVICE NGL THREADS PROMPT GEN CONTEXT BATCH UBATCH SECONDS FA
    if (argc != 12) return 2;
    std::signal(SIGTERM, request_stop);
    std::signal(SIGINT, request_stop);
    try {
        std::string device = argv[2], fa = argv[11];
        int ngl = std::stoi(argv[3]), threads = std::stoi(argv[4]);
        int prompt = std::stoi(argv[5]), generation = std::stoi(argv[6]), context = std::stoi(argv[7]);
        int batch_size = std::stoi(argv[8]), ubatch = std::stoi(argv[9]), duration = std::stoi(argv[10]);
        if (threads < 1 || prompt < 1 || generation < 2 || context < prompt + generation ||
            batch_size < 1 || ubatch < 1 || duration < 1 || (fa != "off" && fa != "on")) return 2;
        double period=std::getenv("GGML_PERF_REQUEST_PERIOD_S")?std::stod(std::getenv("GGML_PERF_REQUEST_PERIOD_S")):0;
        bool repack=!std::getenv("GGML_PERF_CPU_REPACK")||std::atoi(std::getenv("GGML_PERF_CPU_REPACK"))!=0;
        if(!std::isfinite(period)||period<0||period>3600) return 2;
        RuntimeState resources;
        auto mp = llama_model_default_params();
        ggml_backend_dev_t devices[] = {device == "none" ? nullptr : ggml_backend_dev_by_name(device.c_str()), nullptr};
        if (device != "none" && !devices[0]) throw std::runtime_error("requested device missing");
        mp.devices = devices;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        mp.use_extra_bufts = repack;
        mp.n_gpu_layers = device == "none" ? 0 : ngl;
        auto *model = llama_model_load_from_file(argv[1], mp);
        resources.model = model;
        if (!model) throw std::runtime_error("model allocation failed");
        check_stop();
        auto cp = llama_context_default_params();
        cp.n_ctx = context; cp.n_batch = batch_size; cp.n_ubatch = ubatch;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.op_offload = cp.offload_kqv = device != "none";
        auto *ctx = llama_init_from_model(model, cp);
        resources.ctx = ctx;
        if (!ctx) throw std::runtime_error("context allocation failed");
        auto tp = ggml_threadpool_params_default(threads);
        tp.poll = 0;
        auto *pool = ggml_threadpool_new(&tp);
        resources.pool = pool;
        if (!pool) throw std::runtime_error("threadpool allocation failed");
        llama_attach_threadpool(ctx, pool, pool);
        const auto *vocab = llama_model_get_vocab(model);
        int nv = llama_vocab_n_tokens(vocab);
        std::string text = "Write clear Python code and explain its behavior.\n";
        while (int(text.size()) < prompt * 16) text += "def add(a, b):\n    return a + b\n# Explain the result carefully.\n";
        int nt = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
        std::vector<llama_token> tokens(nt);
        nt = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), nt, true, true);
        if (nt < prompt) throw std::runtime_error("prompt tokenization failed");
        tokens.resize(prompt);
        uint64_t hash = 1469598103934665603ULL;
        for (auto token : tokens) { hash ^= uint32_t(token); hash *= 1099511628211ULL; }
        auto *batch = llama_batch_ext_init(ctx);
        resources.batch = batch;
        auto choose = [&]() {
            const float *logits = llama_get_logits_ith(ctx, -1);
            for (int i = 0; i < nv; ++i) if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits");
            return llama_token(std::max_element(logits, logits + nv) - logits);
        };
        ThermalAdmission ready;
        auto request = [&](int iteration, double scheduled = -1) {
            double service_start_mono=monotonic_seconds();
            auto reset_start = steady_clock::now();
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_synchronize(ctx);
            double reset_s = elapsed(reset_start);
            auto start = steady_clock::now(); double start_mono = monotonic_seconds();
            for (int pos = 0; pos < prompt; pos += batch_size) {
                int n = std::min(batch_size, prompt - pos);
                evaluate(ctx, batch, tokens.data() + pos, n, pos, pos + n == prompt);
            }
            double pp_s = elapsed(start);
            llama_token selected = choose();
            double ttft_s = elapsed(start);
            auto decode_start = steady_clock::now();
            uint64_t output_hash = uint32_t(selected);
            for (int step = 1; step < generation; ++step) {
                evaluate(ctx, batch, &selected, 1, prompt + step - 1, true);
                selected = choose();
                output_hash = (output_hash ^ uint32_t(selected)) * 1099511628211ULL;
            }
            double decode_s = elapsed(decode_start), total_s = elapsed(start);
            double service_s=reset_s+total_s;
            double queue_delay_s=scheduled<0?0:std::max(0.0,service_start_mono-scheduled);
            double end_to_end_s=scheduled<0?service_s:monotonic_seconds()-scheduled;
            bool missed_deadline=scheduled>=0&&period>0&&end_to_end_s>period;
            printf("{\"request_period_s\":%.9f,\"cpu_repack\":%s,\"scheduled_arrival_monotonic_s\":%.9f,\"service_start_monotonic_s\":%.9f,\"queue_delay_s\":%.9f,\"service_s\":%.9f,\"arrival_to_completion_s\":%.9f,\"missed_deadline\":%s,\"postload_admission_s\":%.9f,\"initial_CPU_max_C\":%.1f,\"initial_NPU_max_C\":%.1f,\"initial_GPU_max_C\":%.1f,\"initial_battery_C\":%.1f,",period,repack?"true":"false",scheduled,service_start_mono,queue_delay_s,service_s,end_to_end_s,missed_deadline?"true":"false",ready.waited_s,ready.cpu,ready.npu,ready.gpu,ready.battery);
            printf("\"iteration\":%d,\"monotonic_s\":%.9f,\"prompt\":%d,\"generation\":%d,\"decode_steps\":%d,\"context\":%u,\"threads\":%d,\"batch\":%d,\"ubatch\":%d,\"attention\":\"%s\",\"device\":\"%s\",\"ngl\":%d,\"prompt_hash\":\"%llu\",\"output_hash\":\"%llu\",\"reset_s\":%.9f,\"prefill_s\":%.9f,\"prefill_tok_s\":%.9f,\"ttft_s\":%.9f,\"decode_s\":%.9f,\"decode_tok_s\":%.9f,\"total_response_s\":%.9f,\"all_logits_finite\":true}\n",
                   iteration, start_mono, prompt, generation, generation - 1, llama_n_ctx(ctx), threads,
                   batch_size, ubatch, fa.c_str(), device.c_str(), mp.n_gpu_layers,
                   (unsigned long long) hash, (unsigned long long) output_hash, reset_s, pp_s,
                   prompt / pp_s, ttft_s, decode_s, (generation - 1) / decode_s, total_s);
            fflush(stdout);
        };
        request(-1);
        for (int i = 0; i < 100; ++i) {
            check_stop();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ready=postload_admission();
        auto sustained_start = steady_clock::now();
        double arrival_origin=monotonic_seconds();
        fprintf(stderr,"CADENCE protocol period_s=%.3f offered_duration_s=%d postload_wait_s=%.3f CPU=%.1f NPU=%.1f GPU=%.1f battery=%.1f\n",period,duration,ready.waited_s,ready.cpu,ready.npu,ready.gpu,ready.battery);
        if(period>0){
            int arrivals=int(std::ceil(duration/period));
            for(int iteration=0;iteration<arrivals;iteration++){
                double arrival=arrival_origin+iteration*period;
                while(monotonic_seconds()<arrival){check_stop();std::this_thread::sleep_for(std::chrono::milliseconds(50));}
                request(iteration,arrival);
            }
            // Keep the session alive through the complete offered-work interval.
            while(elapsed(sustained_start)<duration){check_stop();std::this_thread::sleep_for(std::chrono::milliseconds(50));}
            fprintf(stderr,"CADENCE complete offered=%d served=%d duration_s=%.3f\n",arrivals,arrivals,elapsed(sustained_start));
        }else{
            for(int iteration=0;elapsed(sustained_start)<duration;iteration++)request(iteration);
        }
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "SUSTAINED failed: %s\n", e.what());
        int interrupted = stop_signal.load(std::memory_order_relaxed);
        return interrupted ? 128 + interrupted : 3;
    }
}
