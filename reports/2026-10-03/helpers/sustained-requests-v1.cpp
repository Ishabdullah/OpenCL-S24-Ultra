#include "llama.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using steady_clock = std::chrono::steady_clock;
static double elapsed(steady_clock::time_point start) {
    return std::chrono::duration<double>(steady_clock::now() - start).count();
}
static double monotonic_seconds() {
    return std::chrono::duration<double>(steady_clock::now().time_since_epoch()).count();
}
static void evaluate(llama_context *ctx, llama_batch_ext *batch, const llama_token *tokens,
                     int count, int position, bool output) {
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
    try {
        std::string device = argv[2], fa = argv[11];
        int ngl = std::stoi(argv[3]), threads = std::stoi(argv[4]);
        int prompt = std::stoi(argv[5]), generation = std::stoi(argv[6]), context = std::stoi(argv[7]);
        int batch_size = std::stoi(argv[8]), ubatch = std::stoi(argv[9]), duration = std::stoi(argv[10]);
        if (threads < 1 || prompt < 1 || generation < 2 || context < prompt + generation ||
            batch_size < 1 || ubatch < 1 || duration < 1 || (fa != "off" && fa != "on")) return 2;
        llama_backend_init();
        auto mp = llama_model_default_params();
        ggml_backend_dev_t devices[] = {device == "none" ? nullptr : ggml_backend_dev_by_name(device.c_str()), nullptr};
        if (device != "none" && !devices[0]) throw std::runtime_error("requested device missing");
        mp.devices = devices;
        mp.n_gpu_layers = device == "none" ? 0 : ngl;
        auto *model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("model allocation failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = context; cp.n_batch = batch_size; cp.n_ubatch = ubatch;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.op_offload = cp.offload_kqv = device != "none";
        auto *ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context allocation failed");
        auto tp = ggml_threadpool_params_default(threads);
        tp.poll = 0;
        auto *pool = ggml_threadpool_new(&tp);
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
        auto choose = [&]() {
            const float *logits = llama_get_logits_ith(ctx, -1);
            for (int i = 0; i < nv; ++i) if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits");
            return llama_token(std::max_element(logits, logits + nv) - logits);
        };
        auto request = [&](int iteration) {
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
            printf("{\"iteration\":%d,\"monotonic_s\":%.9f,\"prompt\":%d,\"generation\":%d,\"decode_steps\":%d,\"context\":%u,\"threads\":%d,\"batch\":%d,\"ubatch\":%d,\"attention\":\"%s\",\"device\":\"%s\",\"ngl\":%d,\"prompt_hash\":\"%llu\",\"output_hash\":\"%llu\",\"reset_s\":%.9f,\"prefill_s\":%.9f,\"prefill_tok_s\":%.9f,\"ttft_s\":%.9f,\"decode_s\":%.9f,\"decode_tok_s\":%.9f,\"total_response_s\":%.9f,\"all_logits_finite\":true}\n",
                   iteration, start_mono, prompt, generation, generation - 1, llama_n_ctx(ctx), threads,
                   batch_size, ubatch, fa.c_str(), device.c_str(), mp.n_gpu_layers,
                   (unsigned long long) hash, (unsigned long long) output_hash, reset_s, pp_s,
                   prompt / pp_s, ttft_s, decode_s, (generation - 1) / decode_s, total_s);
            fflush(stdout);
        };
        request(-1);
        std::this_thread::sleep_for(std::chrono::seconds(10));
        auto sustained_start = steady_clock::now();
        for (int iteration = 0; elapsed(sustained_start) < duration; ++iteration) request(iteration);
        llama_batch_ext_free(batch);
        llama_detach_threadpool(ctx); ggml_threadpool_free(pool);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "SUSTAINED failed: %s\n", e.what());
        return 3;
    }
}
