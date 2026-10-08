// Local experimental probe (not part of llama.cpp upstream): exercises the
// one-way NPU->CPU in-process handoff (llama_perf_switch_to_cpu) extended in
// reports/2026-10-03/patches/npu-prefill-cpu-decode.patch + this session's
// extension (llama_memory_recurrent::perf_migrate_cpu, llama_memory_hybrid
// dispatch, LLM_ARCH_QWEN3NEXT) for Qwen3-Coder-Next's hybrid attn+recurrent
// memory. See docs/QWEN3_TRIAD_PLAN.md Phase C for context.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;
static double seconds(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

static void evaluate(llama_context * ctx, llama_batch_ext * batch,
        const llama_token * tokens, int count, int position, bool logits) {
    llama_batch_ext_clear(batch);
    for (int i = 0; i < count; ++i) {
        int idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        llama_pos pos = position + i;
        if (!llama_batch_ext_set_pos(batch, idx, &pos)) throw std::runtime_error("batch position failed");
    }
    if (logits) llama_batch_ext_set_output_logits(batch, count - 1, true);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) throw std::runtime_error("evaluation failed");
    llama_synchronize(ctx);
}

int main(int argc, char ** argv) {
    // MODEL PROMPT GEN THREADS BATCH UBATCH
    if (argc != 7) {
        fprintf(stderr, "usage: %s MODEL PROMPT_TOKENS GEN_TOKENS THREADS BATCH UBATCH\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const int prompt_n = std::stoi(argv[2]);
    const int gen_n = std::stoi(argv[3]);
    const int threads = std::stoi(argv[4]);
    const int batch_size = std::stoi(argv[5]);
    const int ubatch = std::stoi(argv[6]);

    try {
        llama_backend_init();
        auto mp = llama_model_default_params();
        ggml_backend_dev_t npu_devices[] = { ggml_backend_dev_by_name("HTP0"), nullptr };
        if (!npu_devices[0]) throw std::runtime_error("HTP0 device not found");
        mp.devices = npu_devices;
        mp.n_gpu_layers = 99;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        auto * model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!model) throw std::runtime_error("model load failed");

        auto cp = llama_context_default_params();
        cp.n_ctx = prompt_n + gen_n + 16;
        cp.n_batch = batch_size;
        cp.n_ubatch = ubatch;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.op_offload = true;
        cp.offload_kqv = true;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context allocation failed");

        auto * vocab = llama_model_get_vocab(model);
        const int nv = llama_vocab_n_tokens(vocab);
        std::string text = "The history of the French Revolution began with a severe fiscal crisis caused by France's involvement in several costly wars, including the American Revolution. ";
        while ((int) text.size() < prompt_n * 8) text += "The fiscal crisis deepened as the national debt grew and tax revenues failed to keep pace with expenditures. ";
        int nt = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
        std::vector<llama_token> tokens(nt);
        nt = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), nt, true, true);
        if (nt < prompt_n) throw std::runtime_error("insufficient tokenized prompt");
        tokens.resize(prompt_n);

        auto * batch = llama_batch_ext_init(ctx);

        fprintf(stderr, "PROBE: prefill on NPU (HTP0), %d tokens\n", prompt_n);
        auto prefill_start = clock_type::now();
        for (int pos = 0; pos < prompt_n; pos += batch_size) {
            int n = std::min(batch_size, prompt_n - pos);
            evaluate(ctx, batch, tokens.data() + pos, n, pos, pos + n == prompt_n);
        }
        const double prefill_s = seconds(prefill_start);
        fprintf(stderr, "PROBE: prefill done in %.3fs (%.2f tok/s)\n", prefill_s, prompt_n / prefill_s);

        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token selected = (llama_token)(std::max_element(logits, logits + nv) - logits);

        fprintf(stderr, "PROBE: invoking llama_perf_switch_to_cpu (NPU -> CPU, in-process, no prompt replay)\n");
        llama_perf_handoff_stats stats{};
        auto handoff_start = clock_type::now();
        int rc = llama_perf_switch_to_cpu(ctx, /*drop_gpu_weights=*/true, &stats);
        const double handoff_wall_s = seconds(handoff_start);
        if (rc != 0) throw std::runtime_error("llama_perf_switch_to_cpu failed (see stderr above)");
        fprintf(stderr, "PROBE: handoff OK in %.3fs (reported %.3fms). weights=%.2fMiB kv=%.2fMiB gpu_released=%.2fMiB\n",
                handoff_wall_s, stats.total_ms,
                stats.cpu_shadow_weight_bytes / 1048576.0, stats.kv_copy_bytes / 1048576.0,
                stats.gpu_weight_bytes_released / 1048576.0);

        fprintf(stderr, "PROBE: continuing decode on CPU, %d tokens, same context, same positions\n", gen_n);
        auto decode_start = clock_type::now();
        for (int step = 1; step < gen_n; ++step) {
            evaluate(ctx, batch, &selected, 1, prompt_n + step - 1, true);
            const float * l = llama_get_logits_ith(ctx, -1);
            selected = (llama_token)(std::max_element(l, l + nv) - l);
        }
        const double decode_s = seconds(decode_start);
        fprintf(stderr, "PROBE: decode done in %.3fs (%.2f tok/s)\n", decode_s, (gen_n - 1) / decode_s);

        printf("{\"prefill_s\":%.6f,\"prefill_tok_s\":%.6f,\"handoff_wall_s\":%.6f,\"handoff_reported_ms\":%.3f,"
               "\"kv_copy_mib\":%.3f,\"decode_s\":%.6f,\"decode_tok_s\":%.6f}\n",
               prefill_s, prompt_n / prefill_s, handoff_wall_s, stats.total_ms,
               stats.kv_copy_bytes / 1048576.0, decode_s, (gen_n - 1) / decode_s);

        llama_batch_ext_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "PROBE FAILED: %s\n", e.what());
        return 1;
    }
}
