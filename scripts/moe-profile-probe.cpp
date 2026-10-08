// Local experimental probe (not part of llama.cpp upstream): profiles an
// NPU-resident MoE run before any expert-caching work, per the user's
// explicit request -- "profile your existing run before changing it:
// record expert selections, cache misses, storage reads, NPU transfers,
// and draft acceptance." This probe covers expert selections (via the
// existing common_debug_cb_user_data hook on the "ffn_moe_topk" tensor,
// which holds the router's chosen expert indices) plus cache misses and
// storage reads (via /proc/self/stat minflt/majflt and /proc/self/io
// rchar/read_bytes, sampled at phase boundaries). No MTP/speculative
// decoding here by design -- draft acceptance is measured separately via
// llama-server, which already has that instrumentation built in.
// See docs/QWEN3_TRIAD_PLAN.md Phase F for context.
#include "llama.h"
#include "common.h"
#include "debug.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;
static double seconds(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

struct ProcSnapshot {
    unsigned long long minflt = 0, majflt = 0;
    unsigned long long rchar = 0, read_bytes = 0;
    long vmrss_kb = 0;
};

static ProcSnapshot read_proc_snapshot() {
    ProcSnapshot s;
    {
        std::ifstream f("/proc/self/stat");
        std::string line;
        std::getline(f, line);
        // skip "pid (comm) state ppid pgrp session tty_nr tpgid flags" then minflt cminflt majflt cmajflt
        size_t close_paren = line.rfind(')');
        std::istringstream iss(line.substr(close_paren + 2));
        std::string state; long ppid, pgrp, session, tty_nr, tpgid; unsigned long flags;
        iss >> state >> ppid >> pgrp >> session >> tty_nr >> tpgid >> flags >> s.minflt;
        unsigned long long cminflt;
        iss >> cminflt >> s.majflt;
    }
    {
        std::ifstream f("/proc/self/io");
        std::string key; unsigned long long val;
        while (f >> key >> val) {
            if (key == "rchar:") s.rchar = val;
            else if (key == "read_bytes:") s.read_bytes = val;
        }
    }
    {
        std::ifstream f("/proc/self/status");
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("VmRSS:", 0) == 0) {
                sscanf(line.c_str(), "VmRSS: %ld kB", &s.vmrss_kb);
                break;
            }
        }
    }
    return s;
}

static void print_snapshot(const char * label, const ProcSnapshot & s) {
    fprintf(stderr, "PROC_SNAPSHOT %-10s minflt=%llu majflt=%llu rchar_mib=%.2f read_bytes_mib=%.2f vmrss_mib=%.2f\n",
            label, s.minflt, s.majflt, s.rchar / 1048576.0, s.read_bytes / 1048576.0, s.vmrss_kb / 1024.0);
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
        print_snapshot("start", read_proc_snapshot());

        auto mp = llama_model_default_params();
        ggml_backend_dev_t npu_devices[] = { ggml_backend_dev_by_name("HTP0"), nullptr };
        if (!npu_devices[0]) throw std::runtime_error("HTP0 device not found");
        mp.devices = npu_devices;
        mp.n_gpu_layers = 99;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        auto load_start = clock_type::now();
        auto * model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!model) throw std::runtime_error("model load failed");
        fprintf(stderr, "PROBE: model loaded in %.3fs\n", seconds(load_start));
        print_snapshot("post_load", read_proc_snapshot());

        auto cp = llama_context_default_params();
        cp.n_ctx = prompt_n + gen_n + 16;
        cp.n_batch = batch_size;
        cp.n_ubatch = ubatch;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.op_offload = true;
        cp.offload_kqv = true;

        // Wire the expert-selection tensor dump (filters to the router's
        // top-k output; see src/llama-graph.cpp cb(selected_experts, "ffn_moe_topk", il)).
        common_params dbg_params;
        common_debug_cb_user_data dbg_cb(dbg_params, {"ffn_moe_topk"}, /*abort_on_nan=*/false);
        cp.cb_eval = dbg_params.cb_eval;
        cp.cb_eval_user_data = dbg_params.cb_eval_user_data;

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

        fprintf(stderr, "PROBE: === PREFILL PHASE (%d tokens) ===\n", prompt_n);
        auto prefill_start = clock_type::now();
        for (int pos = 0; pos < prompt_n; pos += batch_size) {
            int n = std::min(batch_size, prompt_n - pos);
            evaluate(ctx, batch, tokens.data() + pos, n, pos, pos + n == prompt_n);
        }
        const double prefill_s = seconds(prefill_start);
        fprintf(stderr, "PROBE: prefill done in %.3fs (%.2f tok/s)\n", prefill_s, prompt_n / prefill_s);
        print_snapshot("post_prefill", read_proc_snapshot());

        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token selected = (llama_token)(std::max_element(logits, logits + nv) - logits);

        fprintf(stderr, "PROBE: === DECODE PHASE (%d tokens) ===\n", gen_n);
        auto decode_start = clock_type::now();
        for (int step = 1; step < gen_n; ++step) {
            evaluate(ctx, batch, &selected, 1, prompt_n + step - 1, true);
            const float * l = llama_get_logits_ith(ctx, -1);
            selected = (llama_token)(std::max_element(l, l + nv) - l);
        }
        const double decode_s = seconds(decode_start);
        fprintf(stderr, "PROBE: decode done in %.3fs (%.2f tok/s)\n", decode_s, (gen_n - 1) / decode_s);
        print_snapshot("post_decode", read_proc_snapshot());

        printf("{\"prefill_s\":%.6f,\"prefill_tok_s\":%.6f,\"decode_s\":%.6f,\"decode_tok_s\":%.6f}\n",
               prefill_s, prompt_n / prefill_s, decode_s, (gen_n - 1) / decode_s);

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
