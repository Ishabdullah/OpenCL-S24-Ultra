// Repeated-prompt routing measurement; no weights are removed or cached.
// stdout is JSON Lines: one complete record per trial, flushed immediately.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Counts = std::map<int, std::map<int, size_t>>;
static Counts selections;
static int expert_count;
static bool invalid_capture = false;
static bool capture(struct ggml_tensor * t, bool ask, void *) {
    if (ask) return strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    int layer = -1;
    if (sscanf(t->name, "ffn_moe_topk-%d", &layer) != 1 || layer < 0 || t->type != GGML_TYPE_I32) {
        invalid_capture = true;
        return false;
    }
    std::vector<int32_t> ids(ggml_nelements(t));
    ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));
    for (int id : ids) {
        if (id < 0 || id >= expert_count) { invalid_capture = true; return false; }
        ++selections[layer][id];
    }
    return true;
}
static std::string quote(const std::string & value) {
    std::string out = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c < 32) { char b[7]; snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
        else out += c;
    }
    return out + '"';
}
static int positive(const char * text) {
    size_t used = 0;
    int result = std::stoi(text, &used);
    if (used != strlen(text) || result <= 0) throw std::runtime_error("arguments must be positive integers");
    return result;
}
static std::string meta(llama_model * model, const std::string & key) {
    char value[256];
    if (llama_model_meta_val_str(model, key.c_str(), value, sizeof(value)) < 0)
        throw std::runtime_error("missing metadata: " + key);
    return value;
}
static void evaluate(llama_context * ctx, llama_batch_ext * batch,
                     const llama_token * tokens, int count, int position, bool logits) {
    llama_batch_ext_clear(batch);
    for (int i = 0; i < count; ++i) {
        int index = llama_batch_ext_add_token(batch, 0, tokens[i]);
        llama_pos pos = position + i;
        if (!llama_batch_ext_set_pos(batch, index, &pos)) throw std::runtime_error("batch position failed");
    }
    if (logits) llama_batch_ext_set_output_logits(batch, count - 1, true);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) throw std::runtime_error("evaluation failed");
    llama_synchronize(ctx);
    if (invalid_capture) throw std::runtime_error("invalid router capture");
}
static void counts_json(const Counts & counts) {
    printf("{"); bool first_layer = true;
    for (const auto & [layer, ids] : counts) {
        printf("%s\"%d\":{", first_layer ? "" : ",", layer); first_layer = false;
        bool first = true;
        for (const auto & [id, count] : ids) { printf("%s\"%d\":%zu", first ? "" : ",", id, count); first = false; }
        printf("}");
    }
    printf("}");
}
static unsigned long long storage_reads() {
    std::ifstream file("/proc/self/io"); std::string key; unsigned long long value;
    while (file >> key >> value) if (key == "read_bytes:") return value;
    return 0;
}
int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s MODEL GEN_TOKENS N_TRIALS THREADS PROMPT_TEXT\n", argv[0]); return 2;
    }
    try {
        const int gen_n = positive(argv[2]), trials = positive(argv[3]), threads = positive(argv[4]);
        if (gen_n > 4096 || trials > 1000 || threads > 64) throw std::runtime_error("arguments exceed probe limits");
        const std::string prompt_text = argv[5];
        if (prompt_text.empty()) throw std::runtime_error("empty prompt");
        llama_backend_init();
        auto mp = llama_model_default_params();
        ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name("HTP0"), nullptr};
        if (!devices[0]) throw std::runtime_error("HTP0 device not found");
        mp.devices = devices; mp.n_gpu_layers = 99; mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        const std::string arch = meta(model.get(), "general.architecture");
        expert_count = positive(meta(model.get(), arch + ".expert_count").c_str());
        const int used_experts = positive(meta(model.get(), arch + ".expert_used_count").c_str());
        auto * vocab = llama_model_get_vocab(model.get());
        int nt = -llama_tokenize(vocab, prompt_text.data(), prompt_text.size(), nullptr, 0, true, true);
        if (nt <= 0 || nt > 4096) throw std::runtime_error("invalid prompt length");
        std::vector<llama_token> prompt(nt);
        nt = llama_tokenize(vocab, prompt_text.data(), prompt_text.size(), prompt.data(), nt, true, true);
        if (nt <= 0) throw std::runtime_error("tokenization failed");
        prompt.resize(nt);
        auto cp = llama_context_default_params();
        cp.n_ctx = nt + gen_n + 16; cp.n_batch = cp.n_ubatch = 128;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.op_offload = cp.offload_kqv = true; cp.cb_eval = capture;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("context allocation failed");
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        if (!batch) throw std::runtime_error("batch allocation failed");
        std::map<int, std::set<int>> accumulated;
        printf("{\"type\":\"config\",\"model\":%s,\"architecture\":%s,\"prompt\":%s,\"prompt_tokens\":%d,\"gen_limit\":%d,\"trials\":%d,\"threads\":%d,\"experts\":%d,\"experts_used\":%d,\"temperature\":0.8,\"top_k\":40,\"repeat_penalty\":1.1}\n",
            quote(argv[1]).c_str(), quote(arch).c_str(), quote(prompt_text).c_str(), nt, gen_n, trials, threads, expert_count, used_experts);
        fflush(stdout);
        for (int trial = 0; trial < trials; ++trial) {
            selections.clear(); llama_memory_clear(llama_get_memory(ctx.get()), true);
            std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
                llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(40));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 64, 1.1f, 0, 0));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(0.8f));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(1000u + trial));
            for (auto token : prompt) llama_sampler_accept(sampler.get(), token);
            auto start = std::chrono::steady_clock::now(); const auto reads_before = storage_reads();
            for (int pos = 0; pos < nt; pos += 128)
                evaluate(ctx.get(), batch.get(), prompt.data() + pos, std::min(128, nt - pos), pos, pos + 128 >= nt);
            Counts prefill = selections; selections.clear();
            std::string text; std::vector<llama_token> generated; bool eog = false;
            for (int step = 0; step < gen_n; ++step) {
                auto token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
                llama_sampler_accept(sampler.get(), token); generated.push_back(token);
                if (llama_vocab_is_eog(vocab, token)) { eog = true; break; }
                std::vector<char> piece(128);
                int n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
                if (n < 0) { piece.resize(-n); n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true); }
                if (n < 0) throw std::runtime_error("token piece conversion failed");
                text.append(piece.data(), n);
                // Last output token has no forward evaluation, as in normal generation.
                if (step + 1 < gen_n) evaluate(ctx.get(), batch.get(), &token, 1, nt + step, true);
            }
            if (prefill.empty()) throw std::runtime_error("no router tensors captured");
            size_t added = 0, total = 0;
            for (const Counts * phase : {&prefill, &selections})
                for (const auto & [layer, ids] : *phase)
                    for (const auto & [id, count] : ids) added += accumulated[layer].insert(id).second;
            for (const auto & [layer, ids] : accumulated) total += ids.size();
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            printf("{\"type\":\"trial\",\"trial\":%d,\"seed\":%u,\"elapsed_s\":%.6f,\"storage_read_bytes\":%llu,\"generated_tokens\":%zu,\"eog\":%s,\"text\":%s,\"new_experts_total\":%zu,\"union_mean\":%.6f,\"prefill\":",
                trial + 1, 1000u + trial, elapsed, storage_reads() - reads_before, generated.size(), eog ? "true" : "false", quote(text).c_str(), added, double(total) / accumulated.size());
            counts_json(prefill); printf(",\"decode\":"); counts_json(selections); printf("}\n"); fflush(stdout);
            fprintf(stderr, "MAP trial=%d/%d tokens=%zu layers=%zu mean_union=%.1f/%d new=%zu elapsed=%.1fs\n", trial + 1, trials, generated.size(), accumulated.size(), double(total)/accumulated.size(), expert_count, added, elapsed);
        }
        batch.reset(); ctx.reset(); model.reset(); llama_backend_free(); return 0;
    } catch (const std::exception & e) { fprintf(stderr, "MAP FAILED: %s\n", e.what()); return 1; }
}
