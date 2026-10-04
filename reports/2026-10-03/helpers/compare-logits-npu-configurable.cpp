#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

static void batch_tokens(llama_batch_ext * batch, const std::vector<llama_token> & tokens, int pos) {
    llama_batch_ext_clear(batch);
    for (size_t i = 0; i < tokens.size(); ++i) {
        int idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        llama_pos p = pos + i;
        llama_batch_ext_set_pos(batch, idx, &p);
    }
    llama_batch_ext_set_output_logits(batch, tokens.size() - 1, true);
}

static std::string piece(const llama_vocab * vocab, llama_token id) {
    char buf[256];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
    return n > 0 ? std::string(buf, n) : std::string("?");
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 7) return 2;
    llama_backend_init();
    const int steps = argc >= 3 ? std::atoi(argv[2]) : 32;
    const int prompt_tokens = argc >= 4 ? std::atoi(argv[3]) : 0;
    if (prompt_tokens < 0 || prompt_tokens > 128) return 2;
    if (steps < 1 || steps > 128) return 2;
    std::vector<std::vector<float>> reference;
    std::vector<llama_token> selected;
    int agreements = 0;
    double square_error = 0, max_error = 0, total_kl = 0;
    size_t values = 0;
    for (int gpu = 0; gpu < 2; ++gpu) {
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu ? (argc >= 5 ? std::atoi(argv[4]) : 99) : 0;
        mp.use_extra_bufts = argc < 6 || std::atoi(argv[5]) != 0;
        ggml_backend_dev_t cpu_devices[] = { nullptr };
        ggml_backend_dev_t htp_devices[] = {ggml_backend_dev_by_name("HTP0"), nullptr};
        if (!htp_devices[0]) htp_devices[0] = ggml_backend_dev_by_name("HTP0:0");
        if (gpu && !htp_devices[0]) return 8;
        mp.devices = gpu ? htp_devices : cpu_devices;
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) return 3;
        auto * vocab = llama_model_get_vocab(model);
        int nv = llama_vocab_n_tokens(vocab);
        std::string prompt = "def add(a, b):";
        if (prompt_tokens) {
            while (int(prompt.size()) < prompt_tokens*16) prompt += "\ndef add(a, b): return a + b\n";
        }
        int nt = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
        std::vector<llama_token> tokens(nt);
        if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), nt, true, true) < 0) return 4;
        if (prompt_tokens) {
            if (int(tokens.size()) < prompt_tokens) return 4;
            tokens.resize(prompt_tokens);
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = 1024;
        cp.n_batch = cp.n_ubatch = 128;
        cp.n_threads = cp.n_threads_batch = 4;
        cp.flash_attn_type = argc >= 7 && std::string(argv[6]) == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.op_offload = gpu;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) return 5;
        auto * batch = llama_batch_ext_init(ctx);
        int pos = 0;
        for (int step = 0; step < steps; ++step) {
            batch_tokens(batch, tokens, pos);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) return 6;
            pos += tokens.size();
            const float * logits = llama_get_logits_ith(ctx, -1);
            int best = std::max_element(logits, logits + nv) - logits;
            if (!gpu) {
                reference.emplace_back(logits, logits + nv);
                selected.push_back(best);
            } else {
                const auto & ref = reference[step];
                int cpu_best = selected[step];
                agreements += best == cpu_best;
                double cpu_max = ref[cpu_best], gpu_max = logits[best];
                double cpu_sum = 0, gpu_sum = 0;
                for (int i = 0; i < nv; ++i) {
                    if (!std::isfinite(logits[i]) || !std::isfinite(ref[i])) return 7;
                    double diff = double(logits[i]) - ref[i];
                    square_error += diff * diff;
                    max_error = std::max(max_error, std::abs(diff));
                    cpu_sum += std::exp(double(ref[i]) - cpu_max);
                    gpu_sum += std::exp(double(logits[i]) - gpu_max);
                }
                double cpu_lse = cpu_max + std::log(cpu_sum);
                double gpu_lse = gpu_max + std::log(gpu_sum);
                double kl = 0;
                for (int i = 0; i < nv; ++i) {
                    kl += std::exp(double(ref[i]) - cpu_lse) * (ref[i] - cpu_lse - logits[i] + gpu_lse);
                }
                total_kl += kl;
                values += nv;
                if (best != cpu_best) {
                    printf("step=%d CPU_token=%d HTP_token=%d CPU_logits=(%.6f,%.6f) HTP_logits=(%.6f,%.6f) KL=%.8f pieces=(%s,%s)\n",
                        step, cpu_best, best, ref[cpu_best], ref[best], logits[cpu_best], logits[best], kl,
                        piece(vocab, cpu_best).c_str(), piece(vocab, best).c_str());
                }
            }
            tokens = { selected[step] };
        }
        llama_batch_ext_free(batch);
        llama_free(ctx);
        llama_model_free(model);
    }
    printf("steps=%d top1_agreements=%d logit_RMSE=%.8f max_abs_logit_error=%.8f mean_KL=%.8f all_logits_finite=true\n",
        steps, agreements, std::sqrt(square_error / values), max_error, total_kl / steps);
    llama_backend_free();
}
