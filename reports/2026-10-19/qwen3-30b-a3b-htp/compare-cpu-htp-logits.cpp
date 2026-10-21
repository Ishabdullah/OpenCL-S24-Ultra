#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void filtered_route_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_DEBUG) {
        if (std::strstr(text, "execute-op ")) std::fputs(text, stderr);
        return;
    }
    std::fputs(text, stderr);
}

static void evaluate(llama_context * ctx, llama_batch_ext * batch,
                     const std::vector<llama_token> & tokens, int pos) {
    llama_batch_ext_clear(batch);
    for (size_t i = 0; i < tokens.size(); ++i) {
        int index = llama_batch_ext_add_token(batch, 0, tokens[i]);
        llama_pos token_pos = pos + static_cast<int>(i);
        if (!llama_batch_ext_set_pos(batch, index, &token_pos)) {
            throw std::runtime_error("failed to set token position");
        }
    }
    llama_batch_ext_set_output_logits(batch, tokens.size() - 1, true);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
        throw std::runtime_error("llama_process failed");
    }
    llama_synchronize(ctx);
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL STEPS(1..16)\n", argv[0]);
        return 2;
    }
    try {
        const int steps = std::stoi(argv[2]);
        if (steps < 1 || steps > 16) throw std::runtime_error("steps must be 1..16");
        const std::string prompt = "Hi";
        std::vector<std::vector<float>> reference;
        std::vector<llama_token> teacher;
        int top1_agreements = 0;
        int exact_vectors = 0;
        size_t values = 0;
        double sum_sq = 0.0;
        double max_abs = 0.0;
        double max_relative = 0.0;
        double total_kl = 0.0;
        int prompt_token_count = 0;
        if (std::getenv("GGML_HEXAGON_OPFILTER")) {
            llama_log_set(filtered_route_log, nullptr);
            std::fprintf(stderr, "MODEL_EXPERT_FILTER regex=%s\n",
                         std::getenv("GGML_HEXAGON_OPFILTER"));
        }
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
        model_params.use_extra_bufts = false;
        for (int pass = 0; pass < 2; ++pass) {
            ggml_backend_dev_t cpu_devices[] = { nullptr };
            ggml_backend_dev_t htp_devices[] = {
                ggml_backend_dev_by_name("HTP0"), nullptr
            };
            if (pass && !htp_devices[0]) throw std::runtime_error("HTP0 not registered");
            model_params.devices = pass ? htp_devices : cpu_devices;
            // Keep the first model-level HTP validation bounded to one
            // transformer layer. The rest stays on CPU and uses mmap-backed
            // tensors just as the admission smoke does.
            model_params.n_gpu_layers = pass ? 2 : 0;
            std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
                llama_model_load_from_file(argv[1], model_params), llama_model_free);
            if (!model) throw std::runtime_error(pass ? "HTP model load failed" : "CPU model load failed");
            const llama_vocab * vocab = llama_model_get_vocab(model.get());
            const int vocab_size = llama_vocab_n_tokens(vocab);
            int token_count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
            if (token_count <= 0 || token_count > 16) throw std::runtime_error("invalid prompt token count");
            prompt_token_count = token_count;
            std::vector<llama_token> prompt_tokens(token_count);
            if (llama_tokenize(vocab, prompt.data(), prompt.size(), prompt_tokens.data(), token_count, true, true) != token_count) {
                throw std::runtime_error("prompt tokenization failed");
            }
            auto context_params = llama_context_default_params();
            context_params.n_ctx = 64;
            context_params.n_batch = 1;
            context_params.n_ubatch = 1;
            context_params.n_threads = 1;
            context_params.n_threads_batch = 1;
            context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            context_params.op_offload = pass != 0;
            // Some attribution modes keep attention/KV cache on CPU. This
            // avoids pre-placing KV tensors in HTP when their compute ops
            // are sent back to CPU by the op filter.
            context_params.offload_kqv = pass == 0 ||
                std::getenv("GGML_HEXAGON_CPU_KV") == nullptr;
            std::unique_ptr<llama_context, decltype(&llama_free)> context(
                llama_init_from_model(model.get(), context_params), llama_free);
            if (!context) throw std::runtime_error("context creation failed");
            std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
                llama_batch_ext_init(context.get()), llama_batch_ext_free);
            if (!batch) throw std::runtime_error("batch allocation failed");
            int position = 0;
            for (int step = 0; step < steps; ++step) {
                const auto & input = step == 0 ? prompt_tokens : std::vector<llama_token>{teacher[step - 1]};
                evaluate(context.get(), batch.get(), input, position);
                position += static_cast<int>(input.size());
                const float * logits = llama_get_logits_ith(context.get(), -1);
                if (!logits) throw std::runtime_error("logits unavailable");
                for (int i = 0; i < vocab_size; ++i) {
                    if (!std::isfinite(logits[i])) throw std::runtime_error("non-finite logits");
                }
                const llama_token best = static_cast<llama_token>(
                    std::max_element(logits, logits + vocab_size) - logits);
                if (!pass) {
                    reference.emplace_back(logits, logits + vocab_size);
                    teacher.push_back(best);
                } else {
                    const auto & cpu_logits = reference[step];
                    int cpu_best = static_cast<int>(std::max_element(
                        cpu_logits.begin(), cpu_logits.end()) - cpu_logits.begin());
                    top1_agreements += best == cpu_best;
                    const double cpu_max = cpu_logits[cpu_best];
                    const double htp_max = logits[best];
                    double cpu_exp_sum = 0.0;
                    double htp_exp_sum = 0.0;
                    for (int i = 0; i < vocab_size; ++i) {
                        cpu_exp_sum += std::exp(static_cast<double>(cpu_logits[i]) - cpu_max);
                        htp_exp_sum += std::exp(static_cast<double>(logits[i]) - htp_max);
                    }
                    const double cpu_lse = cpu_max + std::log(cpu_exp_sum);
                    const double htp_lse = htp_max + std::log(htp_exp_sum);
                    double kl = 0.0;
                    double step_sum_sq = 0.0;
                    double step_max_abs = 0.0;
                    for (int i = 0; i < vocab_size; ++i) {
                        const double log_p = static_cast<double>(cpu_logits[i]) - cpu_lse;
                        const double log_q = static_cast<double>(logits[i]) - htp_lse;
                        kl += std::exp(log_p) * (log_p - log_q);
                        const double diff = static_cast<double>(logits[i]) - cpu_logits[i];
                        step_sum_sq += diff * diff;
                        step_max_abs = std::max(step_max_abs, std::abs(diff));
                    }
                    total_kl += kl;
                    std::fprintf(stderr,
                        "MODEL_LOGIT_METRIC step=%d/%d top1_agree=%d kl=%.9g rmse=%.9g max_abs=%.9g\n",
                        step + 1, steps, best == cpu_best, kl,
                        std::sqrt(step_sum_sq / vocab_size), step_max_abs);
                    if (std::memcmp(logits, cpu_logits.data(),
                                    static_cast<size_t>(vocab_size) * sizeof(float)) == 0) {
                        ++exact_vectors;
                    }
                    for (int i = 0; i < vocab_size; ++i) {
                        const double diff = static_cast<double>(logits[i]) - cpu_logits[i];
                        sum_sq += diff * diff;
                        max_abs = std::max(max_abs, std::abs(diff));
                        const double denom = std::max(1e-12, std::abs(static_cast<double>(cpu_logits[i])));
                        max_relative = std::max(max_relative, std::abs(diff) / denom);
                    }
                    values += static_cast<size_t>(vocab_size);
                }
                std::fprintf(stderr, "MODEL_LOGIT_GATE pass=%s step=%d/%d top1=%d\n",
                             pass ? "HTP0" : "CPU", step + 1, steps, best);
            }
            batch.reset();
            context.reset();
            model.reset();
            if (pass == 0) {
                std::fprintf(stderr, "MODEL_LOGIT_COOLDOWN start_s=180\n");
                std::this_thread::sleep_for(std::chrono::seconds(180));
                std::fprintf(stderr, "MODEL_LOGIT_COOLDOWN complete_s=180\n");
            }
        }
        llama_backend_free();
        std::printf("{\"steps\":%d,\"prompt_tokens\":%zu,\"vocab_vectors\":%d,\"exact_vectors\":%d,\"top1_agreements\":%d,\"logit_rmse\":%.9g,\"max_abs_error\":%.9g,\"max_relative_error\":%.9g,\"mean_kl\":%.9g,\"all_finite\":true}\n",
                    steps, static_cast<size_t>(prompt_token_count), steps, exact_vectors,
                    top1_agreements, std::sqrt(sum_sq / std::max<size_t>(1, values)),
                    max_abs, max_relative, total_kl / steps);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "MODEL_LOGIT_GATE FAILED: %s\n", e.what());
        llama_backend_free();
        return 1;
    }
}
