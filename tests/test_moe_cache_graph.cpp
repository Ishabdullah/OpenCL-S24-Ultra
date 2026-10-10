// Numerical and graph-dependency gate using real CPU quantized MUL_MAT_ID.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

static std::vector<float> run(int slots, int threads, ggml_type type) {
    if (slots) setenv("GGML_CPU_MOE_CACHE_SLOTS", std::to_string(slots).c_str(), 1);
    else unsetenv("GGML_CPU_MOE_CACHE_SLOTS");
    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    auto ctx = ggml_init({1024*1024, nullptr, true});
    auto * w = ggml_new_tensor_3d(ctx, type, 256, 16, 6);
    ggml_set_name(w, "blk.0.ffn_up_exps.weight");
    auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 2);
    auto * router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, 2);
    // Deliberately a strided VIEW into the full argsort output.
    auto * ids = ggml_argsort_top_k(ctx, router, 2);
    auto * y = ggml_mul_mat_id(ctx, w, x, ids);
    // A later node still consumes the ORIGINAL IDs, not compact slot IDs.
    auto * probabilities = ggml_reshape_3d(ctx, router, 1, 6, 2);
    auto * selected = ggml_get_rows(ctx, probabilities, ids);
    auto * combined = ggml_mul(ctx, y, selected);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, combined);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);
    std::vector<float> floats(256*16*6), activation(256*2);
    for (size_t i=0;i<floats.size();++i) floats[i]=std::sin(float(i)*0.07f);
    std::vector<float> importance(256, 1.0f);
    std::vector<uint8_t> quant(ggml_nbytes(w));
    assert(ggml_quantize_chunk(type, floats.data(), quant.data(), 0, 16*6, 256, importance.data())==quant.size());
    ggml_backend_tensor_set(w, quant.data(), 0, quant.size());
    for(size_t i=0;i<activation.size();++i) activation[i]=std::cos(float(i)*0.03f);
    ggml_backend_tensor_set(x, activation.data(), 0, activation.size()*4);
    // First 3 requests exercise hits and eviction; fourth exceeds 2 slots.
    const int pairs[][4]={{1,4,1,4},{0,4,0,4},{3,0,3,0},{0,1,2,3},{3,0,3,0}};
    std::vector<float> result;
    for(const auto & pair:pairs) {
        float scores[12]; std::fill(scores,scores+12,-5.0f);
        for(int token=0;token<2;++token) {
            scores[token*6+pair[token*2]]=2;
            scores[token*6+pair[token*2+1]]=1;
        }
        ggml_backend_tensor_set(router,scores,0,sizeof(scores));
        assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
        size_t old=result.size(); result.resize(old+ggml_nelements(combined));
        ggml_backend_tensor_get(combined,result.data()+old,0,ggml_nbytes(combined));
        int32_t got[2];
        for(int token=0;token<2;++token) {
            ggml_backend_tensor_get(ids,got,token*ids->nb[1],sizeof(got));
            assert(got[0]==pair[token*2] && got[1]==pair[token*2+1]);
        }
    }
    ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(backend);
    return result;
}
int main() {
    for(auto type:{GGML_TYPE_Q2_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M}) {
        for(int threads:{1,6}) {
            auto baseline=run(0,threads,type), cached=run(2,threads,type);
            assert(baseline.size()==cached.size());
            assert(std::all_of(baseline.begin(),baseline.end(),[](float f){return std::isfinite(f);}));
            assert(memcmp(baseline.data(),cached.data(),baseline.size()*4)==0);
            printf("type=%s threads=%d: bit-identical outputs; router IDs preserved\n",ggml_type_name(type),threads);
        }
    }
}
