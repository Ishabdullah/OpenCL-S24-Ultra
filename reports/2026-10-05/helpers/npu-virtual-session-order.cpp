#include "llama.h"
#include <cstdio>
#include <exception>
#include <string>
#include <stdexcept>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const std::string order = argv[1];
    if (order != "0" && order != "1" && order != "0,1") return 2;
    std::vector<ggml_backend_t> backends;
    int result = 0;
    try {
        llama_backend_init();
        for (char index : order) {
            if (index == ',') continue;
            const std::string name = std::string("HTP0:") + index;
            fprintf(stderr, "SESSION_ORDER opening %s\n", name.c_str());
            fflush(stderr);
            auto * device = ggml_backend_dev_by_name(name.c_str());
            if (!device) throw std::runtime_error("requested device absent");
            auto * backend = ggml_backend_dev_init(device, nullptr);
            if (!backend) throw std::runtime_error("backend initialization returned null");
            backends.push_back(backend);
            fprintf(stderr, "SESSION_ORDER opened %s\n", name.c_str());
            fflush(stderr);
        }
        printf("{\"order\":\"%s\",\"sessions_opened\":%zu,\"model_loaded\":false,\"graph_operations\":0}\n", order.c_str(), backends.size());
    } catch (const std::exception & error) {
        fprintf(stderr, "SESSION_ORDER failed: %s\n", error.what());
        result = 3;
    }
    for (auto it = backends.rbegin(); it != backends.rend(); ++it) ggml_backend_free(*it);
    llama_backend_free();
    fprintf(stderr, "SESSION_ORDER cleanup complete\n");
    return result;
}
