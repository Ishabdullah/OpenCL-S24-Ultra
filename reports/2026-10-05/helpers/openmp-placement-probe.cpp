#include <omp.h>
#include <sched.h>
#include <stdio.h>
int main() {
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(6)
    {
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        const int rc = sched_getaffinity(0, sizeof(allowed), &allowed);
        #pragma omp critical
        {
            printf("thread=%d cpu=%d affinity_rc=%d allowed=", omp_get_thread_num(), sched_getcpu(), rc);
            for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &allowed)) printf("%d,", i);
            printf("\n");
        }
    }
}
