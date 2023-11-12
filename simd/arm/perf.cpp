
/*!
* \brief Performance
*/

#include <time.h>
#include <stdio.h>

#define LOOP (1e9)
#define OP_FLOATS (80) // fmla为乘一次加一次，一条操作4个float，共10条 = 2*4*10 = 80
                       // 如16条，则为128

// https://zhuanlan.zhihu.com/p/28226956
// 理论峰值就等于2(port) * 4 * 2(mul+add) * 频率 ( * 核心数 )
//              16 * 2G = 32G
void TEST(int loop) {
    float *pa = new float[1000];
    float *pb = new float[1000];
    float *pc = new float[1000];

    for (int i=0; i<loop; i++) {
        asm volatile(
            "fmla v0.4s, v0.4s, v0.4s \n"
            "fmla v1.4s, v1.4s, v1.4s \n"
            "fmla v2.4s, v2.4s, v2.4s \n"
            "fmla v3.4s, v3.4s, v3.4s \n"
            "fmla v4.4s, v4.4s, v4.4s \n"
            "fmla v5.4s, v5.4s, v5.4s \n"
            "fmla v6.4s, v6.4s, v6.4s \n"
            "fmla v7.4s, v7.4s, v7.4s \n"
            "fmla v8.4s, v8.4s, v8.4s \n"
            "fmla v9.4s, v9.4s, v9.4s \n"
            
            // "fmla v0.4s, v0.4s, v0.s[0] \n"
            // "fmla v1.4s, v1.4s, v1.s[0] \n"
            // "fmla v2.4s, v2.4s, v2.s[0] \n"
            // "fmla v3.4s, v3.4s, v3.s[0] \n"
            // "fmla v4.4s, v4.4s, v4.s[0] \n"
            // "fmla v5.4s, v5.4s, v5.s[0] \n"
            // "fmla v6.4s, v6.4s, v6.s[0] \n"
            // "fmla v7.4s, v7.4s, v7.s[0] \n"
            // "fmla v8.4s, v8.4s, v8.s[0] \n"
            // "fmla v9.4s, v9.4s, v9.s[0] \n"

            "ldr	x10, [%0] \n"   
            "fmla v0.4s, v0.4s, v0.s[0] \n"
            "ldr	x11, [%1] \n"
            "fmla v1.4s, v1.4s, v1.s[0] \n"
            "ldr	x12, [%2] \n"
            "fmla v2.4s, v2.4s, v2.s[0] \n"
            "add	x10, x10, #64 \n"
            "fmla v3.4s, v3.4s, v3.s[0] \n"
            "fmla v4.4s, v4.4s, v4.s[0] \n"
            "fmla v5.4s, v5.4s, v5.s[0] \n"
            "ins    v20.d[1], x10 \n" 

            "fmla v6.4s, v6.4s, v6.s[0] \n"            
            "fmla v7.4s, v7.4s, v7.s[0] \n"
            
            // "ldr	x11, [%0, #8] \n"
            // "ins    v20.d[1], x10 \n" 
            "fmla v8.4s, v8.4s, v8.s[0] \n"
            "fmla v9.4s, v9.4s, v9.s[0] \n"

            : "=r"(pa), "=r"(pb), "=r"(pc)
            : "0"(pa), "1"(pb), "2"(pc)
            : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v20"
        );
    }
}

static double get_time(struct timespec *start,
                       struct timespec *end) {
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}



int main() {
    struct timespec start, end;
    double time_used = 0.0;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    TEST(LOOP);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    time_used = get_time(&start, &end);    
    // LOOP是 1e9，即1G，乘以OP_FLOATS，即为 GFLOP
    // 1e-9 将 time_used 的单位转化为秒，1e-9 / time_used即为每秒。
    printf("perf: %.6lf \r\n", LOOP * OP_FLOATS * 1.0 * 1e-9 / time_used); // GFLOPS, G floating-point operations per second
}
