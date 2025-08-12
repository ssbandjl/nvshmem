#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
 
__global__ void simple_shift(int *destination) {
    int mype = nvshmem_my_pe();    // 当前GPU的PE编号
    int npes = nvshmem_n_pes();    // 总GPU数量
    int peer = (mype + 1) % npes;  // 环形通信中的下一个GPU
    nvshmem_int_p(destination, mype, peer);  // 将mype的值发送到peer
}
 
 
int main(void) {
    int mype_node, msg;
    cudaStream_t stream;
    nvshmem_init();  // 初始化NVSHMEM
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);  // 获取节点编号
    cudaSetDevice(mype_node);  // 设置当前CUDA设备为对应的GPU
    cudaStreamCreate(&stream);  // 创建CUDA流
 
    int *destination = (int *) nvshmem_malloc(sizeof(int));  // 分配共享内存
    simple_shift<<<1, 1, 0, stream>>>(destination);  // 启动CUDA核函数
    nvshmemx_barrier_all_on_stream(stream);  // 同步所有GPU节点
    cudaMemcpyAsync(&msg, destination, sizeof(int), cudaMemcpyDeviceToHost, stream);  // 异步复制结果
    cudaStreamSynchronize(stream);  // 同步CUDA流
 
    printf("%d: received message %d\n", nvshmem_my_pe(), msg);  // 打印接收到的消息
 
    nvshmem_free(destination);  // 释放共享内存
    nvshmem_finalize();  // 清理NVSHMEM环境
    return 0;
}