// 一个简单的通信环
__global__ void simple_shift(int* destination){
    int mype = nvshmem_my_pe();   // 当前的PE ID
    int npes = nvshmem_n_pe();    // PE总数
    int peer = (mype + 1) % npes; // 目标PE ID

    // 行代码调用了 NVSHMEM的单边通信原语：nvshmem_int_p。
    // destination 是目标内存地址，表示数据将被写入的内存区域。
    // 将指定的整数 mype 从当前PE写入到指定 peer 的对称内存位置 destination
    nvshmem_int_p(destination, mype, peer);
}

int main(){
    int mype_node, msg;
    cudaStream_t stream;

    nvshmem_init();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE); // 查询当前PE的ID
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream); 

    // host端调用API，在当前PE上分配symmetric memory
    int* destination = (int*) nvshmem_malloc(sizeof(int));

    simple_shift<<<1, 1, 0, stream>>>(destination);
    nvshmemx_barrier_all_on_stream(stream);// 同步操作，确保所有PE完成NVSHMEM操作
    cudaMemcpyAsync(&msg, destination, sizeof(int), cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);// 同步操作，确保当前stream中所有的操作完成
    printf("%d: received message %d\n", nvshmem_my_pe(), msg);

    nvshmem_free(destination);
    nvshmem_finalize();
    return 0;
}