#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <cuda.h>
#include <curand_kernel.h>

#define N 500000000             //Numero de valores de entrada
#define M 8                     //Tamaño del histograma

#define REPETICONES 10000       //Repeticon de pruevas para calculo de media, max y min
#define SCALA 50                //Datos calculados en cada hilo

__device__ int vector_V[N];     //Vector de datos de entrada
__device__ int vector_H[M];     //Vector del histograma

/**
* Funcion para la comprovacion de errores cuda 
*/
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
*   Kernel para inicializacion de datos de entrada
*/
__global__ void inicializa_v(int random, curandState *states, int threadsPerBlock, int blocksPerGrid){
    int iteraciones= SCALA;
    if(blocksPerGrid-1 == blockIdx.x && threadIdx.x == threadsPerBlock -1){
        iteraciones = iteraciones + (N % SCALA);
    }
    unsigned id_x = blockIdx.x*blockDim.x + threadIdx.x;
    curandState *state = states + id_x;

    curand_init(random, id_x, 0, state);
    for(int i = 0; i < iteraciones; i++){
        if(id_x*SCALA+i < N){
            vector_V[id_x*SCALA+i] = (int)((curand_uniform(state)*1000)) % M;
        }
    }

}

/**
*   Kernel para inicializacion del vector de histograma
*/
__global__ void inicializa_h(){
    unsigned id_x = blockIdx.x*blockDim.x + threadIdx.x;
    vector_H[id_x] = 0;
}
/**
*   Kernel para calculo del histograma
*/
__global__ void histograma(int threadsPerBlock, int blocksPerGrid){
    int vector[M];
    for(int i =0; i < M;i++){
        vector[i] =0;
    }
    int iteraciones= SCALA;
    if(blocksPerGrid-1 == blockIdx.x && threadIdx.x == threadsPerBlock -1){
       iteraciones = iteraciones + (N % SCALA);
    }
    unsigned id_x = blockIdx.x*blockDim.x + threadIdx.x;
    for(int i = 0; i < iteraciones; i++){
        if(id_x*SCALA+i < N){
            int mod = vector_V[id_x*SCALA+i]%M;
            vector[mod]++;
        }
    }
    for(int i =0; i < M;i++){
        int a =vector[i];
        atomicAdd(&vector_H[i],a);
    }
}



int main(){
    srand(time(NULL));
    static curandState *states = NULL;
    //int h_v_d[N];
    int h_v_h[M];
    int threadsPerBlock = 1024;
    int blocksPerGrid =((N/SCALA) + threadsPerBlock - 1) / threadsPerBlock;

    float t_duration[REPETICONES];
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int j = 0; j< REPETICONES; j++){
        CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

        CUDA_CHECK_RETURN(cudaMalloc((void **)&states, sizeof(curandState) * threadsPerBlock  * blocksPerGrid));
        inicializa_v<<<blocksPerGrid, threadsPerBlock>>>(rand(),states, threadsPerBlock,blocksPerGrid);
        CUDA_CHECK_RETURN(cudaGetLastError());
        inicializa_h<<<1,M>>>();
        CUDA_CHECK_RETURN(cudaGetLastError());

        histograma<<<blocksPerGrid,threadsPerBlock>>>(threadsPerBlock,blocksPerGrid);
        CUDA_CHECK_RETURN(cudaGetLastError());

        //CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(h_v_d, vector_V, N*sizeof(int)));
        CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(h_v_h, vector_H, M*sizeof(int)));
        int acumula =0;
        for(int  i = 0; i<M; i++){
            std::cout<<h_v_h[i]<<" ";
            acumula += h_v_h[i];
        }
        std::cout<<"\n-------------------------"<<acumula<<"-----------------------------------\n";
        /*
        for(int  i = 0; i<10; i++){
            for(int  j = 0; j<10; j++){
            std::cout<<h_v_d[10*i+j]<<" ";
            };
            std::cout<<"\n";
        }
        */  
        CUDA_CHECK_RETURN(cudaFree(states));
        CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
        CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

        CUDA_CHECK_RETURN(cudaEventElapsedTime(&t_duration[j],start,stop));  
    }
    float t_max =0, t_min= FLT_MAX, media=0;
    for(int i = 0; i< REPETICONES; i++){
        media +=t_duration[i];
        if(t_duration[i] > t_max){
            t_max =t_duration[i]; 
        }
        if(t_duration[i]< t_min){
            t_min= t_duration[i];
        }
    }
    std::cout<< "Se han realizado "<<REPETICONES<<" repeticones\n";
    std::cout<<"Obteniendo de media: "<<media/REPETICONES<<"ms \n";
    std::cout<<"Y de máximo: "<<t_max<<"ms  y mínimo: "<<t_min<<"ms\n";

    return 0;
}



/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {

	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (EXIT_FAILURE);
}
