#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>


#define TILE_WIDTH 32
#define COARSE_FACTOR 16

__global__ void OnlineSotmax_matmul(float* A, float* V, float* Out, int width){
    __shared__ float Amax[TILE_WIDTH][TILE_WIDTH];
    __shared__ float V_s[TILE_WIDTH][TILE_WIDTH * COARSE_FACTOR];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_WIDTH + ty;
    
    // Registers to track the "Running State" of the Softmax for this row
    float m = -1e38f; // Running Max
    float d = 0.0f;   // Running Sum 

    // Lo
    float acc[COARSE_FACTOR];
    #pragma unroll //For looping
    for(int i = 0; i < COARSE_FACTOR; i++)
    acc[i] = 0.0f;
    for(int c = 0; c<width; c+=TILE_WIDTH){
        if(row < width && (c + tx) < width)
            Amax[ty][tx] = A[row * width + c * TILE_WIDTH + tx];  
        else
            Amax[ty][tx] = -1e38f;
        #pragma unroll 
        for(int k = 0; k<COARSE_FACTOR; k++){
            int col = (bx * TILE_WIDTH * COARSE_FACTOR) + (k * COARSE_FACTOR) + tx;
            if((k+ty)<width && col < (width * COARSE_FACTOR))
                V_s[ty][tx*COARSE_FACTOR] = V[(k * TILE_WIDTH + ty) * width + col];
            else
                V_s[ty][tx*COARSE_FACTOR] = 0.0f;
        }
        __syncthreads();
        //  To the online softmax, initialize first normalization
        for(int i = 0; i < TILE_WIDTH; i++){
        float m_o = m;
        float current_value = Amax[ty][i];
        // first
        float m = fmaxf(current_value,m_o);

        // L1
        float exp_old = __expf(m_o - m);
        float exp_new = __exp(current_value - m);

        d = d * exp_old + exp_new;
        #pragma unroll 
            for(int k = 0; k < COARSE_FACTOR; k++){
                acc[k] = acc[k] * exp_old + exp_new * V_s[i][k * TILE_WIDTH + tx];

            }
        }
        __syncthreads();
    }
    #pragma unroll 
    // Normalization
    for(int k = 0; k < COARSE_FACTOR; k++){
        int out_col = bx * TILE_WIDTH * COARSE_FACTOR + (k * TILE_WIDTH) + tx;
        if(row<width && out_col<(width * COARSE_FACTOR)){
            Out[row * (width*COARSE_FACTOR) + out_col] = acc[k] / d;
        }
    } 
    
}