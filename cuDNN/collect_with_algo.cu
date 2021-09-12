#include <stdio.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <assert.h>
#include <cmath>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <fstream> 
#include "cnn.h"

using namespace std;

cudnnConvolutionFwdAlgo_t algos[] = {CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
  CUDNN_CONVOLUTION_FWD_ALGO_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
  CUDNN_CONVOLUTION_FWD_ALGO_FFT, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED};

int main(int   argc, char*   argv[]){
    init_cudnn();
    int num;
    ifstream infile; 
    int batchSize, inputChan, inputHeight, inputWidth, outputChan, outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, paddingH, paddingW, algo_index;
    paddingH = 0; paddingW = 0;
    cudnnConvolutionFwdAlgo_t algo;
    float tmp_time;

    if(argc > 1){ // profiling ops from file
      infile.open(argv[1], ios::in);

      if(!infile.is_open ())
          cout << "Open file failure" << endl;
      // first line is number of ops
      infile >> num ; 
      // test specific algo
      while (0 != num ){
        infile >> batchSize >> inputChan >> inputHeight >> outputChan >> outputHeight >> kernelH >> strideH >> paddingH  >> algo_index;
        inputWidth = inputHeight; outputWidth = outputHeight; kernelW = kernelH; strideW = strideH; paddingW = paddingH;
        outputHeight = 1 + (inputHeight + 2 * paddingH - kernelH) / strideH; 
        outputWidth = 1 + (inputWidth + 2 * paddingW - kernelW) / strideW;
        algo = algos[algo_index];
        // workaround of unstable dram transaction measurement 
        // warm_up();
        tmp_time = measure_conv2d_time_with_algo(batchSize, inputChan, inputHeight, inputWidth, outputChan, outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, paddingH, paddingW, algo);
        printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.2lf\n", batchSize, inputChan, inputHeight, outputChan, outputHeight, kernelH, strideH, paddingH, static_cast<int>(algo), tmp_time);  
        num--;
      }
      infile.close();     
    }
    else{
      printf("[ERROR] usage: ./collect_with_algo prof-ops.txt\n");
    }
    return 0;
}

