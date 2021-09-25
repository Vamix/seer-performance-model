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
    int batchSize, inputChan, inputHeight, inputWidth, outputChan, outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, paddingH, paddingW;
    paddingH = 0; paddingW = 0;
    cudnnConvolutionFwdAlgo_t algo;

    if(argc > 1){ // profiling ops from file
      
        printf("=====> Test: %s...\n", argv[1]);
        infile.open(argv[1], ios::in);
        if(!infile.is_open ())
            cout << "Open file failure" << endl;
        // first line is number of ops
        infile >> num ; 
        // test specific algo
        while (0 != num ){
          infile >> batchSize >> inputChan >> inputHeight >> outputChan >> outputHeight >> kernelH >> strideH >> paddingH;
          inputWidth = inputHeight; outputWidth = outputHeight; kernelW = kernelH; strideW = strideH; paddingW = paddingH;
          outputHeight = 1 + (inputHeight + 2 * paddingH - kernelH) / strideH; 
          outputWidth = 1 + (inputWidth + 2 * paddingW - kernelW) / strideW;
          printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t", batchSize, inputChan, inputHeight, outputChan, outputHeight, kernelH, strideH, paddingH);  
          int best_cudnn, best_real;
          // cuDNN algo picker
          best_cudnn = find_conv2d_algo(batchSize, inputChan, inputHeight, inputWidth, outputChan, outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, paddingH, paddingW);
          // test all algos
          float timeList[] = {0,0,0,0,0,0,0,0};
          float tmp_time = 100000000.0;
          best_real = -1; 
          for(int m = 0; m < 8; m++){  
            algo = algos[m];
            timeList[m]= measure_conv2d_time_with_algo(batchSize, inputChan, inputHeight, inputWidth, outputChan, outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, paddingH, paddingW, algo);
            if(timeList[m]>0 && timeList[m] < tmp_time){
              best_real = m;
              tmp_time = timeList[m];
            }
          }
          printf("%.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf\t%d\t%d\n", timeList[0], timeList[1], timeList[2], timeList[3], timeList[4], timeList[5], timeList[6], timeList[7], best_real, best_cudnn);  
          num--;
        }
        infile.close();    

    }
    else{
      printf("=====> Algo picker data collecting...\n");
      // for test algorithm picker
      int batchList[] = {128};                                     // 1
      int inwidList[] = {20, 40, 80, 100, 200, 300, 400, 500};     // 1x8 = 8
      int inchanList[] = {10, 20, 50, 100, 200, 400, 800, 1000};   // 1x8x8 = 64
      int outchanList[] = {10, 20, 50, 100, 200, 400, 800, 1000};  // 1x8x8x8 = 512
      int kerwidList[] = {1, 3, 5, 7, 9, 11};                      // 1x8x8x8x6 = 3072
      int strideList[] ={1, 2, 3};                                 // 1x8x8x8x6x3 = 9216 

      for(int i = 0; i < length(batchList); i++)
        for(int j = 0; j < length(inwidList); j++)
          for(int k = 0; k < length(inchanList); k++)
            for(int l = 0; l < length(outchanList); l++)
              for(int m = 0; m < length(kerwidList); m++)
                for(int n = 0; n < length(strideList); n++){
                  batchSize = batchList[i];
                  inputWidth = inputHeight = inwidList[j];
                  inputChan = inchanList[k];
                  outputChan = outchanList[l];
                  kernelH = kernelW = kerwidList[m];
                  strideW = strideH = strideList[n];
                  outputHeight = 1 + (inputHeight + 2 * paddingH - kernelH) / strideH; 
                  outputWidth = 1 + (inputWidth + 2 * paddingW - kernelW) / strideW;   
                  printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t", batchSize, inputChan, inputHeight, outputChan, outputHeight, kernelH, strideH, paddingH);  
          
                  int best_cudnn, best_real;
                  // cuDNN algo picker
                  best_cudnn = find_conv2d_algo(batchSize, inputChan, inputHeight, inputWidth, outputChan, outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, paddingH, paddingW);
                  // test all algos
                  float timeList[] = {0,0,0,0,0,0,0,0};
                  float tmp_time = 100000000.0;
                  best_real = -1; 
                  for(int m = 0; m < 8; m++){  
                    algo = algos[m];
                    timeList[m]= measure_conv2d_time_with_algo(batchSize, inputChan, inputHeight, inputWidth, outputChan, outputHeight, outputWidth, kernelH, kernelW, strideH, strideW, paddingH, paddingW, algo);
                    if(timeList[m]>0 && timeList[m] < tmp_time){
                      best_real = m;
                      tmp_time = timeList[m];
                    }
                  }
                  printf("%.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf\t%d\t%d\n", timeList[0], timeList[1], timeList[2], timeList[3], timeList[4], timeList[5], timeList[6], timeList[7], best_real, best_cudnn);  
                }
    }
    return 0;
}

