#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <assert.h>

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    } while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCUDA(status) do {                                         \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define REPEAT_TIMES 1
#define MAX_SEQ_LENGTH 40
#define TEST_NUM 10485760

const char* algo_names[] = {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM","CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
                            "CUDNN_CONVOLUTION_FWD_ALGO_GEMM", "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
                            "CUDNN_CONVOLUTION_FWD_ALGO_FFT", "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
                            "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD", "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"};
const char* cudnn_status[] = {"CUDNN_STATUS_SUCCESS", "CUDNN_STATUS_NOT_INITIALIZED", "CUDNN_STATUS_ALLOC_FAILED",
                              "CUDNN_STATUS_BAD_PARAM", "CUDNN_STATUS_INTERNAL_ERROR", "CUDNN_STATUS_INVALID_VALUE",
                              "CUDNN_STATUS_ARCH_MISMATCH", "CUDNN_STATUS_MAPPING_ERROR", "CUDNN_STATUS_EXECUTION_FAILED",
                              "CUDNN_STATUS_NOT_SUPPORTED", "CUDNN_STATUS_LICENSE_ERROR", 
                              "cudaSuccess", "cudaErrorInvalidValue", "cudaErrorMemoryAllocation"};

cudnnHandle_t dnn;
cublasHandle_t blas;
void* workSpace;
size_t workSpaceSize;

void init_cudnn(){
  checkCUDNN(cudnnCreate(&dnn));
  checkCUDA(cublasCreate(&blas));
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(dnn, stream));
  checkCUDA(cublasSetStream(blas, stream));
  // workSpaceSize = (size_t) 4* 1024 * 1024 * 1024;
  // checkCUDA(cudaMalloc(&workSpace, workSpaceSize));
}

__global__ void data_copy(float *x, float *y)
{
  int index = (blockIdx.x * blockDim.x + threadIdx.x) * 32;
  y[index] = x[index] + 1;
}

void warm_up(){
  // allocate host memory
  float *x, *y;
  x = (float*)malloc(TEST_NUM * sizeof(float));
  y = (float*)malloc(TEST_NUM * sizeof(float));

  // initialize data
  for (int i = 0; i < TEST_NUM; ++i){
    x[i] = (rand() % 1000)/10.0;
    y[i] = (rand() % 1000)/10.0;
  }

  // allocate device memory
  float *d_x, *d_y;
  cudaMalloc((void**)&d_x, TEST_NUM * sizeof(float));
  cudaMalloc((void**)&d_y, TEST_NUM * sizeof(float));

  // copy host data to device
  cudaMemcpy((void*)d_x, (void*)x, TEST_NUM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*)d_y, (void*)y, TEST_NUM * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(256);
  dim3 gridSize(128);
  data_copy<<<gridSize, blockSize>>>(d_x, d_y);
  checkCUDA(cudaDeviceSynchronize());
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
  // printf("===DEBUG: finish cuda Free\n");
}

template<class T>
int length(T& arr)
{
    return sizeof(arr) / sizeof(arr[0]);
}

float conv2DForwardTimeWithAlgo(cudnnHandle_t handle,
                        const cudnnTensorDescriptor_t xDesc, const void* x,
                        const cudnnFilterDescriptor_t wDesc, const void* w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        void* workSpace, size_t workSpaceSize,
                        const cudnnTensorDescriptor_t yDesc, void* y,
                        cudnnConvolutionFwdAlgo_t algo, cudnnStatus_t* status_output)
{
  float alpha = 1.0f, beta = 0.0f;
  cudaEvent_t t_start, t_end;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start);
  for (int i = 0; i < REPEAT_TIMES; i++){
    cudnnStatus_t status = cudnnConvolutionForward(handle, &alpha, xDesc, x, wDesc, w, convDesc, 
                                       algo, workSpace, workSpaceSize,
                                       &beta, yDesc, y);
    *status_output = status;                                   
    if(status != CUDNN_STATUS_SUCCESS){
      cudaEventDestroy(t_start);
      cudaEventDestroy(t_end);
      return -1;
    }
  }
  cudaEventRecord(t_end);
  checkCUDA(cudaEventSynchronize(t_end));
  float elapsed = 0;
  checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  cudaEventDestroy(t_start);
  cudaEventDestroy(t_end);
  return elapsed / REPEAT_TIMES;
}

int conv2DForwardAlgo(cudnnHandle_t handle,
                        const cudnnTensorDescriptor_t xDesc, const void* x,
                        const cudnnFilterDescriptor_t wDesc, const void* w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        void* workSpace, size_t workSpaceSize,
                        const cudnnTensorDescriptor_t yDesc, void* y)
{
  const int reqAlgCnt = 8;
  int cnt = 0;
  cudnnConvolutionFwdAlgoPerf_t perfResults[reqAlgCnt];

  // search for best algorithm
  // checkCUDNN(cudnnFindConvolutionForwardAlgorithmEx(
  //     handle, xDesc, x, wDesc, w, convDesc, yDesc, y,
  //     reqAlgCnt, &cnt, perfResults, workSpace, workSpaceSize));

  // return best algorithm based on heuristic
  checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
      handle, xDesc, wDesc, convDesc, yDesc, reqAlgCnt, &cnt, perfResults));      
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  return (int)perfResults[0].algo;
}

float measure_conv2d_time_with_algo(int batchSize, int inputSize,
                          int inputHeight, int inputWidth,
                          int outputSize,
                          int outputHeight, int outputWidth,
                          int kernelH, int kernelW,
                          int strideH, int strideW,
                          int paddingH, int paddingW, 
                          cudnnConvolutionFwdAlgo_t algo){
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnStatus_t status;

  status = cudnnCreateTensorDescriptor(&inputTensor);
  if(status != CUDNN_STATUS_SUCCESS){
  //  printf("[1] CUDNN failure: %s\n", cudnnGetErrorString(status)); 
   return -5;
  }
  status = cudnnCreateTensorDescriptor(&outputTensor);
  if(status != CUDNN_STATUS_SUCCESS){
  //  printf("[2] CUDNN failure: %s\n", cudnnGetErrorString(status)); 
   checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
   return -5;
  }  
  status = cudnnCreateFilterDescriptor(&filterDesc);
  if(status != CUDNN_STATUS_SUCCESS){
  //  printf("[3] CUDNN failure: %s\n", cudnnGetErrorString(status)); 
   checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
   checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
   return -5;
  }  
  status = cudnnCreateConvolutionDescriptor(&convDesc);
  if(status != CUDNN_STATUS_SUCCESS){
  //  printf("[4] CUDNN failure: %s\n", cudnnGetErrorString(status)); 
   checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
   checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
   checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
   return -5;
  }    
  status = cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, inputSize, inputHeight, inputWidth);
  if(status != CUDNN_STATUS_SUCCESS){
  //  printf("[5] CUDNN failure: %s\n", cudnnGetErrorString(status)); 
   checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
   checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
   checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
   checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));   
   return -5;
  }    
  status = cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outputSize, inputSize, kernelH, kernelW);
  if(status != CUDNN_STATUS_SUCCESS){
  //  printf("[6] CUDNN failure: %s\n", cudnnGetErrorString(status)); 
   checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
   checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
   checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
   checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));   
   return -5;
  }      
  status = cudnnSetConvolution2dDescriptor(convDesc, paddingH, paddingW, strideH, strideW, 1/*upscale_x*/, 1/*upscale_y*/, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
  if(status != CUDNN_STATUS_SUCCESS){
  //  printf("[7] CUDNN failure: %s\n", cudnnGetErrorString(status)); 
   checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
   checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
   checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
   checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));   
   return -5;
  }      
  int n, c, h, w;
  status = cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensor, filterDesc, &n, &c, &h, &w);
  if(status != CUDNN_STATUS_SUCCESS){
  //  printf("[8] CUDNN failure: %s\n", cudnnGetErrorString(status)); 
   checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
   checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
   checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
   checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));   
   return -5;
  }      
  assert(n == batchSize);
  assert(c == outputSize);
  assert(h == outputHeight);
  assert(w == outputWidth);
  status = cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
  if(status != CUDNN_STATUS_SUCCESS){
  //  printf("[9] CUDNN failure: %s\n", cudnnGetErrorString(status)); 
    checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));     
   return -5;
  }

  float *input_ptr, *filter_ptr, *output_ptr;
  size_t input_size = ((size_t)n * inputSize * inputHeight * inputWidth * sizeof(float));
  size_t filter_size = ((size_t)inputSize * outputSize * kernelH * kernelW * sizeof(float));
  size_t output_size = ((size_t)n * c * h * w * sizeof(float));

  checkCUDA(cudaMalloc(&input_ptr, input_size));
  checkCUDA(cudaMalloc(&filter_ptr, filter_size));
  checkCUDA(cudaMalloc(&output_ptr, output_size));
  checkCUDA(cudaDeviceSynchronize());

  cudaError_t cuda_status;
  float t1 = -1;
  status = cudnnGetConvolutionForwardWorkspaceSize(dnn, inputTensor, filterDesc, convDesc, outputTensor, algo, &workSpaceSize);
  if(status != CUDNN_STATUS_SUCCESS){
		// printf("getWorkSize failed.\n");
	  t1 = -2;
  }
  else{
    cuda_status = cudaMalloc(&workSpace, workSpaceSize);
    if ( cuda_status != 0){
      // printf("cudaMalloc failed. required space size: %ld\n", workSpaceSize);
		  t1 = -3;
      status = cudnnStatus_t(static_cast<int>(cuda_status) + 11);
    }
    else{
      checkCUDA(cudaDeviceSynchronize());
      t1 = conv2DForwardTimeWithAlgo(dnn, inputTensor, input_ptr,
                               filterDesc, filter_ptr, convDesc,
                               workSpace, workSpaceSize,
                               outputTensor, output_ptr, algo, &status);
      checkCUDA(cudaDeviceSynchronize());
    }
  }

  checkCUDA(cudaFree(input_ptr));
  checkCUDA(cudaFree(filter_ptr));
  checkCUDA(cudaFree(output_ptr));
  checkCUDA(cudaFree(workSpace));
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
  checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
  return t1;
}

int find_conv2d_algo(int batchSize, int inputSize,
                          int inputHeight, int inputWidth,
                          int outputSize,
                          int outputHeight, int outputWidth,
                          int kernelH, int kernelW,
                          int strideH, int strideW,
                          int paddingH, int paddingW)
{
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionDescriptor_t convDesc;

  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        batchSize, inputSize, inputHeight, inputWidth));
  checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                        outputSize, inputSize, kernelH, kernelW));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, paddingH, paddingW, strideH, strideW,
                                             1/*upscale_x*/, 1/*upscale_y*/,
                                             CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  int n, c, h, w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensor, filterDesc,
                                                   &n, &c, &h, &w));
  assert(n == batchSize);
  assert(c == outputSize);
  assert(h == outputHeight);
  assert(w == outputWidth);

  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        n, c, h, w));
  float *input_ptr, *filter_ptr, *output_ptr;
  size_t input_size = ((size_t)n * inputSize * inputHeight * inputWidth * sizeof(float));
  size_t filter_size = ((size_t)inputSize * outputSize * kernelH * kernelW * sizeof(float));
  size_t output_size = ((size_t)n * c * h * w * sizeof(float));
  checkCUDA(cudaMalloc(&input_ptr, input_size));
  checkCUDA(cudaMalloc(&filter_ptr, filter_size));
  checkCUDA(cudaMalloc(&output_ptr, output_size));
  checkCUDA(cudaDeviceSynchronize());
  int algo_num = conv2DForwardAlgo(dnn, inputTensor, input_ptr,
                               filterDesc, filter_ptr, convDesc,
                               workSpace, workSpaceSize,
                               outputTensor, output_ptr);
  checkCUDA(cudaFree(input_ptr));
  checkCUDA(cudaFree(filter_ptr));
  checkCUDA(cudaFree(output_ptr));
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
  checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
  return algo_num;
}
