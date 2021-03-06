#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <cassert>
using namespace std;

typedef long long int lli;

void usage()
{
  cout << "USAGE: ./exec <filename> <device>" << endl;
  exit(0);
}

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__
void PreProcess(double* xCuda, double* xx, lli TOTAL, lli xSIZE)
{
  lli ind = (blockDim.x * blockIdx.x + threadIdx.x);
  double* my_xCuda = xCuda + ind;

  lli index = 0;
  for(lli j = 0; j < xSIZE; j+=TOTAL)
    {
      my_xCuda[j] = xx[index++];
    }
}
__global__
void PermanentCalculator(lli K, float* xCuda, float* pCuda, int* MCuda, lli CHUNK, lli N, lli xSIZE, lli TOTAL, lli Nsqr, lli BLOCK)
{
  lli index = (blockDim.x * blockIdx.x + threadIdx.x) * CHUNK;
  if(index < K)
    {
      lli ind = threadIdx.x;
      lli LIM = BLOCK * N;
      extern __shared__ float M[];
      float* temp_xCuda = M;
      float* my_xCuda = temp_xCuda + ind;

      lli new_ind = 0;
      for(lli i = 0; i < LIM; i+=BLOCK)
	my_xCuda[i] = xCuda[new_ind++];

      __syncthreads();
      
      lli START = index + 1;
      lli END = index + CHUNK + 1;
      
      lli yy = index ^ (index >> 1LL);
      lli y, y_prev, FFS, z;
      double s, prodSign;
      
      float pSelf = 0.0;
      lli temp_y = yy;
      lli BPC = __popcll(yy);
      for(lli n = 0; n < BPC; n++)
	{
	  FFS = __ffsll(temp_y) - 1;
	  temp_y &= ~(1LL << FFS);
	  new_ind = 0;
	  for(lli m = 0; m < LIM; m += BLOCK)
	    {
	      my_xCuda[m] += MCuda[(new_ind++) + (FFS*N)];
	      __syncthreads();
	    }
	}
      prodSign = ((index + 1) & 1LL) ? -1.0 : 1.0;
      for(lli ii = START; (ii < END) && (ii < K); ii++)
	{
	  y = (ii ^ (ii >> 1LL));
	  y_prev = (ii - 1) ^ ((ii - 1) >> 1LL);
	  z = __ffsll(y ^ y_prev) - 1;      
	  s = ((y >> z) & 1LL) ? 1.0 : -1.0;
	  new_ind = 0;
	  float temp = 1.0;
	  //#pragma unroll
	  for(lli jj = 0; jj < LIM; jj += BLOCK)
	    {
	      my_xCuda[jj] += (s * MCuda[(new_ind++) + (z * N)]);
	      temp *= my_xCuda[jj];
	      __syncthreads();
	    }
	  pSelf += (prodSign * temp);
	  prodSign *= -1.0;      
	}
      atomicAdd(pCuda, pSelf);
    }
}



int main(int argc, const char** argv)
{

  if(argc != 3)
    usage();

  string line;

  const char* filename = argv[1];
  ifstream input (filename);
  if(input.fail())
    return 0;

  int cudaDevice = atoi(argv[2]);
  cudaSetDevice(cudaDevice);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, cudaDevice);
  
  lli N;
  lli Nsqr;
  int **M;
  int *Mrow;
  int *MCuda;
  getline(input,line);
  N = atoi(line.c_str());
  Nsqr = N*N;
  checkCuda(cudaMalloc((void**)&MCuda, sizeof(int)*Nsqr));
  
  Mrow = (int*)(malloc(sizeof(int)*Nsqr));
  M = (int**)(malloc(sizeof(int*)*N));
  for(int i = 0; i < N; i ++)
    {
      M[i] = (int*)(malloc(sizeof(int)*N));
    }

  int linectr = 0;
  while(getline(input,line))
    {
      stringstream ss(line);
      int temp;
      int ctr = 0;
      while(ss >> temp)
	{
	  M[linectr][ctr++] = temp;
	}
      linectr++;
    }

  int trctr = 0;
  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++)
      Mrow[trctr++] = M[j][i];
	 
  cudaMemcpy(MCuda, Mrow, sizeof(int)*Nsqr, cudaMemcpyHostToDevice);
  
  double start, end, initt;
  ////YOUR CODE GOES HERE
  lli K;
  K = 1LL << (N-1);
  float result = 0.0;
  
  float *p = (float*)(malloc(sizeof(float)));
  *p = 1.0;

  float *x = (float*)(malloc(sizeof(float)*N));
  float *sumCol = (float*)(malloc(sizeof(float)*N)); 
  float *lastCol = (float*)(malloc(sizeof(float)*N));
  memset(sumCol, 0, sizeof(float)*N);
  memset(lastCol, 0, sizeof(float)*N);


  //lli GRID = prop.maxThreadsDim[0];
  lli GRID = 1024*64;
  lli BLOCK = 256;
  lli TOTAL = GRID * BLOCK;
  
  lli CHUNK = (K + (TOTAL-1)) / TOTAL;
  lli xSIZE = TOTAL * N;
  size_t SHARED = (BLOCK * N * sizeof(float));
  
  float *pCuda;  
  float *xx;
  cudaMalloc((void**)&pCuda, sizeof(float));
  cudaMalloc((void**)&xx, sizeof(float)*N);
  //printf("Device Variable Copying:\t%s\n", cudaGetErrorString(cudaGetLastError()));

  //BEGINNING
  initt = omp_get_wtime();

  for(int i = 0; i< N; i++)
    {
      lastCol[i] = M[i][N-1];
      for(int j = 0; j < N; j++)
	sumCol[i] += M[i][j];
      x[i] = (lastCol[i] - sumCol[i]/2);
      (*p) *= x[i];
    }
      
  cudaMemcpy(xx, x, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(pCuda, p, sizeof(float), cudaMemcpyHostToDevice);
  
  start = omp_get_wtime();
  PermanentCalculator<<<GRID, BLOCK, SHARED>>>(K, xx, pCuda, MCuda, CHUNK, N, xSIZE, TOTAL, Nsqr, BLOCK);
  cudaDeviceSynchronize();
  cudaMemcpy(p, pCuda, sizeof(float), cudaMemcpyDeviceToHost);
  
  result = (4 * (N & 1) - 2) * (*p);
  
  //ENDING
  end = omp_get_wtime();
  
  cout << "Threads:" << TOTAL << "\tResult:" << result << "\tTime:" << end - start << "s" << "\tTotal Time:"<< end - initt << endl;
  //cout << TOTAL << "," << result << "," << end - start << "\n";
  
  for(int i = 0; i < N; i++)
    free(M[i]);

  free(p);
  free(x);
  free(sumCol);
  free(lastCol);
  cudaFree(xx);
  cudaFree(pCuda);
  cudaFree(MCuda);
  free(M);
  free(Mrow);
  
  return 0;
}

      
