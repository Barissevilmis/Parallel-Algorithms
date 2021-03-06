#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>
#include <cassert>
using namespace std;


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
void PreProcess(double* xCuda, double* xx, int N)
{
  long long int ind = ((blockDim.x * blockIdx.x + threadIdx.x) * N);
  double* my_xCuda = xCuda + ind;
  
  for(int j = 0; j < N; j++)
    my_xCuda[j] = xx[j];
}
__global__
void PermanentCalculator(long long int K, double* xCuda, double* pCuda, int* my_MCuda, long long int CHUNK, long long int N)
{
  long long int index = ((blockDim.x * blockIdx.x + threadIdx.x) * CHUNK);

  if(index < K)
    {
      long long int ind = ((blockDim.x * blockIdx.x + threadIdx.x) * N);
      long long int START = index + 1;
      long long int END = index + CHUNK + 1;
      long long int yy = index ^ (index >> 1LL);
      long long int y, y_prev;
      long long int FFS, z;
      double s, prodSign;
      double* my_xCuda = xCuda + ind;
      double pSelf = 0.0;
      
      long long int temp_y = yy;      
      long long int BPC = __popcll(yy);
      for(int n = 0; n < BPC; n++)
	{
	  FFS = __ffsll(temp_y) - 1;
	  temp_y &= ~(1LL << FFS);
	  for(int m = 0; m < N; m++)
	    {
	      my_xCuda[m] += my_MCuda[m + (FFS*N)];
	    }
	}
      prodSign = ((index + 1) & 1LL) ? -1.0 : 1.0;
      for(long long int ii = START; (ii < END) && (ii < K); ii++)
	{
	  y = (ii ^ (ii >> 1LL));
	  y_prev = (ii - 1) ^ ((ii - 1) >> 1LL);
	  z = __ffsll(y ^ y_prev) - 1;      
	  s = ((y >> z) & 1LL) ? 1.0 : -1.0;
	  double temp = 1.0;
	  for(int jj = 0; jj < N; jj++)
	    {
	      my_xCuda[jj] += (s * my_MCuda[jj + (z * N)]); 
	      temp *= my_xCuda[jj];
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
  
  long long int N;
  long long int Nsqr;
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
  long long int K;
  K = 1LL << (N-1);
  double result = 0;
  
  double *p = (double*)(malloc(sizeof(double)));
  *p = 1.0;

  double *x = (double*)(malloc(sizeof(double)*N));
  double *sumCol = (double*)(malloc(sizeof(double)*N)); 
  double *lastCol = (double*)(malloc(sizeof(double)*N));
  memset(sumCol, 0, sizeof(double)*N);
  memset(lastCol, 0, sizeof(double)*N);

  //Decide on size given K
  //dim3 GRID(1024);
  //dim3 BLOCK(32);
  //long long int TOTAL = BLOCK.x*GRID.x;
  long long int GRID = 1024;
  long long int BLOCK = 128;
  long long int TOTAL = BLOCK * GRID;
  
  long long int CHUNK = (K + (TOTAL-1)) / TOTAL;
  long long int xSIZE = TOTAL * N;

  double *pCuda;  
  double *xCuda;
  double *xx;
  cudaMalloc((void**)&pCuda, sizeof(double));
  cudaMalloc((void**)&xx, sizeof(double)*N);
  cudaMalloc((void**)&xCuda, sizeof(double)*xSIZE);
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
      
  cudaMemset(xCuda, 0, sizeof(double)* xSIZE);
  cudaMemcpy(xx, x, sizeof(double)* N, cudaMemcpyHostToDevice);
  cudaMemcpy(pCuda, p, sizeof(double), cudaMemcpyHostToDevice);
  
  start = omp_get_wtime();
  PreProcess<<<GRID,BLOCK>>>(xCuda, xx, N);
  cudaDeviceSynchronize();
  PermanentCalculator<<<GRID,BLOCK>>>(K, xCuda, pCuda, MCuda, CHUNK, N);
  cudaDeviceSynchronize();
  cudaMemcpy(p, pCuda, sizeof(double), cudaMemcpyDeviceToHost);
  
  result = (4 * (N & 1) - 2) * (*p);
  
  //ENDING
  end = omp_get_wtime();
  
  //cout << "Threads:" << TOTAL << "\tResult:" << result << "\tTime:" << end - start << "s" << "\tTotal Time:"<< end - initt << endl;
  cout << TOTAL << "," << result << "," << end - initt << "\n";
  
  for(int i = 0; i < N; i++)
    free(M[i]);

  free(p);
  free(x);
  free(sumCol);
  free(lastCol);
  cudaFree(xx);
  cudaFree(xCuda);
  cudaFree(pCuda);
  cudaFree(MCuda);
  free(M);
  free(Mrow);
  
  return 0;
}

      
