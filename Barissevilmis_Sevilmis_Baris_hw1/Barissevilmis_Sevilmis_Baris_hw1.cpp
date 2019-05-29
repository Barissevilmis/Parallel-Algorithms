#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>
using namespace std;


void usage()
{
  cout << "USAGE: ./exec <filename>" << endl;
  exit(0);
}

int main(int argc, const char** argv)
{

  if(argc != 2)
    usage();

  string line;

  const char* filename = argv[1];
  ifstream input (filename);
  if(input.fail())
    return 0;

  int N;
  int **M;
  
  getline(input,line);
  N = atoi(line.c_str());

  M = (int**)(malloc(sizeof(int*)*N));
  for(int i = 0; i < N; i ++)
    {
      M[i] = (int*)(malloc(sizeof(int)*N));;
      memset(M[i], 0, N*sizeof(int));
    }

  int linectr = 0;
  while(getline(input,line)){
    stringstream ss(line);
    int temp;
    int ctr = 0;
    while(ss >> temp)
      M[linectr][ctr++] = temp;


    linectr++;
  }

  double start, end;
  for(int t = 1; t <=16; t*=2) { //t is the number of threads
    start = omp_get_wtime();
  ////YOUR CODE GOES HERE
    long long int K;
    unsigned long result = 0;
    long double p = 1;
    //long double temp_p = 1;
    K = 1 << N-1;
    long long int CHUNK = K/t;

    double *x = (double*)(malloc(sizeof(double)*N));
    double *sumCol = (double*)(malloc(sizeof(double)*N));
    double *lastCol = (double*)(malloc(sizeof(double)*N));
    memset(sumCol, 0, N*sizeof(double));
    memset(lastCol, 0, N*sizeof(double));
    
    int temp_i,temp_y,yy, y, y_prev, z, prodSign, s, id, BPC, CTZ;
    for(int i = 0; i< N; i++)
      {
	lastCol[i] = M[i][N-1];
	for(int j = 0; j <N; j++)
	  sumCol[i] += M[i][j];
	x[i] = (lastCol[i] - sumCol[i]/2);
	p *= x[i];
      }
    omp_set_num_threads(t);
#pragma omp parallel private(BPC,CTZ,temp_i,temp_y,id,yy) proc_bind(spread)
    {
      int ** M_new = (int**)(malloc(sizeof(int*)*N));
      for(int i = 0; i < N; i++)
	{
	  M_new[i] = (int*)(malloc(sizeof(int)*N));
	  memset(M_new[i], 0, N*sizeof(int));
	  for(int j = 0; j < N; j++)
	    M_new[i][j] = M[j][i];
	} 
      double *x_new = (double*)(malloc(sizeof(double)*N));
      memset(x_new,0,N*sizeof(double));
      id = omp_get_thread_num();
      temp_i = (CHUNK*id);
      yy = (temp_i ^ (temp_i >> 1));
      for(int m = 0; m<N; m++)
	{
	  x_new[m] = x[m];
	  BPC = __builtin_popcount(yy);
	  temp_y = yy;
	  for(int n = 0; n < BPC; n++)
	    {
	      CTZ = __builtin_ctz(temp_y);
	      temp_y &= ~(1 << CTZ);
	      x_new[m] += (double)(M_new[CTZ][m]);
	    }
	}

#pragma omp for reduction(+:p) schedule(static) private(y,y_prev,z,s,prodSign)
    for(unsigned long long int m = 1; m < K ; m++)
      {
	y = (m ^ (m >> 1));
	y_prev = (m-1) ^ ((m-1) >> 1);
	prodSign = (m & 1) ? -1 : 1;
	//z = log2((double)(y ^ ((m-1) ^ ((m-1) >> 1))));
	//z = 31 - __builtin_clz(y^y_prev);
	z = __builtin_ctz(y^y_prev);
	s = ((y >> z) & 1) ? 1 : -1;
	double temp = 1;
#pragma omp simd reduction(*:temp)
	for(int n = 0; n < N; n++)	  
	  {
	    x_new[n] += (double)(s * M_new[z][n]);
	    temp *= x_new[n];
	  }
	p += (prodSign*temp);
      }
    free(x_new);
    for(int i = 0; i < N; i++)
      free(M_new[i]);
    free(M_new);
    }
    result = (4 * (N & 1) - 2) * p;

    free(x);
    free(sumCol);
    free(lastCol);
    //// YOUR CODE ENDS HERE
    end = omp_get_wtime();

    cout << "Threads:" << t << "\tResult:" << result << "\tTime:" << end - start << "s" << endl;
    //cout << t << "," << result << "," << end - start << "\n";
    
  }
  for(int i = 0; i < N; i++)
    free(M[i]);
  free(M);
   
  return 0;
}
      
