#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <stdlib.h>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <queue>
#include <algorithm>
#include <cassert>

using namespace std;

void usage()
{
  cout << "USAGE: ./exec <filename> <deviceNum>" << endl;
  exit(0);
}

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

void ConvertToCRS(int* & target_ind, int* & start_ind, const vector<vector<int> > &adj_list, int RS)
{
  int init = 0;
  for(int ii = 0; ii < RS ; ii++)
    { 
      start_ind[ii] = init;
      int siz = adj_list[ii].size();
      int where = 0;
      for(int jj = init; jj < init + siz; jj++)
	target_ind[jj] = adj_list[ii][where++];
      init += siz;
    }
  start_ind[RS] = init;
}
__global__
void INIT(int* target_ind, int* start_ind, int* results, unsigned int startNode, int* SIZE)
{

  int tid = threadIdx.x;
  int v = results[startNode];
  int st = start_ind[v];
  results[target_ind[st+tid]] = 1;
  atomicSub(SIZE,1);
  
}
__global__
void Check_BFS(int* results, int RS, int* SIZE, int* switchCtr, unsigned int TOTAL, int v)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  for(int jj = index; jj < RS; jj+=TOTAL)
    {
      if(results[jj] == -1)
	{
	  atomicAdd(SIZE,1);
	}
      else if(results[jj] == v)
	{
	  atomicAdd(switchCtr,1);
	}
    }
}

__global__
void BFS_Top_Down(int* target_ind, int* start_ind, int* results, int v, int RS, unsigned int TOTAL, int* SIZE)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  for(int jj = index; jj < RS; jj+=TOTAL)
    {
      if(results[jj] == v)
	{
	  atomicAdd(SIZE, 1);
	  int start_loc = start_ind[jj];
	  int end_loc = start_ind[jj + 1];
	  int curr = results[jj];
	  for(int ii = start_loc; ii < end_loc; ii++)
	    {
	      if(results[target_ind[ii]] == -1)
		{
		  results[target_ind[ii]] = curr + 1;
		}
	   
	    }
	}
    }
}

__global__
void BFS_Bottom_Up(int* target_ind, int* start_ind, int* results, int v, unsigned int RS, unsigned int TOTAL, int* SIZE)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  for(int jj = index; jj < RS; jj+=TOTAL)
    {
      if(results[jj] == -1)
	{
	  int start_loc = start_ind[jj];
	  int end_loc = start_ind[jj + 1];
	  int target;
	  bool raviolli = false;
	  for(int ii = start_loc; ii < end_loc && raviolli == false; ii++)
	    {
	      target = results[target_ind[ii]];
	      if(target == v)
		{
		  results[jj] = target + 1;
		  atomicAdd(SIZE, 1);
		  raviolli = true;
		}
	    }
	}
    }
}
__global__
void BFS_Bottom_Up_Directed(int* target_rev, int* start_rev, int* results, int v, unsigned int RS, unsigned int TOTAL, int* SIZE)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  for(int jj = index; jj < RS; jj+=TOTAL)
    {
      if(results[jj] == -1)
	{
	  int start_loc = start_rev[jj];
	  int end_loc = start_rev[jj + 1];
	  int target;
	  bool raviolli = false;
	  for(int ii = start_loc; ii < end_loc && raviolli == false; ii++)
	    {
	      target = results[target_rev[ii]];
	      if(target == v)
		{
		  results[jj] = target + 1;
		  atomicAdd(SIZE, 1);
		  raviolli = true;
		}
	    }
	}
    }
}
void BFS(int* & target_ind, int* & start_ind, int* & results, unsigned int RS, unsigned int NNS, unsigned int startNode)
{
  results[startNode] = 0;
  int v = 0;
  int *res;
  int *target_arr;
  int *start_arr;
  checkCuda(cudaMalloc((void **)&res, sizeof(int)*RS));
  checkCuda(cudaMalloc((void **)&target_arr, sizeof(int)*NNS));
  checkCuda(cudaMalloc((void **)&start_arr, sizeof(int)*(RS+1)));
  //printf("Device Variable Copying:\t%s\n", cudaGetErrorString(cudaGetLastError()));

  int *SIZE;
  int switchCtr = RS/32;
  int *size = (int*)(malloc(sizeof(int)));
  int *zero = (int*)(malloc(sizeof(int)));
  unsigned int CHUNK_SIZE = 1024;
  unsigned int LIMIT = (RS+CHUNK_SIZE-1) / CHUNK_SIZE;
  unsigned int TOTAL = LIMIT*CHUNK_SIZE;
  dim3 GRID(LIMIT);
  dim3 BLOCK(CHUNK_SIZE);
  *size = 1;
  *zero = 0;
  //cudaMallocHost(&size,sizeof(int));
  checkCuda(cudaMalloc((void**)&SIZE,sizeof(int)));
  checkCuda(cudaMemcpy(res, results, sizeof(int)*RS,cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(start_arr, start_ind, sizeof(int)*(RS+1),cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(target_arr, target_ind, sizeof(int)*NNS,cudaMemcpyHostToDevice));
 
  double start = omp_get_wtime();
  while(*size > 0)
    {
      cudaMemcpy(SIZE, zero, sizeof(int), cudaMemcpyHostToDevice);
      if(*size < switchCtr)
	{
	  BFS_Top_Down<<<GRID,BLOCK>>>(target_arr, start_arr, res, v, RS, TOTAL, SIZE);
	  cudaDeviceSynchronize();
	}
      else
	{
	  BFS_Bottom_Up<<<GRID,BLOCK>>>(target_arr, start_arr, res, v, RS, TOTAL, SIZE);
	  cudaDeviceSynchronize();
	}      
      //*switchCtr = 0;
      //Check_BFS<<<GRID,BLOCK>>>(res, RS, SIZE, SWITCH, TOTAL, v);
      //cudaDeviceSynchronize();
      cudaMemcpy(size, SIZE, sizeof(int),cudaMemcpyDeviceToHost);
      v++;
    }
  
  double end = omp_get_wtime();
  cout << "\tSource:" << startNode << "\tTime:" << end - start << " s" << endl;
  checkCuda(cudaMemcpy(results, res, sizeof(int)*RS,cudaMemcpyDeviceToHost));

  free(size);
  free(zero);
  cudaFree(SIZE);
  cudaFree(res);
  cudaFree(target_arr);
  cudaFree(start_arr);
}

void BFS_Directed(int* & target_ind, int* & start_ind, int* & target_rev, int* & start_rev, int* & results, unsigned int RS, unsigned int NNS, unsigned int NNSrev, unsigned int startNode)
{
  results[startNode] = 0;
  int v = 0;
  int *res;
  int *target_rev_arr;
  int *start_rev_arr;
  int *target_arr;
  int *start_arr;
  checkCuda(cudaMalloc((void **)&res, sizeof(int)*RS));
  checkCuda(cudaMalloc((void **)&target_arr, sizeof(int)*NNS));
  checkCuda(cudaMalloc((void **)&start_arr, sizeof(int)*(RS+1)));
  checkCuda(cudaMalloc((void **)&target_rev_arr, sizeof(int)*NNSrev));
  checkCuda(cudaMalloc((void **)&start_rev_arr, sizeof(int)*(RS+1)));
  //printf("Device Variable Copying:\t%s\n", cudaGetErrorString(cudaGetLastError()));

  int *SIZE;
  int switchCtr = RS/32;
  int *size = (int*)(malloc(sizeof(int)));
  int *zero = (int*)(malloc(sizeof(int)));
  unsigned int CHUNK_SIZE = 1024;
  unsigned int LIMIT = (RS+CHUNK_SIZE-1) / CHUNK_SIZE;
  unsigned int TOTAL = LIMIT*CHUNK_SIZE;
  dim3 GRID(LIMIT);
  dim3 BLOCK(CHUNK_SIZE);
  *size = 1;
  *zero = 0;
  //cudaMallocHost(&size,sizeof(int));
  checkCuda(cudaMalloc((void**)&SIZE,sizeof(int)));
  checkCuda(cudaMemcpy(res, results, sizeof(int)*RS,cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(start_arr, start_ind, sizeof(int)*(RS+1),cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(target_arr, target_ind, sizeof(int)*NNS,cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(start_rev_arr, start_rev, sizeof(int)*(RS+1),cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(target_rev_arr, target_rev, sizeof(int)*NNSrev,cudaMemcpyHostToDevice));
 
  double start = omp_get_wtime();
  while(*size > 0)
    {
      cudaMemcpy(SIZE, zero, sizeof(int), cudaMemcpyHostToDevice);
      if(*size < switchCtr)
	{
	  BFS_Top_Down<<<GRID,BLOCK>>>(target_arr, start_arr, res, v, RS, TOTAL, SIZE);
	  cudaDeviceSynchronize();
	}
      else
	{
	  BFS_Bottom_Up_Directed<<<GRID,BLOCK>>>(target_rev_arr, start_rev_arr, res, v, RS, TOTAL, SIZE);
	  cudaDeviceSynchronize();
	}      
      //*switchCtr = 0;
      //Check_BFS<<<GRID,BLOCK>>>(res, RS, SIZE, SWITCH, TOTAL, v);
      //cudaDeviceSynchronize();
      v++;
      cudaMemcpy(size, SIZE, sizeof(int),cudaMemcpyDeviceToHost);
    }
  
  double end = omp_get_wtime();
  cout << "\tSource:" << startNode << "\tTime:" << end - start << " s" << endl;
  checkCuda(cudaMemcpy(results, res, sizeof(int)*RS,cudaMemcpyDeviceToHost));

  free(size);
  free(zero);
  cudaFree(SIZE);
  cudaFree(res);
  cudaFree(target_rev_arr);
  cudaFree(start_rev_arr);
  cudaFree(target_arr);
  cudaFree(start_arr);
}


int main(int argc, const char** argv)
{

  if(argc != 3)
    usage();

  
  int devId = 0;
  devId = atoi(argv[2]); 
  cudaSetDevice(devId); 
  string line;
  vector<vector<int> > adj_list;
  vector<vector<int> > rev_list;
  //See which node you are on by index+min
  
  const char* filename = argv[1];
  
  ifstream input (filename);
  if(input.fail())
    return 0;

  //Find first node
  unsigned int mini = 10000000;
  while(getline(input,line))
    {
      stringstream sf(line);
      int temp;
      if(line.find('%') != string::npos)
	continue;
      else
	{
	  while(sf >> temp)
	    {
	      if(temp < mini)
		mini = temp;
	    }
	}
    }
  //Set to begin
  input.clear();
  input.seekg(0, ios::beg);
  int ctr = 0;
  bool mode = false;
  unsigned int NNS, RS, NNSrev;
  while(getline(input,line))
    {
      stringstream ss(line);
      int temp, temp2, temp3;
      if(line.find("%%") != string::npos && line.find(" symmetric") != string::npos)
	mode = true;
      else if(mode)
	{
	  if(line.find('%') != string::npos)
	    continue;
	  //First line containing row number, column number and NNS
	  else if(!ctr)
	    {
	      ss >> temp >> temp2 >> temp3;
	      NNS = temp3;
	      adj_list.resize(temp);
	      RS = temp;
	      ctr++;
	    }
      //Lines containing start and end of an edge
	  else if(ctr)
	    {
	      ss >> temp >> temp2;
	      if(temp != temp2)
		{
		  adj_list[temp - mini].push_back(temp2 - mini);
		  adj_list[temp2 - mini].push_back(temp - mini);
		}
	    }
	}
      //Get past comment lines
      else if(!mode)
	{
	  if(line.find('%') != string::npos)
	    continue;
	  //First line containing row number, column number and NNS
	  else if(!ctr)
	    {
	      ss >> temp >> temp2 >> temp3;
	      NNS = temp3;
	      adj_list.resize(temp);
	      rev_list.resize(temp);
	      RS = temp;
	      ctr++;
	    }
	  //Lines containing start and end of an edge
	  else if(ctr)
	    {
	      ss >> temp >> temp2;
	      if(temp != temp2)
		{
		  adj_list[temp - mini].push_back(temp2 - mini);
		  rev_list[temp2 - mini].push_back(temp - mini);
		}
	    }
	}
    }

  //Remove duplicates
  NNS = 0;
  NNSrev = 0;
  for(int i = 0; i < adj_list.size(); i++)
    {
      sort(adj_list[i].begin(), adj_list[i].end());
      adj_list[i].erase(unique(adj_list[i].begin(), adj_list[i].end()), adj_list[i].end());
      NNS += adj_list[i].size();
      if(!mode)
	{
	  sort(rev_list[i].begin(), rev_list[i].end());
	  rev_list[i].erase(unique(rev_list[i].begin(), rev_list[i].end()), rev_list[i].end());
	  NNSrev += rev_list[i].size();
	}
      
    }
  
  int* target_ind = (int*)(malloc(sizeof(int)*NNS));
  //cudaMallocHost(&target_ind, sizeof(int)*NNS);
  int* start_ind = (int*)(malloc(sizeof(int)*(RS+1)));
  //cudaMallocHost(&start_ind, sizeof(int)*(RS+1));
  int* results = (int*)(malloc(sizeof(int)*RS));
  //cudaMallocHost(&results, sizeof(int)*RS);
 
  memset(target_ind, 0, sizeof(int)* NNS);
  memset(start_ind, 0, sizeof(int)* (RS+1));
  memset(results, -1, sizeof(int)* RS);
  
  ConvertToCRS(target_ind, start_ind, adj_list, RS);

  int* target_rev;
  int* start_rev;
  if(!mode)
    {
      target_rev = (int*)(malloc(sizeof(int)*NNSrev));
      start_rev = (int*)(malloc(sizeof(int)*(RS+1)));
      memset(target_rev, 0, sizeof(int)* NNSrev);
      memset(start_rev, 0, sizeof(int)* (RS+1));
      ConvertToCRS(target_rev, start_rev, rev_list, RS);
    }
  
  
  //Start Node as parameter
  cout << "Graph converted to 0-base(Initial node is 0)\n";
  unsigned int startNode;
  //cout << "Please enter the start node: ";
  //cin >> startNode;
  //cout << endl;
  startNode = 0;

  if(mode)
    BFS(target_ind, start_ind, results, RS, NNS, startNode);
  else
    BFS_Directed(target_ind, start_ind, target_rev, start_rev, results, RS, NNS, NNSrev, startNode);

  ofstream myfile;
  myfile.open("hybridcudaresults.txt");
  for(int i = 0; i < RS; i++)
    myfile<< results[i] <<"\n";
  myfile.close();
  
  cudaDeviceSynchronize();
  //cudaFreeHost(target_ind);
  //cudaFreeHost(start_ind);
  //cudaFreeHost(results);
  if(!mode)
    {
      free(target_rev);
      free(start_rev);
    }
  free(target_ind);
  free(results);
  free(start_ind);
  
  return 0;
}
