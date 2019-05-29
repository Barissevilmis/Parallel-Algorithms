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
void BFS_Top_Down(int* target_ind, int* start_ind, int* results, int* frontier, int* frontsize, int* newfrontier, int* newfrontsize, int RS)
{
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(p < (*frontsize))
    {
      int index = frontier[p];
      int start_loc = start_ind[index];
      int end_loc = start_ind[index + 1];
      int curr = results[index]; 
      for(int ii = start_loc; ii < end_loc; ii++)
	{
	  if(results[target_ind[ii]] == -1)
	    {
	      results[target_ind[ii]] = curr + 1;
	      
	      int cc = atomicAdd(newfrontsize,1);
	      newfrontier[cc] = target_ind[ii];
	      
	    }
	}
    }
}

__global__
void FRONT_COPY(int* front, int* newfront, int* newfrontsize)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if(ii < (*newfrontsize))
    {
      int index = newfront[ii];
      front[ii] = index;
    }
  
}

void BFS(int* & target_ind, int* & start_ind, int* & results, unsigned int RS, unsigned int NNS, unsigned int startNode)
{
  results[startNode] = 0;
  int *res;
  int *target_arr;
  int *start_arr;
  checkCuda(cudaMalloc((void **)&res, sizeof(int)*RS));
  checkCuda(cudaMalloc((void **)&target_arr, sizeof(int)*NNS));
  checkCuda(cudaMalloc((void **)&start_arr, sizeof(int)*(RS+1)));
  //printf("Device Variable Copying:\t%s\n", cudaGetErrorString(cudaGetLastError()));

  int *size = (int*)(malloc(sizeof(int)));
  int *zero = (int*)(malloc(sizeof(int)));
  int *frontbeg = (int*)(malloc(sizeof(int)*RS));
  unsigned int CHUNK_SIZE = 1024;
  //unsigned int LIMIT = (RS+CHUNK_SIZE-1) / CHUNK_SIZE;
  dim3 GRID(1);
  dim3 BLOCK(CHUNK_SIZE);
  *size = 1;
  *zero = 0;
  memset(frontbeg, startNode, sizeof(int)*RS);
  
  //cudaMallocHost(&size,sizeof(int));
  int* frontsize;
  int* frontier;
  int* newfrontsize;
  int* newfrontier;
  int* tempptr;
  checkCuda(cudaMalloc((void**)&frontier,sizeof(int)*RS));
  checkCuda(cudaMalloc((void**)&frontsize,sizeof(int)));
  checkCuda(cudaMalloc((void**)&newfrontier,sizeof(int)*RS));
  checkCuda(cudaMalloc((void**)&newfrontsize,sizeof(int)));
  
  checkCuda(cudaMemcpy(frontier,frontbeg,sizeof(int)*RS, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(frontsize,size,sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(newfrontsize,zero,sizeof(int), cudaMemcpyHostToDevice));
  
  checkCuda(cudaMemcpy(res, results, sizeof(int)*RS,cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(start_arr, start_ind, sizeof(int)*(RS+1),cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(target_arr, target_ind, sizeof(int)*NNS,cudaMemcpyHostToDevice));
  
  double start = omp_get_wtime();
  while(*size > 0)
    {

      BFS_Top_Down<<<GRID,BLOCK>>>(target_arr, start_arr, res, frontier, frontsize, newfrontier, newfrontsize, RS);
      cudaDeviceSynchronize();
      cudaMemcpy(size, newfrontsize, sizeof(int),cudaMemcpyDeviceToHost);
      cudaMemcpy(frontsize, size, sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(newfrontsize, zero, sizeof(int),cudaMemcpyHostToDevice);
      tempptr = frontier;
      frontier = newfrontier;
      newfrontier = tempptr;
  
      GRID.x = (*size + 1023) / 1024;
    }
  
  double end = omp_get_wtime();
  cout << "\tSource:" << startNode << "\tTime:" << end - start << " s" << endl;
  checkCuda(cudaMemcpy(results, res, sizeof(int)*RS,cudaMemcpyDeviceToHost));

  free(size);
  free(zero);
  free(frontbeg);
  cudaFree(frontier);
  cudaFree(newfrontier);
  cudaFree(frontsize);
  cudaFree(newfrontsize);
  cudaFree(res);
  cudaFree(target_arr);
  cudaFree(start_arr);
}

int main(int argc, const char** argv){

  if(argc != 3)
    usage();

  
  int devId = 0;
  devId = atoi(argv[2]); 
  cudaSetDevice(devId); 
  string line;
  vector<vector<int> > adj_list;
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
  unsigned int NNS, RS;
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
	      RS = temp;
	      ctr++;
	    }
	  //Lines containing start and end of an edge
	  else if(ctr)
	    {
	      ss >> temp >> temp2;
	      if(temp != temp2)
		adj_list[temp - mini].push_back(temp2 - mini);
	    }
	}
    }

  //Remove duplicates
  NNS = 0;
  for(int i = 0; i < adj_list.size(); i++)
    {
      sort(adj_list[i].begin(), adj_list[i].end());
      adj_list[i].erase(unique(adj_list[i].begin(), adj_list[i].end()), adj_list[i].end());
      NNS += adj_list[i].size();
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
  
  //Start Node as parameter
  cout << "Graph converted to 0-base(Initial node is 0)\n";
  unsigned int startNode;
  //cout << "Please enter the start node: ";
  //cin >> startNode;
  //cout << endl;
  startNode = 0;
  
  BFS(target_ind, start_ind, results, RS, NNS, startNode);

  ofstream myfile;
  myfile.open("cudafrontresults.txt");
  for(int i = 0; i < RS; i++)
    myfile<< results[i] <<"\n";
  myfile.close();
  
  cudaDeviceSynchronize();
  //cudaFreeHost(target_ind);
  //cudaFreeHost(start_ind);
  //cudaFreeHost(results);
  free(target_ind);
  free(results);
  free(start_ind);
  
  return 0;
}
