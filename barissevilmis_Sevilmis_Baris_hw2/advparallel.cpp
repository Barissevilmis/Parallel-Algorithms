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

using namespace std;

void usage()
{
  cout << "USAGE: ./exec <filename>" << endl;
  exit(0);
}
void ConvertToCRS(int * &target_ind, int * &start_ind, const vector<vector<int> > &adj_list, int RS)
{
  //Initialize CRS format
  //First node maybe 0 or 1
  //Last node maybe RS-1 or RS
  //Therefore, make sure of target_ind and start_ind initialization
  int ii = 0;
  int init = 0;
  //Turn into CRS format
  //adj_list useless after this point if CRS will be used!!!
  for(ii; ii < RS ; ii++)
    {
      start_ind[ii] = init;
      int siz = adj_list[ii].size();
      int where = 0;
      for(int jj = init; jj < init + siz; jj++)
	target_ind[jj] = adj_list[ii][where++];
	
      init += siz;
    }
  start_ind[RS-1] = init;
}

void check(int * &target_ind, int * &start_ind, int * &colors, int RS, int & max)
{
  for(int ii = 0; ii < RS; ii++)
    {
      max = colors[ii] > max ? colors[ii] : max;
      for(int jj = start_ind[ii]; jj < start_ind[ii+1]; jj++)
	{
	  if(colors[ii] == colors[target_ind[jj]])
	    {
	      printf("conflict - %d - %d - %d\n", ii, target_ind[jj], colors[ii]);
	    }
	}
    }
}

void GraphColoring(int * &colors, int * &U, int ** &forbidden, int * &target_ind, int * &start_ind, int RS, int t, int REM)
{
  while(REM != 0)
    {
      //cout << "Remaining " << REM << endl;
#pragma omp parallel num_threads(t) proc_bind(spread)
      {
	int *forbiddenThread = forbidden[omp_get_thread_num()];
	forbiddenThread = (int*)(malloc(sizeof(int)*RS));
	//memset(forbiddenThread, 0, sizeof(int)*RS);
#pragma omp for schedule(guided, 32)
	for(int ii = 0; ii < REM; ii++)
	  {
	    int curr = U[ii];
	    int beg = start_ind[curr];
	    int end = start_ind[curr+1];
	    int color;
	    for(int jj = beg; jj < end; jj++)
	      {
		color = colors[target_ind[jj]];
		if(color > 0)
		  {
		    forbiddenThread[color] = curr;
		  }
	      }
	    for(int kk = 1; kk < RS; kk++) {
	      if(forbiddenThread[kk] != curr)
		{
		  colors[curr] = kk;
		  break;
		}
	    }
	  }
      }

      int conflict = 0;
      if(t > 1)
	{
	  int *R = (int*)(malloc(sizeof(int)*REM));
	  memset(R, 0, sizeof(int)*REM);
#pragma omp parallel for schedule(guided, 32) proc_bind(spread)
	for(int ii = 0; ii < REM; ii++)
	  {
	    int curr = U[ii];
	    int beg = start_ind[curr];
	    int end = start_ind[curr+1];
	    int color, ind;
	    for(int jj = beg; jj < end; jj++)
	      {
		ind = target_ind[jj];
		color = colors[ind];
		if(colors[curr] == color && curr > ind)
		  {
#pragma omp critical
		    {
		      R[conflict++] = curr;
		    }
		    colors[curr] = 0;
		    break;
		  }
	      } 
	  }
	U = R;
	}
      REM = conflict;
    }
}

int main(int argc, const char** argv)
{

  if(argc != 2)
    usage();

  string line;
  //See which node you are on by index+min
  
  const char* filename = argv[1];
  ifstream input (filename);
  if(input.fail())
    return 0;

  vector<vector<int> > adj_list;
  unsigned int mini = 1000000;
  int fctr = 0;
  //Find first node
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
  int NNS, RS, CS;
  int ctr = 0;
  while(getline(input,line))
    {
      stringstream ss(line);
      int temp, temp2, temp3;
      if(line.find('%') != string::npos)
	continue;
      //First line containing row number, column number and NNS
      else if(!ctr)
	{
	  ss >> temp >> temp2 >> temp3;
	  RS = temp+1;
	  CS = temp2;
	  NNS = temp3;
	  adj_list.resize(RS);
	  
	  ctr++;
	}
      //Lines containing start and end of an edge
      else if(ctr)
	{
	  ss >> temp >> temp2;
	  if(temp != temp2)
	    {
	      adj_list[temp-mini].push_back(temp2-mini);
	      adj_list[temp2-mini].push_back(temp-mini);
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
  int **forbidden = (int**)(malloc(sizeof(int*)));
  int *target_ind = (int*)(malloc(sizeof(int)*NNS));
  int *start_ind = (int*)(malloc(sizeof(int)*RS));
  int *color = (int*)(malloc(sizeof(int)*RS));
  memset(target_ind, 0, sizeof(int) * NNS);
  memset(start_ind, 0, sizeof(int)* RS);
  memset(color, 0, sizeof(int) * RS);

  ConvertToCRS(target_ind, start_ind, adj_list, RS);
  for(int i = 0; i < adj_list.size();i++)
    adj_list[i].clear();
  adj_list.clear();
  int res;
  int max = -1;
  int REM = RS;
  int t = 1;
  //printf("Starting Coloring\n");
  double start, end;
  for(t = 1; t <= 16; t=2*t)
    {  
      int *U = (int*)(malloc(sizeof(int)*RS));
      forbidden = (int**)(realloc(forbidden,sizeof(int*)*t));
      for(int u = 0; u < RS; u++)
	U[u] = u;
      //Graph Coloring//
      start = omp_get_wtime();
      GraphColoring(color, U, forbidden, target_ind, start_ind, RS, t, REM);
      end = omp_get_wtime();
      //////////////////
      check(target_ind, start_ind, color, RS, max);
      cout << "Threads: " << t << "\tColors:" << max << "\tTime:" << end - start << " s" << endl;
      //cout << t << "," << max << "," << end - start << "\n";
      max = -1;
      memset(color, 0, sizeof(int) * RS);
      free(U);
    }
  free(forbidden);
  free(target_ind);
  free(start_ind);
  free(color);
  return 0;
}
