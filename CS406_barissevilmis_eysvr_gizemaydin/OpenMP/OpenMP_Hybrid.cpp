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
#include <boost/dynamic_bitset.hpp>

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
  start_ind[RS] = init;
}

  
void BFS_Top_Down(int** & tempArr, int* & target_ind, int* & start_ind, int* & results , int* & front, int* & prefixSum, int & size, unsigned int startNode, unsigned int RS, int t)
{
  int ctr = 0;
#pragma omp parallel num_threads(t)
  {
    int* temp = tempArr[omp_get_thread_num()];
    int tempCtr = 0;
    int sum = 0;
    int i, v, end;
#pragma omp for reduction(+:ctr) schedule(guided)
    for(int j = 0; j < size; j++)
      {
	v = front[j];
	i = start_ind[v];
	end = start_ind[v+1];
	for(i; i < end; i++)
	  {
	    int index = target_ind[i];
	    if(results[index] == -1)
	      {
		results[index] = results[v]+1;
		temp[tempCtr++] = index;
		ctr++;
	      }
	  }
      }
    int tid = omp_get_thread_num();
    prefixSum[tid] = tempCtr;
#pragma omp barrier
    size = ctr;
    for(int m = 0; m < tid; m++)
      sum += prefixSum[m];
    for(int k = 0 ; k < tempCtr; k++)
      front[sum++] = temp[k];
  }
}
void BFS_Bottom_Up(boost::dynamic_bitset<> &front_check, int** & tempArr, int* & target_ind, int* & start_ind, int* & results , int* & front, int* & prefixSum, int & size, unsigned int startNode, unsigned int RS, int t)
{
  //can replace front with a vertex sized array, where if a node is present fron[node] =1 if not = 0. This way checking whether a node is in the front goes from O(frontsize) to O(1).
  //and since checking if front is done for every unvisited vertice, this may be very useful.
  int ctr = 0;
#pragma omp parallel num_threads(t)
  {
    int* temp = tempArr[omp_get_thread_num()];
    int tempCtr = 0;
    int sum = 0;
#pragma omp for reduction(+:ctr) schedule(guided)
    for(int ii = 0; ii < RS; ii++)
      {
	int st, end;
	if(results[ii] == -1)//check if it has any distance(parent)
	  {
	    st = start_ind[ii];
	    end = start_ind[ii+1];
	    bool raviolli = false;
	    for(int jj = st; jj < end && raviolli == false; jj++)
	      {
		int target = target_ind[jj];//target is a neighbor of v
		if(front_check[target])//if
		  {
		    results[ii] = results[target] + 1;
		    temp[tempCtr++] = ii;
		    ctr++;
		    raviolli = true;
		  }
	      }
	  }
      }
    int tid = omp_get_thread_num();
    prefixSum[tid] = tempCtr;
#pragma omp barrier
    size = ctr;
    for(int m = 0; m < tid; m++)
      sum += prefixSum[m];
    for(int k = 0 ; k < tempCtr; k++)
      front[sum++] = temp[k];

  }

}
void BFS_Bottom_Up_Directed(boost::dynamic_bitset<> &front_check, int** & tempArr, int* & target_ind, int* & start_ind, int* & target_rev, int* & start_rev, int* & results , int* & front, int* & prefixSum, int & size, unsigned int startNode, unsigned int RS, int t)
{
  //can replace front with a vertex sized array, where if a node is present fron[node] =1 if not = 0. This way checking whether a node is in the front goes from O(frontsize) to O(1).
  //and since checking if front is done for every unvisited vertice, this may be very useful.
  int ctr = 0;
#pragma omp parallel num_threads(t)
  {
    int* temp = tempArr[omp_get_thread_num()];
    int tempCtr = 0;
    int sum = 0;
#pragma omp for reduction(+:ctr) schedule(guided)
    for(int ii = 0; ii < RS; ii++)
      {
	int st, end;
	if(results[ii] == -1)//check if it has any distance(parent)
	  {
	    st = start_rev[ii];
	    end = start_rev[ii+1];
	    bool raviolli = false;
	    for(int jj = st; jj < end && raviolli == false; jj++)
	      {
		int target = target_rev[jj];//target is a neighbor of v
		if(front_check[target])//if
		  {
		    results[ii] = results[target] + 1;
		    temp[tempCtr++] = ii;
		    ctr++;
		    raviolli = true;
		  }
	      }
	  }
      }
    int tid = omp_get_thread_num();
    prefixSum[tid] = tempCtr;
#pragma omp barrier
    size = ctr;
    for(int m = 0; m < tid; m++)
      sum += prefixSum[m];
    for(int k = 0 ; k < tempCtr; k++)
      front[sum++] = temp[k];

  }

}



void BFS(int** & tempArr, int* & target_ind, int* & start_ind, int* & results, int* & prefixSum, boost::dynamic_bitset<> & front_check, unsigned int RS, unsigned int startNode, int t)
{
  int* mrQ = (int*)(malloc(sizeof(int)*RS));
  int size = 1;
  int switchCtr = RS/32;
  results[startNode] = 0;
  mrQ[0] = startNode;
  double start, end;
  start = omp_get_wtime();
  while(size)
    {
      if(size < switchCtr)
	{
	  BFS_Top_Down(tempArr, target_ind, start_ind, results, mrQ, prefixSum, size,  startNode, RS, t);
	  //printf("Top Down Phase!\n");
	}
      else
	{
	  front_check.reset();
	  for(int n = 0; n < size; n++)
	    {
	      front_check[mrQ[n]] = 1;
	    }
	  BFS_Bottom_Up(front_check, tempArr, target_ind, start_ind, results, mrQ, prefixSum, size,  startNode, RS, t);
	  //printf("Bottom Up Phase!\n");
	}
    }
  end = omp_get_wtime();
  cout << "Threads: " << t << "\tSource:" << startNode << "\tTime:" << end - start << " s" << endl;
}
void BFS_Directed(int** & tempArr, int* & target_ind, int* & start_ind, int* & target_rev, int* & start_rev, int* & results, int* & prefixSum, boost::dynamic_bitset<> & front_check, unsigned int RS, unsigned int startNode, int t)
{
  int* mrQ = (int*)(malloc(sizeof(int)*RS));
  int size = 1;
  int switchCtr = RS/32;
  results[startNode] = 0;
  mrQ[0] = startNode;
  double start, end;
  start = omp_get_wtime();
  while(size)
    {
      if(size < switchCtr)
	{
	  BFS_Top_Down(tempArr, target_ind, start_ind, results, mrQ, prefixSum, size,  startNode, RS, t);
	  //printf("Top Down Phase!\n");
	}
      else
	{
	  front_check.reset();
	  for(int n = 0; n < size; n++)
	    {
	      front_check[mrQ[n]] = 1;
	    }
	  BFS_Bottom_Up_Directed(front_check, tempArr, target_ind, start_ind, target_rev, start_rev, results, mrQ, prefixSum, size,  startNode, RS, t);
	  //printf("Bottom Up Phase!\n");
	}
    }
  end = omp_get_wtime();
  cout << "Threads: " << t << "\tSource:" << startNode << "\tTime:" << end - start << " s" << endl;

}

int main(int argc, const char** argv)
{

  if(argc != 2)
    usage();

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
  unsigned int NNS, RS, CS, NNSrev;
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
	      CS = temp2;
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
	      CS = temp2;
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
      if(!mode)
	{
	  sort(rev_list[i].begin(), rev_list[i].end());
	  rev_list[i].erase(unique(rev_list[i].begin(), rev_list[i].end()), rev_list[i].end());
	  NNSrev += rev_list[i].size();
	}
      NNS += adj_list[i].size();    
    }

  int* target_ind = (int*)(malloc(sizeof(int)*NNS));
  int* start_ind = (int*)(malloc(sizeof(int)*(RS+1)));
  int* results = (int*)(malloc(sizeof(int)*RS));
  boost::dynamic_bitset<> front_check(RS);
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
  //Change min to startNode later
  for(int t = 1; t <= 16; t=2*t)
    {
      int* prefixSum = (int*)(malloc(sizeof(int)*t));
      int** tempArr = (int**)(malloc(sizeof(int*)*t));
      for(int j = 0; j < t; j++)
	{
	  tempArr[j] = (int*)(malloc(sizeof(int)*RS));
	  memset(tempArr[j], 0, sizeof(int)*RS);
	}
      memset(prefixSum, 0, sizeof(int)*t);
      memset(results, -1, sizeof(int)*RS); 

      if(mode)
	{
	  BFS(tempArr, target_ind, start_ind, results, prefixSum, front_check, RS, startNode, t);
	}
      
      else
	{
	  BFS_Directed(tempArr, target_ind, start_ind, target_rev, start_rev, results, prefixSum, front_check, RS, startNode, t);
	}
      for(int j = 0; j < t; j++)
	free(tempArr[j]);
      free(tempArr);
      free(prefixSum);
    }

  ofstream myfile;
  myfile.open("hybridresults.txt");
  for(int i = 0; i < RS; i++)
    myfile<< results[i] <<"\n";
  myfile.close();

  if(!mode)
    {
      free(target_rev);
      free(start_rev);
    }
  free(target_ind);
  free(start_ind);
  free(results);
  
  return 0;
}
