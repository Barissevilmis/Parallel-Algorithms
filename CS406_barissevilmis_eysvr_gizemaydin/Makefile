all: OMP OMPHYBRID CUDA CUDAFRONT CUDAHYBRID CUDAHYBFRONT

OMP: OpenMP_TD.cpp
	g++ -o omp OpenMP_TD.cpp -O3 -fopenmp -std=c++11

OMPHYBRID: OpenMP_Hybrid.cpp
	g++ -o omphyb OpenMP_Hybrid.cpp -O3 -fopenmp -std=c++11

CUDA: CUDA_Level_TD.cu
	nvcc -o cuda CUDA_Level_TD.cu -O3 -Xcompiler -fopenmp -Xcompiler -O3

CUDAFRONT: CUDA_Front_TD.cu
	nvcc -o cudafront CUDA_Front_TD.cu -O3 -Xcompiler -fopenmp -Xcompiler -O3

CUDAHYBRID: CUDA_Level_Hybrid.cu
	nvcc -o cudahyb CUDA_Level_Hybrid.cu -O3 -Xcompiler -fopenmp -Xcompiler -O3

CUDAHYBFRONT: CUDA_Front_Hybrid.cu
	nvcc -o cudahybfront CUDA_Front_Hybrid.cu -O3 -Xcompiler -fopenmp -Xcompiler -O3	
clear:
	rm -f omp
	rm -f cuda
	rm -f omphyb
	rm -f cudahyb
	rm -f cudafront
	rm -f cudahybfront
