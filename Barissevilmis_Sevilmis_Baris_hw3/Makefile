all: CUDA ADVCUDA SUPERCUDA NEWCUDA FL1 FL2 FL3

FL1: sharedFloat.cu
	nvcc -o sharedfloat sharedFloat.cu -O3 -arch=sm_61 -Xcompiler -fopenmp -Xcompiler -O3

FL2: coalFloat.cu
	nvcc -o coalfloat coalfloat.cu -O3 -arch=sm_61 -Xcompiler -fopenmp -Xcompiler -O3

FL3: hybridFloat.cu
	nvcc -o hybfloat hybridFloat.cu -O3 -arch=sm_61 -Xcompiler -fopenmp -Xcompiler -O3

CUDA: naive.cu
	nvcc -o naive naive.cu -O3 -arch=sm_61 -Xcompiler -fopenmp -Xcompiler -O3

ADVCUDA: sharedMem.cu
	nvcc -o sharedMem sharedMem.cu -O3 -arch=sm_61 -Xcompiler -fopenmp -Xcompiler -O3

SUPERCUDA: coalescing.cu
	nvcc -o coalescing coalescing.cu -O3 -arch=sm_61 -Xcompiler -fopenmp -Xcompiler -O3

NEWCUDA: sharedMem_Coal.cu
	nvcc -o sharedMemCoal sharedMem_Coal.cu -O3 -arch=sm_61 -Xcompiler -fopenmp -Xcompiler -O3

clear:
	rm -f naive
	rm -f sharedMem
	rm -f coalescing
	rm -f sharedMemCoal
	rm -f coalfloat
	rm -f hybfloat
	rm -f sharedfloat 
