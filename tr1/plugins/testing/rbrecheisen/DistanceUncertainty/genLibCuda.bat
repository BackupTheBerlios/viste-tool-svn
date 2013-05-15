echo "Building and installing CUDA library..."
nvcc -o DistanceUncertaintyCUDA.lib -g --machine 32 -lib PBA3D\pba\pba3DHost.cu
copy DistanceUncertaintyCUDA.lib d:\applications\dtitool\plugins\testing\Ralph\DistanceUncertainty\Debug