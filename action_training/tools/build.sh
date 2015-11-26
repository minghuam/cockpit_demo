#g++ -std=c++11 calc_flow.cpp  -o calc_flow -lPocoFoundation -L/usr/local/cuda/lib64 `pkg-config --cflags opencv --libs opencv` 

g++ -std=c++11 calc_flow_folder.cpp directory.cpp  -o calc_flow_folder -lPocoFoundation -L/usr/local/cuda/lib64 `pkg-config --cflags opencv --libs opencv` 
