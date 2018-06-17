

#include <iostream>
#include <assert.h>
#include <boost/lexical_cast.hpp>
#include <csignal>
#include <unistd.h>
using namespace std;

char* gpu_mem = NULL;

void term(int sig)
{
    cout << "Received signal number: " << sig << endl;
    if(sig==15){
        cudaError_t err=cudaSuccess;
        if(gpu_mem) 
            err = cudaFree(gpu_mem); 	
        assert(err==cudaSuccess);
        exit(0);
    }
 
}

void wait_forever(){
    //wait
    while(true){
     sleep(1);
    } 
}

int main(int argc, char* argv[]){

 if(argc<2){
     cout << "Missing parameter!" << endl;
     cout << argv[0] << " <number of bytes to allocate on GPU>" << endl;
     return -1;
 }

 if(argc>2){
     cout << "To many parameters!" << endl;
     cout << argv[0] << " <number of bytes to allocate on GPU>" << endl;
     return -1;
 }

 signal(SIGTERM, term); // register a SIGTERM handler

 size_t size_in_bytes = boost::lexical_cast<size_t>(argv[1]);

 cudaError_t err = cudaMalloc((void**)&gpu_mem, size_in_bytes);
 assert(err==cudaSuccess);

 wait_forever();

 return 0;
}
