echo "Installing packages required by CoGaDB..."

#for Ubuntu 14.04, we require at least llvm and clang in version 3.6
LLVM_SOURCES="clang-3.6 libclang-3.6-dev libclang-common-3.6-dev llvm-3.6 llvm-3.6-dev llvm-3.6-examples llvm-3.6-runtime llvm-3.6-tools"
OPENCL_SOURCES="ocl-icd-libopencl1 ocl-icd-opencl-dev"

#LLVM Sources
sudo apt-get install clang-3.6 libclang-3.6-dev libclang-common-3.6-dev llvm-3.6 llvm-3.6-dev llvm-3.6-examples llvm-3.6-runtime llvm-3.6-tools
#OpenCL
sudo apt-get install ocl-icd-libopencl1 ocl-icd-opencl-dev

#core tools and libraries
sudo apt-get install gcc g++ astyle highlight make cmake flex libtbb-dev libreadline6 libreadline6-dev doxygen doxygen-gui graphviz xsltproc libxslt1-dev libnuma1 libnuma-dev libsparsehash-dev sharutils bison libsparsehash-dev netcat-openbsd libbam-dev zlib1g zlib1g-dev libboost-filesystem-dev libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-serialization-dev libboost-chrono-dev libboost-date-time-dev libboost-random-dev libboost-iostreams-dev libboost-log-dev

#latex packages required for building manual and plotting experimental results
sudo apt-get install texlive-base texlive-binaries texlive-extra-utils texlive-font-utils texlive-fonts-recommended texlive-fonts-recommended-doc texlive-generic-recommended texlive-lang-german texlive-latex-base texlive-latex-base-doc texlive-latex-extra texlive-latex-extra-doc texlive-latex-recommended texlive-latex-recommended-doc texlive-pictures texlive-pictures-doc texlive-pstricks texlive-pstricks-doc texlive-science texlive-science-doc

#python
sudo apt-get install liblapack-dev gfortran
sudo apt-get install python3-pip
sudo pip3 install pandas
sudo pip3 install scipy

#install CUDA (optional)
#sudo apt-get install nvidia-cuda-toolkit 

mkdir cogadb_new_depency_installer_tmp_dir
cd "cogadb_new_depency_installer_tmp_dir"

sudo apt-get update
#install OpenCL include and ICT loader
sudo apt-get install ocl-icd-libopencl1 ocl-icd-opencl-dev
#install LLVM
sudo apt-get install clang-3.6 libclang-3.6-dev libclang-common-3.6-dev llvm-3.6 llvm-3.6-dev llvm-3.6-examples llvm-3.6-runtime llvm-3.6-tools

#install boost compute
wget https://github.com/boostorg/compute/archive/v0.5.tar.gz
ls
tar xvfz v0.5.tar.gz
cd compute-0.5/
ls
mkdir build
cd build/
cmake -DCMAKE_BUILD_TYPE=Release ..
make
sudo make install
cd ..

#install rapid JSON
wget https://github.com/miloyip/rapidjson/archive/v1.0.2.tar.gz
tar xvfz v1.0.2.tar.gz
cd rapidjson-1.0.2
mkdir build
cd build
cmake ..
make
sudo make install

#install AMD APP SDK 3.0 for 64 bit linux (optional, only required for OpenCL features such as OpenCL code generation)
#http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/

#tar xvfj AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2 
#sudo ./AMD-APP-SDK-v3.0.130.136-GA-linux64.sh

#cd /usr/lib/
#sudo ln -s /opt/AMDAPPSDK-3.0/lib/x86_64/libOpenCL.so
#cd /usr/include/
#sudo ln -s /opt/AMDAPPSDK-3.0/include/CL

cd ../..
rm -rf "cogadb_new_depency_installer_tmp_dir"

