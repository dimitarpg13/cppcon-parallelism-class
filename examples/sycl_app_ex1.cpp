#define CATCH_CONFIG_MAIN
#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

// size of the matrices
const size_t N = 2000;
const size_t M = 3000;
class init_a;
class init_b;
class matrix_add;

int main() {
   //the SYCL tasks must complete before exiting the block
   // open scope to make lifetime of the queue clear
   {
      // create a queue to work on
      queue myQueue;

      // create 2d buffers of float for our matrices
      cl::sycl::buffer<float, 2> a(range<2>{N, M});
      cl::sycl::buffer<float, 2> b(range<2>{N, M});
      cl::sycl::buffer<float, 2> c(range<2>{N, M});

      // launch a first asynchronous kernel to initialize a 
      myQueue.submit( [&](handler& cgh) {
         // the kernel writes a so get an write accessor on it
         auto A = a.template get_access<cl::sycl::access::mode::write>(cgh);

         // enqueue a parallel kernel iterating on a N*M 2D iteration space
         cgh.parallel_for<init_a>(
            range<2>{N, M}, [=](id<2> index) { A[index] = index[0] * 2 + index[1]; });
         });

            
      myQueue.submit( [&](handler& cgh) {
         auto B = b.template get_access<cl::sycl::access::mode::write>(cgh);
         /* from the access pattern the SYCL runtime detects that 
            this 
          */
         cgh.parallel_for<init_b>(range<2>{N, M}, [=](id<2> index) {
            B[index] = index[0] *2014 + index[1] * 42;
         });
      });

      myQueue.submit([&](handler& cgh) {
         auto A = a.get_access<cl::sycl::access::mode::read>(cgh);
         auto B = b.get_access<cl::sycl::access::mode::read>(cgh);
         auto C = c.get_access<cl::sycl::access::mode::write>(cgh);
         // from these accessors the SYCL runtime will ensure that when 
         // that this kernel is run the kernels computing a and b have completed

         // enqueue a parallel kernel iteration on N*M 2D iteration space
         cgh.parallel_for<matrix_add>(
             range<2>{N,M}, [=](id<2> index) { C[index] = A[index] + B[index]; });
         });

      auto A = a.template get_access<cl::sycl::access::mode::read>();
      std::cout << std::endl << "Result: (N=" << N << ", M=" << M << ")" << std::endl;
      for (size_t i = 0; i < N; ++i) {
          for (size_t j = 0; j < M; ++j) {
             // compare the result to the analytic value
             std::cout << A[i][j] << std::endl;
                 
          }
      }

      auto C = c.template get_access<cl::sycl::access::mode::read>();
      std::cout << std::endl << "Result: (N=" << N << ", M=" << M << ")" << std::endl;
      for (size_t i = 0; i < N; ++i) {
          for (size_t j = 0; j < M; ++j) {
             // compare the result to the analytic value
             if (C[i][j] != i * (2 + 2014) + j * (1 + 42)) {
                 std::cout << "Wrong value " << C[i][j] << " on element " << i << " " << j << std::endl;
                 exit(-1);
             }
          }
      }   

   } // enclosing block ensuring all SYCL task will complete
}
