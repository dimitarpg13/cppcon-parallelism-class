#define CATCH_CONFIG_MAIN
#include <CL/sycl.hpp>
#include <iostream>

int main() {
  using namespace cl::sycl;

  int data[1024]; // initialize data to be worked on

  // all SYCL tasks must complete before exiting the block
  {
     // create a queue to enqueue work to 
     queue myQueue;

     // wrap host data in a buffer
     buffer<int, 1> resultBuf(data, range<1>(1024));

     // create command_group to issue commands to the queue
     myQueue.submit([&](handler& cgh) {
        auto writeResult = resultBuf.get_access<cl::sycl::access::mode::write>(cgh);

        // enqueue a parallel_for task
        cgh.parallel_for<class simple_test>(range<1>(1024), [=](id<1> idx) {
           writeResult[idx] = static_cast<int>(idx[0]);
        });

     });
   }

   for (int i = 0; i < 1024; ++i)
      std::cout << "data[" << i << "] = "<<data[i]<<std::endl;

     return 0;
}
