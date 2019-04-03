/*!
* \brief Records the basic usage of flush.
* \operation flush: It makes a thread's temporary 
* view of memory consistent with memory, and enforces 
* an order on the memory operations of the variables. 
*/

#include <omp.h>  
#include <stdio.h>  

void process(int *data, int len) {
  for (int i = 0; i < len; i++) {
    data[i] += 1;
 }
}

int main() {
  int len = 100;
  int *data = new int[len];
  int flag = 0;

  #pragma omp parallel sections num_threads(2)
  {
    #pragma omp section  
    {
      printf("<%d> Initialize data: \n", omp_get_thread_num());
      for (int i = 0; i < len; i++)
        data[i] = 1;

      //   flush also means refresh.
      //   If var is not specified, all memory is flushed.
      //   After executing this command, the memory of data will be flushed 
      // and synchronized with the real memory. Then all of the threads will 
      // see the same data.
      //   Note: Not executing this sentence will also synchronize, 
      // but it may not be in time
      #pragma omp flush
      flag = 1;
      #pragma omp flush(flag)

      // You can do more work here.
      printf("Finish section 0.\n");
    }

    #pragma omp section   
    {
      // Check the flag.
      while (!flag) {
        #pragma omp flush(flag)  
      }
      // If the flag has been synchronized, the data has been initialized.
      // But it may not be synchronized, so emphasize it.
      #pragma omp flush
      
      // Process data from thread 0. 
      printf("<%d> Process data: \n", omp_get_thread_num());
      process(data, len);
      for (int i = 0; i < len; i++) {
        printf("%d, ", data[i]);
      }
      printf("Finish section 1.\n");
    }
  }

  delete data;
}
