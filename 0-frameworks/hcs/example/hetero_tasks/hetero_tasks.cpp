#define HETERO_TASKS
#ifdef HETERO_TASKS

#include <thread>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "hcs/executor.hpp"
#include "hcs/profiler.hpp"

extern int CallGemmGPU(const int M, const int N,
                      const int K, const float alpha,
                      const float *A, const int lda,
                      const float *B, const int ldb,
                      const float beta,
                      float *C, const int ldc, cudaStream_t stream = nullptr);

void Div2(float *src, int len, float *dst) {
  for (int c = 0; c < 10; c++) {
    for (int i = 0; i < len; i++) {
      dst[i] = src[i] / 2 + 1;
    }
  }
}

float AccMean(float *arr, float *arr2, int len) {
  float ave = 0.0;
  for (int i = 0; i < len; i++) {
    ave = (i / (i + 1.0)) * ave + (arr[i] + arr2[i]) / (i + 1.0);
  }
  return ave;
}

class TestClass : public hcs::TaskAssistor {
public:
  TestClass() {
    count_ = 0;
  }
  std::atomic<int> count_;
};

// C in G out.
void TaskHost(hcs::TaskAssistor *assistor, std::vector<hcs::Blob *> inputs, hcs::Blob *output) {
  if (inputs.size() != 1) {
    std::cout << "inputs.size() != 1" << std::endl;
  }
  //printf("<s-%d-TaskHost2Device, %d>", assistor->id(), assistor->stream());

  // Fetch input.
  float* data = (float *)inputs[0]->GetHostData();
  std::vector<int> shape = inputs[0]->shape();
  // Fetch output.
  output->SyncParams(1, 1, shape[2], shape[3], hcs::ON_DEVICE, hcs::FLOAT32);
  float* out_data = (float *)output->GetHostData();

  // Process. 
  Div2(data, inputs[0]->len(), out_data);

  //printf("TaskHost(%d, %d)", inputs[0]->object_id_, assistor->stream());
}

// G in C out
void TaskDevice(hcs::TaskAssistor *assistor, std::vector<hcs::Blob *> inputs, hcs::Blob *output) {
  if (inputs.size() != 1) {
    std::cout << "inputs.size() != 1" << std::endl;
  }
  //printf("<s-%d-TaskDevice2Host, %d>", assistor->id(), assistor->stream());

  // Fetch input.
  float* d_in_data = (float *)inputs[0]->GetDeviceData();
  std::vector<int> shape = inputs[0]->shape();
  // Fetch output.
  output->SyncParams(1, 1, shape[2], shape[3], hcs::ON_HOST, hcs::FLOAT32);
  float *d_buffer = (float *)output->GetDeviceData();

  // Process.
  CallGemmGPU(shape[2], shape[2], 
              shape[2], 1.0,
              d_in_data, shape[2],
              d_in_data, shape[2],
              0,
              d_buffer, shape[2], assistor->stream());

  //printf("TaskDevice(%d, %d)", inputs[0]->object_id_, assistor->stream());
}

// 2C in C out
void TaskHost2(hcs::TaskAssistor *assistor, std::vector<hcs::Blob *> inputs, hcs::Blob *output) {
  if (inputs.size() != 2) {
    std::cout << "inputs.size() != 2" << std::endl;
  }
  //printf("<s-%d-TaskHost, %d>", assistor->id(), assistor->stream());

  // Fetch input.
  float* h_in_data = (float *)inputs[0]->GetHostData();
  float* h_in2_data = (float *)inputs[1]->GetHostData();
  std::vector<int> shape = inputs[0]->shape();
  // Fetch output.
  output->SyncParams(1, 1, 1, 1, hcs::ON_HOST, hcs::FLOAT32);
  float* h_out_data = (float *)output->GetHostData();

  // Process.
  *h_out_data = AccMean(h_in_data, h_in2_data, inputs[0]->len());

  //printf("TaskHost2(%d, %d)", inputs[0]->object_id_, assistor->stream());
}

void Task() {
  printf("<master thread: %d>", std::this_thread::get_id());
  hcs::Blob input("in");
  input.Create(1, 1, 1280, 1280, hcs::ON_HOST, hcs::FLOAT32);
  input.object_id_ = 1;
  float *data = (float *)input.data();
  for (int i = 0; i < input.len(); i++) {
    data[i] = 2;
  }

  hcs::Graph graph;

  hcs::Node *A = graph.emplace()->name("A");
  hcs::Node *B1C = graph.emplace(TaskHost)->name("B1C");
  hcs::Node *B2G = graph.emplace(TaskDevice)->name("B2G");
  hcs::Node *C1C = graph.emplace(TaskHost)->name("C1C");
  hcs::Node *C2G = graph.emplace(TaskDevice)->name("C2G");
  hcs::Node *D1C = graph.emplace(TaskHost)->name("D1C");
  hcs::Node *D2G = graph.emplace(TaskDevice)->name("D2G");
  hcs::Node *OUT = graph.emplace(TaskHost2)->name("OUT");

  //                      | -- C1C -- C2G |
  // AC(input) -- B1C -- B2G              -- O(acc)
  //                      | -- D1C -- D2G |
  A->precede(B1C);
  B1C->precede(B2G);

  B2G->precede(C1C);
  C1C->precede(C2G);
  B2G->precede(D1C);
  D1C->precede(D2G);
  
  D2G->precede(OUT);
  C2G->precede(OUT);

  int queue_size = 60;
  graph.Initialize(queue_size);

  hcs::Executor executor("HeteroTest");
  TestClass ass;
  executor.Bind(&graph, hcs::PARALLEL_MULTI_STREAMS, &ass); // hcs::SERIAL  hcs::PARALLEL hcs::PARALLEL_MULTI_STREAMS

  hcs::Profiler profiler(&executor, &graph);
  int config_flag = hcs::VIEW_STATUS_RUN_TIME | hcs::VIEW_NODE | hcs::VIEW_NODE_RUN_TIME;//  | hcs::VIEW_STATUS;
  profiler.Config(config_flag, 200);
  profiler.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  clock_t time;
  float serial_time = 0.0;
  float push_time = 0.0;
  float get_time = 0.0;

  ////////////////////////
  {
    hcs::Blob out_blob("out");
    // warn.
    A->Enqueue(&input);
    executor.Run().wait();
    OUT->Dequeue(&out_blob);

    time = clock();
    A->Enqueue(&input);
    executor.Run().wait();
    OUT->Dequeue(&out_blob);
    serial_time = (clock() - time) * 1000.0 / CLOCKS_PER_SEC;

    printf("Finish.\n");
  }

  // Loop.
  int loop_cnt = 10;
  while (1) {
    if (loop_cnt <= 0)
      break;
    loop_cnt--;

    time = clock();
    printf(">>>>>>>>>>>>>>>>> New Round <<<<<<<<<<<<<<<<.\n");
    for (int i = 0; i < queue_size; i++) {
      input.object_id_ = i;
      A->Enqueue(&input);
    } 
    executor.Run();
    push_time = (clock() - time) * 1000.0 / CLOCKS_PER_SEC;

    time = clock();
    int count = 0;
    while (count < queue_size) {
      hcs::Blob out_blob("out");
      bool flag = OUT->Dequeue(&out_blob);
      if (flag == true) {
        count++;
        float *data = (float *)out_blob.data();
        printf("< %d , <", out_blob.object_id_);
        for (int i = 0; i < 1; i++) {//out.num_element_
          printf("%f, ", data[i]);
        }
        printf(">.\n");
      }
      else {
        std::cout << count << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    }
    get_time = (clock() - time) * 1000.0 / CLOCKS_PER_SEC;
    printf("%d-times: push %f, get %f, serial: %f. \n", queue_size, push_time / queue_size, get_time / queue_size, serial_time);
  }

  profiler.Stop();
  graph.Clean();
  input.Release();
}

int main() {
  while (1)
    Task();

  return 0;
}

#endif // HETERO_TASKS