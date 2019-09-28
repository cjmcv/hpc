#define SIMPLE
#ifdef SIMPLE

#include <thread>

#include "hcs/executor.hpp"
#include "hcs/profiler.hpp"
#include "hcs/blob.hpp"
#include "hcs/util/timer.hpp"

template <typename... Args>
auto PrintArgs(Args&&... args) {
  std::tuple<Args...> tup_data;
  tup_data = std::make_tuple(args...);

  int size = std::tuple_size<decltype(tup_data)>::value;
  std::string str = new std::string[size];

  const char *argc[] = { typeid(Args).name()... };
  for (int i = 0; i < size; i++) {
    str[i] = (std::string)argc[i];
  }
  return tup_data;
}

void WorkB(std::vector<hcs::Blob *> inputs, hcs::Blob *output) {
  printf("(<%s>: %d, start)", output->name().c_str(), std::this_thread::get_id());

  if (inputs.size() != 1) {
    std::cout << "inputs.size() != 1" << std::endl;
  }

  // Fetch input.
  int* data = (int *)inputs[0]->data_; // 2 int
  
  // Process.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  data[0]++;
  data[1]++;

  // Set output.
  inputs[0]->CloneTo(output);

  printf("(<%s>: %d, end)", output->name().c_str(), std::this_thread::get_id());
}

void WorkC(std::vector<hcs::Blob *> inputs, hcs::Blob *output) {
  printf("(<%s>: %d, start)", output->name().c_str(), std::this_thread::get_id());

  if (inputs.size() != 1) {
    std::cout << "inputs.size() != 1" << std::endl;
  }

  // Fetch input.
  int* data = (int *)inputs[0]->data_; // 2 int
  
  // Process.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  data[0]++;
  data[1]++;

  // Set output.
  inputs[0]->CloneTo(output);
  printf("(<%s>: %d, end)", output->name().c_str(), std::this_thread::get_id());
}

void WorkD(std::vector<hcs::Blob *> inputs, hcs::Blob *output) {
  printf("(<%s>: %d, start)", output->name().c_str(), std::this_thread::get_id());

  if (inputs.size() != 1) {
    std::cout << "dependents.size() != 1" << std::endl;
  }

  // Fetch input.
  int* in_data = (int *)inputs[0]->data_; // 2 int
  // Fetch output.
  output->SyncParams(1, 1, 1, 3, hcs::ON_HOST, hcs::FLOAT32);  
  float* out_data = (float *)output->data_;

  // Pass object id.
  output->object_id_ = inputs[0]->object_id_;

  // Process.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  out_data[0] = in_data[0] + 1.1;
  out_data[1] = in_data[1] + 1.1;
  out_data[2] = 1.1;

  printf("(<%s>: %d, end)", output->name().c_str(), std::this_thread::get_id());
}

void WorkE1(std::vector<hcs::Blob *> inputs, hcs::Blob *output) {
  printf("(<%s>: %d, start)", output->name().c_str(), std::this_thread::get_id());

  if (inputs.size() != 1) {
    std::cout << "dependents.size() != 1" << std::endl;
  }

  // Fetch input.
  float* in_data = (float *)inputs[0]->data_; // 2 int
  // Fetch output.
  output->SyncParams(1, 1, 1, 3, hcs::ON_HOST, hcs::FLOAT32);
  float* out_data = (float *)output->data_;

  // Pass object id.
  output->object_id_ = inputs[0]->object_id_;

  // Process.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  out_data[0] = in_data[0] + 1.1;
  out_data[1] = in_data[1] + 1.1;
  out_data[2] = in_data[2] + 1.1;

  printf("(<%s>: %d, end)", output->name().c_str(), std::this_thread::get_id());
}

void WorkE(std::vector<hcs::Blob *> inputs, hcs::Blob *output) {
  printf("(<%s>: %d, start)", output->name().c_str(), std::this_thread::get_id());

  if (inputs.size() != 2) {
    std::cout << "inputs.size() != 2" << std::endl;
  }

  // Fetch input.
  int* in_data = (int *)inputs[0]->data_; // 2 int
  float* in2_data = (float *)inputs[1]->data_; // 3 float
  // Fetch output.
  output->SyncParams(1, 1, 1, 4, hcs::ON_HOST, hcs::FLOAT32);
  float* out_data = (float *)output->data_;

  // Pass object id.
  output->object_id_ = inputs[0]->object_id_;

  // Process.
  std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
  out_data[0] = in_data[0] + in2_data[0] + 1.1;
  out_data[1] = in_data[1] + in2_data[1] + 1.1;
  out_data[2] = in2_data[2] + 1.1;
  out_data[3] = 1.2;

  printf("(<%s>: %d, end)", output->name().c_str(), std::this_thread::get_id());
}

void Add() {
  printf("<master thread: %d>", std::this_thread::get_id());
  hcs::Blob input("in");
  input.Create(1, 1, 1, 2, hcs::ON_HOST, hcs::INT32);
  input.object_id_ = 1;
  int *data = (int *)input.data_;
  data[0] = 1;
  data[1] = 1;

  hcs::Graph graph;

  hcs::Node *A = graph.emplace()->name("A");
  hcs::Node *B = graph.emplace(WorkB)->name("B");
  hcs::Node *C = graph.emplace(WorkC)->name("C");
  hcs::Node *D = graph.emplace(WorkD)->name("D");
  hcs::Node *OUT = graph.emplace(WorkE)->name("E");

  //      | -- C |
  // A -- B       -- E
  //      | -- D |
  A->precede(B);
  B->precede(C);
  B->precede(D);
  C->precede(OUT);
  D->precede(OUT);

  //// A -- B -- C -- D  -- E
  //A->precede(B);
  //B->precede(C);
  //C->precede(D);
  //D->precede(OUT);

  int buffer_size = 100;
  graph.Initialize(buffer_size);

  hcs::Executor executor;
  executor.name_ = "AAA";
  executor.Bind(&graph);

  hcs::Profiler profiler(&executor, &graph);
  profiler.Start(0, 200);
  
  {
    hcs::Blob out("out");
    input.object_id_ = -1;
    A->PushOutput(&input);
    executor.Run().wait();
    OUT->PopOutput(&out, 0);
  }

  hcs::CpuTimer timer;
  float push_time = 0.0;
  float get_time = 0.0;

  int loop_cnt = 10;
  while (1) {
    if (loop_cnt <= 0)
      break;
    loop_cnt--;

    timer.Start();
    printf(">>>>>>>>>>>>>>>>> New Round <<<<<<<<<<<<<<<<.\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    for (int i = 0; i < buffer_size; i++) {
      input.object_id_ = i;
      A->PushOutput(&input);
    }
    executor.Run();

    timer.Stop();
    push_time = timer.MilliSeconds();

    timer.Start();
    int count = 0;
    while (count < buffer_size) {
      hcs::Blob out("out");
      bool flag = OUT->PopOutput(&out, 0);
      if (flag == true) {
        count++;
        float *data = (float *)out.data_;
        printf("< %d , <", out.object_id_);
        for (int i = 0; i < out.num_element_; i++) {
          printf("%f, ", data[i]);
        }
        printf(">.\n");
      }
      else {
        std::cout << count << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        executor.NotifyAll();
      }
    }
    timer.Stop();
    get_time = timer.MilliSeconds();
    printf("time: %f, %f.\n", push_time, get_time);
  }

  graph.Clean();
  input.Release();
}

int main() {
  while(1)
    Add();

  return 0;
}
#endif // SIMPLE