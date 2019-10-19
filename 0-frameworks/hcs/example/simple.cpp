//#define SIMPLE
#ifdef SIMPLE

#include <thread>
#include <ctime>

#include "hcs/executor.hpp"
#include "hcs/profiler.hpp"

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

class TestClass : public hcs::TaskAssistor {
public:
  TestClass() {
    count_ = 0;
  }
  std::atomic<int> count_;
};

void WorkB(hcs::TaskAssistor *assistor, std::vector<hcs::Blob *> inputs, hcs::Blob *output) {
  if (inputs.size() != 1) {
    std::cout << "inputs.size() != 1" << std::endl;
  }
  ((TestClass *)assistor)->count_++;
  //printf("tB(%d),", ((TestClass *)assistor)->id());

  // Fetch input.
  int* data = (int *)inputs[0]->GetHostData(); // 2 int
  
  // Process.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  data[0]++;
  data[1]++;

  // Set output.
  inputs[0]->CloneTo(output);
}

void WorkC(hcs::TaskAssistor *assistor, std::vector<hcs::Blob *> inputs, hcs::Blob *output) {
  if (inputs.size() != 1) {
    std::cout << "inputs.size() != 1" << std::endl;
  }
  ((TestClass *)assistor)->count_++;
  //printf("tC(%d),", ((TestClass *)assistor)->id());

  // Fetch input.
  int* data = (int *)inputs[0]->GetHostData(); // 2 int
  
  // Process.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  data[0]++;
  data[1]++;

  // Set output.
  inputs[0]->CloneTo(output);
}

void WorkD(hcs::TaskAssistor *assistor, std::vector<hcs::Blob *> inputs, hcs::Blob *output) {
  if (inputs.size() != 1) {
    std::cout << "dependents.size() != 1" << std::endl;
  }
  ((TestClass *)assistor)->count_++;
  //printf("tD(%d),", ((TestClass *)assistor)->id());

  // Fetch input.
  int* in_data = (int *)inputs[0]->GetHostData(); // 2 int
  // Fetch output.
  output->SyncParams(1, 1, 1, 3, hcs::ON_HOST, hcs::FLOAT32);  
  float* out_data = (float *)output->GetHostData();

  // Process.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  out_data[0] = in_data[0] + 1.1;
  out_data[1] = in_data[1] + 1.1;
  out_data[2] = 1.1;
}

void WorkE(hcs::TaskAssistor *assistor, std::vector<hcs::Blob *> inputs, hcs::Blob *output) {
  if (inputs.size() != 2) {
    std::cout << "inputs.size() != 2" << std::endl;
  }
  ((TestClass *)assistor)->count_++;
  //printf("tE(%d),", ((TestClass *)assistor)->id());

  // Fetch input.
  int* in_data = (int *)inputs[0]->GetHostData(); // 2 int
  float* in2_data = (float *)inputs[1]->GetHostData(); // 3 float
  // Fetch output.
  output->SyncParams(1, 1, 1, 4, hcs::ON_HOST, hcs::FLOAT32);
  float* out_data = (float *)output->GetHostData();

  // Process.
  std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
  out_data[0] = in_data[0] + in2_data[0] + 1.1;
  out_data[1] = in_data[1] + in2_data[1] + 1.1;
  out_data[2] = in2_data[2] + 1.1;
  out_data[3] = 1.2;
}

void Add() {
  printf("<master thread: %d>", std::this_thread::get_id());
  hcs::Blob input("in");
  input.Create(1, 1, 1, 2, hcs::ON_HOST, hcs::INT32);
  input.object_id_ = 1;
  int *data = (int *)input.GetHostData();
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

  hcs::Executor executor("SimpleTest");
  TestClass ass;
  executor.Bind(&graph, hcs::PARALLEL, &ass); // hcs::SERIAL  hcs::PARALLEL

  hcs::Profiler profiler(&executor, &graph);
  int config_flag = hcs::VIEW_NODE | hcs::VIEW_STATUS_RUN_TIME | hcs::VIEW_STATUS;// | 
  profiler.Config(config_flag, 200);
  profiler.Start();

  //  Note: If you don't wait a bit, the first run in the main thread
  // may be faster than the profiler's thread, causing the failure of
  // first timing in profiler.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  { // Test wait().
    hcs::Blob out("out");
    input.object_id_ = -1;
    A->Enqueue(&input);
    A->Enqueue(&input);
    executor.Run().wait();
    OUT->Dequeue(&out);
    OUT->Dequeue(&out);

    A->Enqueue(&input);
    executor.Run().wait();
    OUT->Dequeue(&out);
    assert(OUT->num_cached_buf(0) == 0);
  }

  clock_t time;
  float push_time = 0.0;
  float get_time = 0.0;

  int loop_cnt = 10;
  while (1) {
    if (loop_cnt <= 0)
      break;
    loop_cnt--;

    time = clock();
    printf(">>>>>>>>>>>>>>>>> New Round <<<<<<<<<<<<<<<<.\n");
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    for (int i = 0; i < buffer_size; i++) {
      input.object_id_ = i;
      A->Enqueue(&input);
    }
    executor.Run();

    push_time = (clock() - time) * 1000.0 / CLOCKS_PER_SEC;

    time = clock();
    int count = 0;
    while (count < buffer_size) {
      hcs::Blob out("out");
      bool flag = OUT->Dequeue(&out);
      if (flag == true) {
        count++;
        float *data = (float *)out.data();
        printf("< %d , <", out.object_id_);
        for (int i = 0; i < out.len(); i++) {
          printf("%f, ", data[i]);
        }
        printf(">.\n");
      }
      else {
        std::cout << count << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        //executor.NotifyAll();
      }
    }
    get_time = (clock() - time) * 1000.0 / CLOCKS_PER_SEC;
    printf("time: %f, %f. ass: %d.\n", push_time, get_time, ass.count_.load());
  }

  profiler.Stop();
  graph.Clean();
  input.Release();
}

int main() {
  while(1)
    Add();

  return 0;
}
#endif // SIMPLE