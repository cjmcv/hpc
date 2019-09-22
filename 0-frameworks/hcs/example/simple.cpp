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

void WorkB(std::vector<hcs::Node*> &dependents, hcs::Blob *output) {
  printf("(<1>: %d, start)", std::this_thread::get_id());

  if (dependents.size() != 1) {
    std::cout << "dependents.size() != 1" << std::endl;
  }
  hcs::Blob *in = dependents[0]->BorrowOut(); // 2 int
  
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  int* data = (int *)in->data_;
  data[0]++;
  data[1]++;

  // Set output.
  in->CloneTo(output);

  dependents[0]->RecycleOut(in);
  printf("(<1>: %d, end).", std::this_thread::get_id());
}

void WorkC(std::vector<hcs::Node*> &dependents, hcs::Blob *output) {
  printf("(<2>: %d, start)", std::this_thread::get_id());

  if (dependents.size() != 1) {
    std::cout << "dependents.size() != 1" << std::endl;
  }

  // Fetch input from the former node.
  hcs::Blob *in = dependents[0]->BorrowOut(0); // 2 int

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  int* data = (int *)in->data_;
  data[0]++;
  data[1]++;

  // Set output.
  in->CloneTo(output);

  dependents[0]->RecycleOut(in);
  printf("(<2>: %d, end).", std::this_thread::get_id());
}

void WorkD(std::vector<hcs::Node*> &dependents, hcs::Blob *output) {
  printf("(<3>: %d, start)", std::this_thread::get_id());

  if (dependents.size() != 1) {
    std::cout << "dependents.size() != 1" << std::endl;
  }

  // Fetch input from the former node.
  hcs::Blob *in = dependents[0]->BorrowOut(1); // 2 int

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  int* data = (int *)in->data_;
  data[0]++;
  data[1]++;

  hcs::Blob out_temp;
  out_temp.Create(1, 1, 1, 3, hcs::ON_HOST, hcs::FLOAT32);
  out_temp.object_id_ = in->object_id_;
  float *data2 = (float *)out_temp.data_;
  data2[0] = data[0] + 1.1;
  data2[1] = data[1] + 1.1;
  data2[2] = 1.1;

  // Set output.
  out_temp.CloneTo(output);
  out_temp.Release();

  dependents[0]->RecycleOut(in);
  printf("(<3>: %d, end).", std::this_thread::get_id());
}

void WorkE(std::vector<hcs::Node*> &dependents, hcs::Blob *output) {
  printf("(<4>: %d, start).", std::this_thread::get_id());

  if (dependents.size() != 2) {
    std::cout << "dependents.size() != 2" << std::endl;
  }

  // Fetch input from the former node.
  hcs::Blob *in = dependents[0]->BorrowOut(); // 2 int
  hcs::Blob *in2 = dependents[1]->BorrowOut(); // 3 float

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  int* data = (int *)in->data_;
  data[0]++;
  data[1]++;
  float* data2 = (float *)in2->data_;
  data2[0] += 1.0;
  data2[1] += 1.0;
  data2[2] += 1.0;

  hcs::Blob out_temp;
  out_temp.Create(1, 1, 1, 4, hcs::ON_HOST, hcs::FLOAT32);
  out_temp.object_id_ = in->object_id_;
  float *data3 = (float *)out_temp.data_;
  data3[0] = data[0] + data2[0] + 1.1;
  data3[1] = data[1] + data2[1] + 1.1;
  data3[2] = data2[2] + 1.1;
  data3[3] = 1.2;

  // Set output.
  out_temp.CloneTo(output);
  out_temp.Release();

  dependents[0]->RecycleOut(in);
  dependents[1]->RecycleOut(in2);

  printf("(<4>: %d, end).", std::this_thread::get_id());
}

void Add() {
  printf("<master thread: %d>", std::this_thread::get_id());
  hcs::Blob input;
  input.Create(1, 1, 1, 2, hcs::ON_HOST, hcs::INT32);
  input.object_id_ = 1;
  int *data = (int *)input.data_;
  data[0] = 1;
  data[1] = 1;

  hcs::Graph graph;

  hcs::Node *A = graph.emplace()->name("A");
  hcs::Node *B = graph.emplace(WorkB)->name("B");;
  hcs::Node *C = graph.emplace(WorkC)->name("C");;
  hcs::Node *D = graph.emplace(WorkD)->name("D");;
  hcs::Node *E = graph.emplace(WorkE)->name("E");;

  //      | -- C |
  // A -- B       -- E
  //      | -- D |
  A->precede(B);
  B->precede(C);
  B->precede(D);
  C->precede(E);
  D->precede(E);
  graph.Initialize(100);

  hcs::Executor executor;
  executor.name_ = "AAA";
  executor.Bind(&graph);
  //hcs::Profiler profiler(&executor, &graph);
  //profiler.Start(1, 200);
  
  hcs::CpuTimer timer;
  float push_time = 0.0;
  float get_time = 0.0;
  //while (1) 
  {
    timer.Start();
    printf(">>>>>>>>>>>>>>>>> New Round <<<<<<<<<<<<<<<<.\n");
    for (int i = 0; i < 100; i++) {
      input.object_id_ = i;
      A->PushOutput(&input);
      executor.Run();
    }
    timer.Stop();
    push_time = timer.MilliSeconds();

    timer.Start();
    int count = 0;
    while (count < 100) {
      hcs::Blob out;
      bool flag = E->PopOutput(&out);
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
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
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
  Add();
  //std::cout << "*******************************" << std::endl << std::endl;

  system("pause");
  return 0;
}
#endif // SIMPLE