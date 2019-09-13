#define SIMPLE
#ifdef SIMPLE

#include <thread>

#include "hcs/executor.hpp"
#include "hcs/profiler.hpp"
#include "hcs/params.hpp"

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

void Work1(std::vector<hcs::Node*> &dependents, hcs::IOParams *output) {
  printf("(Work1: %d)", std::this_thread::get_id());

  if (dependents.size() != 1) {
    std::cout << "dependents.size() != 1" << std::endl;
  }
  hcs::ParamsIF *in = (hcs::ParamsIF *)(dependents[0]->BorrowOut());

  in->i++;
  in->f++;

  // Set output.
  hcs::Assistor::CopyParams(in, output);

  dependents[0]->RecycleOut(in);
}

void Work2(std::vector<hcs::Node*> &dependents, hcs::IOParams *output) {
  printf("(Work2: %d)", std::this_thread::get_id());

  if (dependents.size() != 1) {
    std::cout << "dependents.size() != 1" << std::endl;
  }

  // Fetch input from the former node.
  hcs::ParamsIF *in = (hcs::ParamsIF *)(dependents[0]->BorrowOut(0));

  in->i++;
  in->f++;

  // Set output.
  hcs::Assistor::CopyParams(in, output);

  dependents[0]->RecycleOut(in);
}

void Work3(std::vector<hcs::Node*> &dependents, hcs::IOParams *output) {
  printf("(Work3: %d)", std::this_thread::get_id());

  if (dependents.size() != 1) {
    std::cout << "dependents.size() != 1" << std::endl;
  }

  // Fetch input from the former node.
  hcs::ParamsIF *in = (hcs::ParamsIF *)(dependents[0]->BorrowOut(1));

  in->i++;
  in->f++;

  hcs::ParamsFxII out_temp;
  out_temp.i1 = 1;
  out_temp.i2 = 2;
  out_temp.fx = new float[2];
  out_temp.fx[0] = in->i;
  out_temp.fx[1] = in->f;
  
  out_temp.obj_id = in->obj_id;

  // Set output.
  hcs::Assistor::CopyParams(&out_temp, output);

  dependents[0]->RecycleOut(in);
}

void Work4(std::vector<hcs::Node*> &dependents, hcs::IOParams *output) {
  printf("(Work4: %d)", std::this_thread::get_id());

  if (dependents.size() != 2) {
    std::cout << "dependents.size() != 2" << std::endl;
  }

  // Fetch input from the former node.
  hcs::ParamsIF *in = (hcs::ParamsIF *)(dependents[0]->BorrowOut());
  hcs::ParamsFxII *in2 = (hcs::ParamsFxII *)(dependents[1]->BorrowOut());

  in->i += in2->i1;
  in->f += in2->i2;

  hcs::ParamsFxII out_temp;
  int len = 2;
  out_temp.fx = new float[len * len];
  out_temp.fx[0] = in->i;
  out_temp.fx[1] = in->f;
  out_temp.fx[2] = in2->i1;
  out_temp.fx[3] = in2->i2;
  out_temp.i1 = len;
  out_temp.i2 = len;

  out_temp.obj_id = in->obj_id;

  // Set output.
  hcs::Assistor::CopyParams(&out_temp, output);

  dependents[0]->RecycleOut(in);
  dependents[1]->RecycleOut(in2);
}

void Add() {
  hcs::ParamsIF input;
  input.i = 10;
  input.f = 12.5;
  input.obj_id = 1;
  input.struct_id = hcs::ParamsMode::PARAMS_IF;

  hcs::Graph graph;

  hcs::Node *A = graph.emplace(hcs::PARAMS_IF)->name("A");
  hcs::Node *B = graph.emplace(Work1, hcs::PARAMS_IF)->name("B");;
  hcs::Node *C = graph.emplace(Work2, hcs::PARAMS_IF)->name("C");;
  hcs::Node *D = graph.emplace(Work3, hcs::PARAMS_FXII)->name("D");;
  hcs::Node *E = graph.emplace(Work4, hcs::PARAMS_FXII)->name("E");;

  //      | -- C |
  // A -- B       -- E
  //      | -- D |
  A->precede(B);
  B->precede(C);
  B->precede(D);
  C->precede(E);
  D->precede(E);

  hcs::Executor executor;
  executor.name_ = "AAA";

  while (1) {
    printf(">>>>>>>>>>>>>>>>> New Round <<<<<<<<<<<<<<<<.\n");
    for (int i = 0; i < 10; i++) {
      input.obj_id = i;
      A->PushOutput(&input);

      executor.Run(graph);
    }

    int count = 0;
    while (count < 10) {
      hcs::ParamsFxII out;
      bool flag = E->PopOutput(&out);
      if (flag == true) {
        count++;
        printf("< %d, (%d, %d), <", out.obj_id, out.i1, out.i2);
        for (int i = 0; i < out.i1 * out.i2; i++) {
          printf("%f, ", out.fx[i]);
        }
        printf(">.\n");
      }
      else {
        std::cout << count << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    }
  }

  //while (1) {
  //  hcs::Profiler::StatusView(graph);
  //  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  //}
}

int main() {
  Add();
  //std::cout << "*******************************" << std::endl << std::endl;

  system("pause");
  return 0;
}
#endif // SIMPLE