#define SIMPLE
#ifdef SIMPLE

#include <thread>

#include "hcs/executor.hpp"
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

void Work1(std::vector<hcs::Node*> &dependents, hcs::IOParams **output) {
  std::cout << std::this_thread::get_id() << ": Work1" << std::endl;

  if (dependents.size() != 1) {
    std::cout << "dependents.size() != 1" << std::endl;
  }
  hcs::ParamsIF in;
  dependents[0]->PopOutput(&in);

  printf("1>before: %d, %f, %d.\n", in.i, in.f, in.obj_id);
  in.i++;
  in.f++;
  printf("1>after: %d, %f, %d.\n", in.i, in.f, in.obj_id);

  // Set output.
  hcs::Assistor::SetOutput(&in, output);
}

void Work2(std::vector<hcs::Node*> &dependents, hcs::IOParams **output) {
  std::cout << std::this_thread::get_id() << ": Work2" << std::endl;

  if (dependents.size() != 1) {
    std::cout << "dependents.size() != 1" << std::endl;
  }

  // Fetch input from the former node.
  hcs::ParamsIF in;
  dependents[0]->PopOutput(&in, 0);

  printf("2>before: %d, %f, %d.\n", in.i, in.f, in.obj_id);
  in.i++;
  in.f++;
  printf("2>after: %d, %f, %d.\n", in.i, in.f, in.obj_id);

  // Set output.
  hcs::Assistor::SetOutput(&in, output);
}

void Work3(std::vector<hcs::Node*> &dependents, hcs::IOParams **output) {
  std::cout << std::this_thread::get_id() << ": Work3" << std::endl;

  if (dependents.size() != 1) {
    std::cout << "dependents.size() != 1" << std::endl;
  }

  // Fetch input from the former node.
  hcs::ParamsIF in;
  dependents[0]->PopOutput(&in, 1);

  printf("3>before: %d, %f, %d.\n", in.i, in.f, in.obj_id);
  in.i++;
  in.f++;
  printf("3>after: %d, %f, %d.\n", in.i, in.f, in.obj_id);

  hcs::ParamsCxII out_temp;
  out_temp.i1 = in.i;
  out_temp.i2 = in.f;
  
  out_temp.obj_id = in.obj_id;

  // Set output.
  hcs::Assistor::SetOutput(&out_temp, output);
}

void Work4(std::vector<hcs::Node*> &dependents, hcs::IOParams **output) {
  std::cout << std::this_thread::get_id() << ": Work4" << std::endl;

  if (dependents.size() != 2) {
    std::cout << "dependents.size() != 2" << std::endl;
  }

  // Fetch input from the former node.
  hcs::ParamsIF in;
  dependents[0]->PopOutput(&in);
  hcs::ParamsCxII in2;
  dependents[1]->PopOutput(&in2);

  printf("4>before: %d, %f, %d.\n", in.i, in.f, in.obj_id);
  in.i += in2.i1;
  in.f += in2.i2;
  printf("4>after: %d, %f, %d.\n", in.i, in.f, in.obj_id);

  hcs::ParamsCxII out_temp;
  out_temp.i1 = in.i;
  out_temp.i2 = in.f;

  out_temp.obj_id = in.obj_id;

  // Set output.
  hcs::Assistor::SetOutput(&out_temp, output);
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
  hcs::Node *D = graph.emplace(Work3, hcs::PARAMS_CXII)->name("D");;
  hcs::Node *E = graph.emplace(Work4, hcs::PARAMS_CXII)->name("E");;

  A->PushOutput(&input);

  //      | -- C |
  // A -- B       -- E 
  //      | -- D |
  A->precede(B);
  B->precede(C);
  B->precede(D);
  C->precede(E);
  D->precede(E);

  hcs::Executor executor;
  std::future f1 = executor.Run(graph);

  hcs::Executor executor2;
  input.obj_id = 2;
  A->PushOutput(&input);
  std::future f2 = executor2.Run(graph);

  // TODO：核对executor中的成员变量，分析哪些是与当前计算图的计算状态有关的，看哪些在算完后是需要重置的。
  //f1.get();
  //f2.get();

  //while(1)
  //  executor.StatusView(graph);
  //std::this_thread::sleep_for(std::chrono::seconds(5));
  
  int count = 0;
  while (1) {
    if (count >= 2) { break; }

    hcs::ParamsCxII out;
    bool flag = E->PopOutput(&out);
    if (flag == true) {
      count++;
      std::cout << out.i1 << ", " << out.i2 << ", " << out.obj_id << std::endl;
    }
    else {
      std::cout << count << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

int main() {
  Add();
  //std::cout << "*******************************" << std::endl << std::endl;

  //Simple();
  //std::cout << "*******************************" << std::endl << std::endl;

  //Simple2();
  //std::cout << "*******************************" << std::endl << std::endl;

  //Simple3();
  //std::cout << "*******************************" << std::endl << std::endl;

  //run_variants();
  //std::cout << "*******************************" << std::endl << std::endl;

  system("pause");
  return 0;
}
#endif // SIMPLE