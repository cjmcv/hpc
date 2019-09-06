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
  hcs::Assistor::GetInput(dependents[0]->out_, &in);

  in.i++;
  in.f++;

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
  hcs::Assistor::GetInput(dependents[0]->out_, &in);

  in.i++;
  in.f++;

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
  hcs::Assistor::GetInput(dependents[0]->out_, &in);

  in.i++;
  in.f++;

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
  hcs::Assistor::GetInput(dependents[0]->out_, &in);
  hcs::ParamsCxII in2;
  hcs::Assistor::GetInput(dependents[1]->out_, &in2);

  in.i += in2.i1;
  in.f += in2.i2;

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

  hcs::Executor executor;
  hcs::Graph graph;

  hcs::Node *A = graph.emplace();
  hcs::Node *B = graph.emplace(Work1);
  hcs::Node *C = graph.emplace(Work2);
  hcs::Node *D = graph.emplace(Work3);
  hcs::Node *E = graph.emplace(Work4);

  hcs::Assistor::SetOutput(&input, &(A->out_));

  A->precede(B);
  B->precede(C);
  B->precede(D);
  C->precede(E);
  D->precede(E);

  executor.Run(graph).wait(); 

  hcs::IOParams *out;  
  E->get_output(&out);
  std::cout << ((hcs::ParamsCxII *)out)->i1 << ", " << ((hcs::ParamsCxII *)out)->i2 << ", " << ((hcs::ParamsIF *)out)->obj_id << std::endl;
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