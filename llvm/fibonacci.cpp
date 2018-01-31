/*!
* \brief An example of how to build quickly a small module
*      with function Fibonacci and execute it with the JIT.
*        Compile the module via JIT, then execute the `fib'
*      function and return result to a driver.(This program
*      is adapted from an official example in examples\Fibonacci).
*/

#include <iostream>

#include "llvm/ADT/APInt.h"
#include "llvm/IR/Verifier.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

int FibonacciNormal(int x) {
  if (x <= 2) return 1;
  return FibonacciNormal(x - 1)
    + FibonacciNormal(x - 2);
}

class FibonacciLLVM {
public:
  static llvm::Function *CreateFibFunction(llvm::Module *M, llvm::LLVMContext &context);
  int Initialize();
  int Run(const int n);

private:
  // Context should be preserved until this module is released.
  std::shared_ptr<llvm::LLVMContext> ctx_;
  llvm::ExecutionEngine *exec_engine_;
  llvm::Function *fib_func_;
};

llvm::Function *FibonacciLLVM::CreateFibFunction(llvm::Module *M, llvm::LLVMContext &context) {
  using namespace llvm;
  // Create the fib function and insert it into module M. This function is said
  // to return an int and take an int parameter.
  Function *fib_func =
    cast<Function>(M->getOrInsertFunction("fib", Type::getInt32Ty(context),
      Type::getInt32Ty(context)));

  // Add a basic block to the function.
  BasicBlock *entry_bb = BasicBlock::Create(context, "EntryBlock", fib_func);

  // Get pointers to the constants.
  Value *One = ConstantInt::get(Type::getInt32Ty(context), 1);
  Value *Two = ConstantInt::get(Type::getInt32Ty(context), 2);

  // Get pointer to the integer argument of the add1 function...
  Argument *ArgX = &*fib_func->arg_begin(); // Get the arg.
  ArgX->setName("AnArg");            // Give it a nice symbolic name for fun.

                                     // Create the true_block.
  BasicBlock *ret_bb = BasicBlock::Create(context, "return", fib_func);
  // Create an exit block.
  BasicBlock* recurse_bb = BasicBlock::Create(context, "recurse", fib_func);

  // Create the "if (arg <= 2) goto exitbb"
  Value *CondInst = new ICmpInst(*entry_bb, ICmpInst::ICMP_SLE, ArgX, Two, "cond");
  BranchInst::Create(ret_bb, recurse_bb, CondInst, entry_bb);

  // Create: ret int 1
  ReturnInst::Create(context, One, ret_bb);

  // create fib(x-1)
  Value *Sub = BinaryOperator::CreateSub(ArgX, One, "arg", recurse_bb);
  CallInst *CallFibX1 = CallInst::Create(fib_func, Sub, "fibx1", recurse_bb);
  CallFibX1->setTailCall();

  // create fib(x-2)
  Sub = BinaryOperator::CreateSub(ArgX, Two, "arg", recurse_bb);
  CallInst *CallFibX2 = CallInst::Create(fib_func, Sub, "fibx2", recurse_bb);
  CallFibX2->setTailCall();

  // fib(x-1)+fib(x-2)
  Value *sum = BinaryOperator::CreateAdd(CallFibX1, CallFibX2,
    "addresult", recurse_bb);

  // Create the return instruction and add it to the basic block
  ReturnInst::Create(context, sum, recurse_bb);

  return fib_func;
}

int FibonacciLLVM::Initialize() {
  using namespace llvm;

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  ctx_.reset(new LLVMContext);

  // Create some module to put our function into it.
  std::unique_ptr<llvm::Module> module_owner(new Module("test", *ctx_));
  Module *module = module_owner.get();

  // We are about to create the "fib" function:
  fib_func_ = CreateFibFunction(module, *ctx_);

  // Now we going to create JIT
  // After std::move, the exec_engine_ becomes the module holder.
  std::string err_str;
  exec_engine_ =
    EngineBuilder(std::move(module_owner))
    .setErrorStr(&err_str)
    .create();

  if (!exec_engine_) {
    std::cout << "Failed to construct ExecutionEngine: " << err_str << "\n";
    return 1;
  }

  std::cout << "verifying... ";
  if (verifyModule(*module)) {
    std::cout << ": Error constructing function!\n";
    return 1;
  }
  std::cout << "OK\n";

  // outs() and errs() for printing variable in llvm.
  errs() << "We just constructed this LLVM module:\n\n---------\n" << *module;

  return 0;
}

int FibonacciLLVM::Run(const int n) {
  using namespace llvm;

  std::cout << "---------\nstarting fibonacci(" << n << ") with JIT...\n";
  // Call the Fibonacci function with argument n:
  std::vector<GenericValue> in_params(1);
  in_params[0].IntVal = APInt(32, n); // numBits = 32, val = n.
  GenericValue gv = exec_engine_->runFunction(fib_func_, in_params);

  // Import result of execution
  //errs() << "Result: " << gv.IntVal << "\n";

  return gv.IntVal.getZExtValue();
}

int main(int argc, char **argv) {
  int n = 24;
  int result = 0;
  time_t stime;

  FibonacciLLVM fib_llvm;
  fib_llvm.Initialize();

  result = 0;
  stime = clock();
  result = fib_llvm.Run(n);
  std::cout << "llvm: time " << clock() - stime << ", result " << result << std::endl;

  result = 0;
  stime = clock();
  result = FibonacciNormal(n);
  std::cout << "normal: time " << clock() - stime << ", result " << result << std::endl;

  system("pause");
  return 0;
}
