#include <iostream>
#include <thread>
#include <cstdlib>
#include <queue>

int Partition(int *array, int left, int right) {
  int pivot = array[left];
  int low = left, high = right;
  while (low < high) {
    while (array[high] >= pivot && low < high)
      high--;
    array[low] = array[high];
    while (array[low] <= pivot && low < high)
      low++;
    array[high] = array[low];
  }
  array[low] = pivot;
  return low;
}

// 递归
void QuickSortV1(int *array, int left, int right) {
  if (left >= right) {
    return;
  }
  int mid = Partition(array, left, right);
  QuickSortV1(array, left, mid - 1);
  QuickSortV1(array, mid + 1, right);
}

// 任务排队
void QuickSortV2(int *array, int left, int right) {
  std::queue<std::pair<int, int>> fragments;
  fragments.push(std::make_pair(left, right));
  while (fragments.size() > 0) {
    std::pair<int, int> pair = fragments.front();
    fragments.pop();
    int mid = Partition(array, pair.first, pair.second);
    if (mid - 1 > pair.first)
      fragments.push(std::make_pair(pair.first, mid - 1));
    if (mid + 1 < pair.second)
      fragments.push(std::make_pair(mid + 1, pair.second));
  }
}

void ParallelQuickSort(int num_threads, int *array, int left, int right) {
  std::queue<std::pair<int, int>> fragments;
  fragments.push(std::make_pair(left, right));
  // 分段
  while (fragments.size() < num_threads && fragments.size() > 0) {
    std::pair<int, int> pair = fragments.front();
    fragments.pop();
    int mid = Partition(array, pair.first, pair.second);
    if (mid - 1 > pair.first)
      fragments.push(std::make_pair(pair.first, mid - 1));
    if (mid + 1 < pair.second)
      fragments.push(std::make_pair(mid + 1, pair.second));
  }
  // 取最小值开线程
  int size;
  if (fragments.size() < num_threads)
    size = fragments.size();
  else
    size = num_threads;
  // 每段分别处理
  std::thread *threads = new std::thread[size];
  for (int i = 0; i < size; i++) {
    std::pair<int, int> pair = fragments.front();
    threads[i] = std::thread(QuickSortV1, array, pair.first, pair.second);
    //threads[i] = std::thread(QuickSortV2, array, pair.first, pair.second);
    fragments.pop();
  }
  for (int i = 0; i < size; i++) {
    threads[i].join();
  }
  delete[]threads;
}

#define LEN 1000000
int main() {
  int *in1 = new int[LEN];
  int *in2 = new int[LEN];
  int *in3 = new int[LEN];
  // Initialize.
  srand(time(0));
  for (int i = 0; i < LEN; i++) {
    in1[i] = rand() % LEN / 2;
    in3[i] = in1[i];
    in2[i] = in1[i];
  }

  //QuickSortV1->time: 122
  //QuickSortV2->time : 140
  //ParallelQuickSort->time : 65
  // Compute.
  time_t time = clock();
  QuickSortV1(in1, 0, LEN - 1);
  std::cout << "QuickSortV1 -> time: " << clock() - time << std::endl;
  time = clock();
  QuickSortV2(in2, 0, LEN - 1);
  std::cout << "QuickSortV2 -> time: " << clock() - time << std::endl;
  int num_threads = 5;
  time = clock();
  ParallelQuickSort(num_threads, in3, 0, LEN - 1);
  std::cout << "ParallelQuickSort -> time: " << clock() - time << std::endl;
  // Check result.
  for (int i = 0; i < LEN; i++) {
    if (in1[i] != in2[i] || in1[i] != in3[i]) {
      printf("Failed.\n");
    }
  }
  std::cout << std::endl;

  delete[]in1;
  delete[]in2;
  delete[]in3;

  system("pause");
  return 0;
}

