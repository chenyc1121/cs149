# C++同步原语入门（中文版）

你的编程作业2解决方案肯定需要创建线程，并且可能需要使用两种类型的同步原语：互斥锁和条件变量。以下笔记解释这两种同步类型。

我们在起始代码的 `tutorial/tutorial.cpp` 中提供了创建C++线程、锁定/解锁互斥锁以及使用条件变量的基本示例。

## 创建C++线程

在C++中创建新线程很简单。要创建线程，应用程序构造 `std::thread` 对象的新实例。例如，在以下代码中，主线程创建两个运行 `my_func` 函数的线程：

```cpp
#include <thread>
#include <stdio.h>

void my_func(int thread_id, int num_threads) {
    printf("Hello from spawned thread %d of %d\n", thread_id, num_threads);
}

int main(int argc, char** argv) {

  std::thread t0 = std::thread(my_func, 0, 2);
  std::thread t1 = std::thread(my_func, 1, 2);

  printf("The main thread is running concurrently with spawned threads.\n");

  t0.join();
  t1.join();

  printf("Spawned threads have terminated at this point.\n");

  return 0;
}
```

`std::thread` 完整文档：<https://en.cppreference.com/w/cpp/thread/thread>

有用的C++11线程创建教程：<https://www.geeksforgeeks.org/multithreading-in-cpp/>

## 互斥锁（Mutexes）

C++标准库提供了互斥锁同步原语 `std::mutex`，用于保护共享数据不被多个应用程序线程同时访问。

<https://en.cppreference.com/w/cpp/thread/mutex>

线程使用 `mutex::lock()` 锁定互斥锁。调用线程将阻塞直到获取到互斥锁。当 `lock()` 返回时，调用线程保证拥有锁。线程使用 `mutex::unlock()` 解锁。

C++还提供了一些包装类来减少使用锁时的错误：
- [`std::unique_lock`](https://en.cppreference.com/w/cpp/thread/unique_lock)
- [`std::lock_guard`](https://en.cppreference.com/w/cpp/thread/lock_guard)

例如，`lock_guard` 在构造时自动锁定指定的互斥锁，并在离开作用域时自动解锁。

推荐查看 `tutorial/tutorial.cpp` 中的 `mutex_example()` 函数，了解使用互斥锁保护共享计数器更新的简单示例。

## 条件变量（Condition Variables）

条件变量管理等待某个条件成立的线程列表，并允许其他线程通知等待线程感兴趣的事件已发生。条件变量与互斥锁配合使用，提供了在线程之间发送通知的便捷方式。

条件变量有两个主要操作：`wait()` 和 `notify()`。

线程调用 `wait(lock)` 表示希望等待来自另一个线程的通知。注意，`wait()` 传递了一个互斥锁（包装在 `std::unique_lock` 中）。当线程被通知时，条件变量将获取该锁。这意味着当 `wait()` 返回时，调用线程是锁的当前持有者。通常锁用于保护共享变量，线程现在需要检查该变量以确保其等待的条件为真。

例如，`tutorial/tutorial.cpp` 创建了N个线程。N-1个线程等待线程0的通知，收到通知后原子性地递增受共享互斥锁保护的计数器。

线程在条件变量上调用 `notify()` 来通知**恰好一个**等待该条件变量的线程，调用 `notify_all()` 来通知**所有**等待的线程。

在你的任务执行系统实现中，考虑如何使用 `notify_all()`：比如所有工作线程都在等待新的批量任务启动，而应用程序调用了 `run()` 来提供新任务执行。

额外参考：<https://www.modernescpp.com/index.php/condition-variables>

## C++原子操作（Atomics）

C++还提供了一种简单的方法使变量上的操作成为原子操作——只需创建 `std::atomic<T>` 类型的变量。例如创建一个支持原子递增的整数：

```cpp
std::atomic<int> my_counter;
```

现在 `my_counter` 上的操作，如 `my_counter++`，保证以原子方式执行。更多细节：<https://en.cppreference.com/w/cpp/atomic/atomic>
