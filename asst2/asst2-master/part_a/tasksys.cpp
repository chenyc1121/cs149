#include "tasksys.h"
#include <thread>
#include <atomic>


IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    _num_threads=num_threads;
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    // for (int i = 0; i < num_total_tasks; i++) {
    //     runnable->runTask(i, num_total_tasks);
    // }
    std::thread threads[_num_threads];
    for(int t=0;t<_num_threads;t++){
        int st=(t*num_total_tasks)/_num_threads;
        int ed=((t+1)*num_total_tasks)/_num_threads;
        threads[t]=std::thread([=](){
            for(int i=st;i<ed;i++){
                runnable->runTask(i,num_total_tasks);
            }
        });
    }
    for (int t = 0; t < _num_threads; t++) {
        threads[t].join();
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads),_num_threads(num_threads),_runnable(nullptr),_num_total_tasks(0) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    _go.store(0);
    _done_count.store(0);
    _exit.store(0);
    for(int t=0;t<num_threads;t++){
        _threads.emplace_back([this,t](){
            while(true){
                while(! _exit.load() && !_go.load()){}

                if(_exit.load())break;
                int start=(t*_num_total_tasks)/_num_threads;
                int end=((t+1)*_num_total_tasks)/_num_threads;
                for (int i=start;i<end;i++){
                    _runnable->runTask(i,_num_total_tasks);
                }
                _done_count.fetch_add(1);
                while (_go.load() && !_exit.load());
            }
        });
    }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    _exit.store(true);
    for (auto& t : _threads) {
        t.join();
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    // for (int i = 0; i < num_total_tasks; i++) {
    //     runnable->runTask(i, num_total_tasks);
    // }
    _runnable=runnable;
    _num_total_tasks=num_total_tasks;
    _done_count.store(0);
    _go.store(true);
    while (_done_count.load() < _num_threads);
    _go.store(false);
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads),_num_threads(num_threads),_runnable(nullptr),_num_total_tasks(0),_tasks_done(0),_go(false),_exit(false){
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    for(int t=0;t<num_threads;t++){
        _threads.emplace_back([this,t](){
            while (true)
            {
                std::unique_lock<std::mutex> lk(_mutex);
                while(!_go && !_exit){
                    _cv.wait(lk);
                }
                if(_exit)break;
                
                int start = (t * _num_total_tasks) / _num_threads;
                int end = ((t + 1) * _num_total_tasks) / _num_threads;

                lk.unlock();

                for (int i = start; i < end; i++) {
                    _runnable->runTask(i, _num_total_tasks);
                }

                lk.lock();
                _tasks_done++;

                if (_tasks_done == _num_threads) {
                    _cv.notify_one();     
                }

                while (_go && !_exit) {
                    _cv.wait(lk);
                }
                _cv.notify_all();
            }
        });
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    {
        std::lock_guard<std::mutex> lk(_mutex);
        _exit = true;
    }
    _cv.notify_all();                     // 叫醒所有休眠的线程
    for (auto& t : _threads) {
        t.join();                         // 等它们退出
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    // for (int i = 0; i < num_total_tasks; i++) {
    //     runnable->runTask(i, num_total_tasks);
    // }
    {
        std::unique_lock<std::mutex> lk(_mutex);
        _runnable=runnable;
        _num_total_tasks=num_total_tasks;
        _tasks_done=0;
        _go=true;
    }
    _cv.notify_all();
    {
        std::unique_lock<std::mutex> lk(_mutex);
        while(_tasks_done<_num_threads){
            _cv.wait(lk);
        }
        _go=false;
    }
    _cv.notify_all();
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
