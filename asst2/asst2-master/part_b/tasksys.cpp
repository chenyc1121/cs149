#include "tasksys.h"


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
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
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
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
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

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
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

bool TaskSystemParallelThreadPoolSleeping::has_ready_work(){
    for(TaskID bid:_ready_bulks){
        const TaskBulk& b=_bulks[bid];
        if(!b.completed && b.next_task <b.num_total_tasks){
            return true;
        }
    }
    return false;
}
TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): 
    ITaskSystem(num_threads),
    _num_threads(num_threads),
    _next_bulk_id(0),
    _num_incomplete_bulks(0)
    ,_exit(false){
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    for(int t=0;t<_num_threads;t++){
        _threads.emplace_back([this,t](){
            while (true)
            {
                std::unique_lock<std::mutex> lk(_mutex);
                while(!_exit && !has_ready_work()){
                    _cv.wait(lk);
                }
                if(_exit)break;

                TaskID bulk_id=-1;
                int task_idx=-1;
                for(TaskID bid:_ready_bulks){
                    TaskBulk& b=_bulks[bid];
                    if(!b.completed && b.next_task < b.num_total_tasks){
                        task_idx=b.next_task++;
                        bulk_id=bid;
                        break;
                    }
                }
                if (bulk_id==-1){
                    continue;
                }

                IRunnable* r=_bulks[bulk_id].runnable;
                int ntt=_bulks[bulk_id].num_total_tasks;
                lk.unlock();
                r->runTask(task_idx,ntt);
                lk.lock();
                TaskBulk& me=_bulks[bulk_id];
                me.tasks_completed++;
                if(me.tasks_completed==me.num_total_tasks){
                    me.completed=true;
                    _num_incomplete_bulks--;

                    for(TaskID dep_id:me.dependents){
                        TaskBulk& dep_bulk=_bulks[dep_id];
                        dep_bulk.remaining_deps--;
                        if(dep_bulk.remaining_deps==0){
                            _ready_bulks.push_back(dep_id);
                        }
                    }
                    _cv.notify_all();
                }
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
        _exit=true;
    }
    _cv.notify_all();
    for(auto & t: _threads){
        t.join();
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    TaskID id = runAsyncWithDeps(runnable, num_total_tasks, {});

    // 等待这批任务完成（不等待其他异步批次）
    std::unique_lock<std::mutex> lk(_mutex);
    while (!_bulks[id].completed) {
        _cv.wait(lk);
    }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    // for (int i = 0; i < num_total_tasks; i++) {
    //     runnable->runTask(i, num_total_tasks);
    // }

    // return 0;
    std::unique_lock<std::mutex> lk(_mutex);

    TaskID my_id=_next_bulk_id++;
    _bulks.push_back({my_id,runnable,num_total_tasks,0,0,0,false,{}});
    TaskBulk& me=_bulks[my_id];

    int incomplete_deps=0;
    for(TaskID dep_id:deps){
        TaskBulk& dep=_bulks[dep_id];
        if(!dep.completed){
            incomplete_deps++;
            dep.dependents.push_back(my_id);
        }
    }
    me.remaining_deps=incomplete_deps;

    if(me.remaining_deps==0){
        _ready_bulks.push_back(my_id);
    }
    
    _num_incomplete_bulks++;

    _cv.notify_all();

    return my_id;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    std::unique_lock<std::mutex> lk(_mutex);
    while (_num_incomplete_bulks > 0) {
        _cv.wait(lk);
    }
}
