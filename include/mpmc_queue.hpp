#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

namespace mpi {

template <typename T>
class MPMCQueue {
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
    bool is_finished_ = false;
    
public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(value));
        }
        cond_var_.notify_one();
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        cond_var_.wait(lock, [this]{ return !queue_.empty() || is_finished_; });

        if (queue_.empty() && is_finished_) {
            return false; 
        }

        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void signal_finished() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_finished_ = true;
        }
        cond_var_.notify_all();
    }
};

}
