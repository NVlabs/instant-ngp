/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// This file was taken from the tev image viewer and is re-released here
// under the NVIDIA Source Code License with permission from the author.

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/thread_pool.h>

#include <chrono>

NGP_NAMESPACE_BEGIN

using namespace std;

ThreadPool::ThreadPool()
: ThreadPool{thread::hardware_concurrency()} {}

ThreadPool::ThreadPool(size_t max_num_threads, bool force) {
	if (!force) {
		max_num_threads = min((size_t)thread::hardware_concurrency(), max_num_threads);
	}
	start_threads(max_num_threads);
}

ThreadPool::~ThreadPool() {
	shutdown_threads(m_threads.size());
}

void ThreadPool::start_threads(size_t num) {
	m_num_threads += num;
	for (size_t i = m_threads.size(); i < m_num_threads; ++i) {
		m_threads.emplace_back([this, i] {
			while (true) {
				unique_lock<mutex> lock{m_task_queue_mutex};

				// look for a work item
				while (i < m_num_threads && m_task_queue.empty()) {
					// if there are none wait for notification
					m_worker_condition.wait(lock);
				}

				if (i >= m_num_threads) {
					break;
				}

				function<void()> task{move(m_task_queue.front())};
				m_task_queue.pop_front();

				// Unlock the lock, so we can process the task without blocking other threads
				lock.unlock();

				task();
			}
		});
	}
}

void ThreadPool::shutdown_threads(size_t num) {
	auto num_to_close = min(num, m_num_threads);

	{
		lock_guard<mutex> lock{m_task_queue_mutex};
		m_num_threads -= num_to_close;
	}

	// Wake up all the threads to have them quit
	m_worker_condition.notify_all();
	for (auto i = 0u; i < num_to_close; ++i) {
		m_threads.back().join();
		m_threads.pop_back();
	}
}

void ThreadPool::set_n_threads(size_t num) {
	if (m_num_threads > num) {
		shutdown_threads(m_num_threads - num);
	} else if (m_num_threads < num) {
		start_threads(num - m_num_threads);
	}
}

void ThreadPool::flush_queue() {
	lock_guard<mutex> lock{m_task_queue_mutex};
	m_task_queue.clear();
}

NGP_NAMESPACE_END
