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

ThreadPool::ThreadPool(size_t maxNumThreads, bool force) {
	if (!force) {
		maxNumThreads = min((size_t)thread::hardware_concurrency(), maxNumThreads);
	}
	startThreads(maxNumThreads);
	mNumTasksInSystem.store(0);
}

ThreadPool::~ThreadPool() {
	shutdownThreads(mThreads.size());
}

void ThreadPool::startThreads(size_t num) {
	mNumThreads += num;
	for (size_t i = mThreads.size(); i < mNumThreads; ++i) {
		mThreads.emplace_back([this, i] {
			while (true) {
				unique_lock<mutex> lock{mTaskQueueMutex};

				// look for a work item
				while (i < mNumThreads && mTaskQueue.empty()) {
					// if there are none wait for notification
					mWorkerCondition.wait(lock);
				}

				if (i >= mNumThreads) {
					break;
				}

				function<void()> task{move(mTaskQueue.front())};
				mTaskQueue.pop_front();

				// Unlock the lock, so we can process the task without blocking other threads
				lock.unlock();

				task();

				mNumTasksInSystem--;

				{
					unique_lock<mutex> localLock{mSystemBusyMutex};

					if (mNumTasksInSystem == 0) {
						mSystemBusyCondition.notify_all();
					}
				}
			}
		});
	}
}

void ThreadPool::shutdownThreads(size_t num) {
	auto numToClose = min(num, mNumThreads);

	{
		lock_guard<mutex> lock{mTaskQueueMutex};
		mNumThreads -= numToClose;
	}

	// Wake up all the threads to have them quit
	mWorkerCondition.notify_all();
	for (auto i = 0u; i < numToClose; ++i) {
		mThreads.back().join();
		mThreads.pop_back();
	}
}

void ThreadPool::waitUntilFinished() {
	unique_lock<mutex> lock{mSystemBusyMutex};

	if (mNumTasksInSystem == 0) {
		return;
	}

	mSystemBusyCondition.wait(lock);
}

void ThreadPool::waitUntilFinishedFor(const chrono::microseconds Duration) {
	unique_lock<mutex> lock{mSystemBusyMutex};

	if (mNumTasksInSystem == 0) {
		return;
	}

	mSystemBusyCondition.wait_for(lock, Duration);
}

void ThreadPool::flushQueue() {
	lock_guard<mutex> lock{mTaskQueueMutex};

	mNumTasksInSystem -= mTaskQueue.size();
	mTaskQueue.clear();
}

NGP_NAMESPACE_END
