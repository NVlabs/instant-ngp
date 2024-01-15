#pragma once

#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/thread_pool.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/multi_stream.h>

namespace sng {

using namespace ngp;
using namespace tcnn;

class CudaDevice {
public:
    struct Data {
        GPUMemory<uint8_t> density_grid_bitfield;
        uint8_t* density_grid_bitfield_ptr;

        GPUMemory<network_precision_t> params;
        std::shared_ptr<Buffer2D<uint8_t>> hidden_area_mask;
    };

    CudaDevice(int id, bool is_primary);

    CudaDevice(const CudaDevice&) = delete;
    CudaDevice& operator=(const CudaDevice&) = delete;

    CudaDevice(CudaDevice&&) = default;
    CudaDevice& operator=(CudaDevice&&) = default;

    ScopeGuard device_guard();

    int id() const {
        return m_id;
    }

    bool is_primary() const {
        return m_is_primary;
    }

    std::string name() const {
        return cuda_device_name(m_id);
    }

    int compute_capability() const {
        return cuda_compute_capability(m_id);
    }

    cudaStream_t stream() const {
        return m_stream->get();
    }

    void wait_for(cudaStream_t stream) const {
        CUDA_CHECK_THROW(cudaEventRecord(m_primary_device_event.event, stream));
        m_stream->wait_for(m_primary_device_event.event);
    }

    void signal(cudaStream_t stream) const {
        m_stream->signal(stream);
    }

    const CudaRenderBufferView& render_buffer_view() const {
        return m_render_buffer_view;
    }

    void set_render_buffer_view(const CudaRenderBufferView& view) {
        m_render_buffer_view = view;
    }

    Data& data() const {
        return *m_data;
    }

    bool dirty() const {
        return m_dirty;
    }

    void set_dirty(bool value) {
        m_dirty = value;
    }

    void clear() {
        m_data = std::make_unique<Data>();
        m_render_buffer_view = {};
        set_dirty(true);
    }

    template <class F>
    auto enqueue_task(F&& f) -> std::future<std::result_of_t<F()>> {
        if (is_primary()) {
            return std::async(std::launch::deferred, std::forward<F>(f));
        } else {
            return m_render_worker->enqueue_task(std::forward<F>(f));
        }
    }

private:
    int m_id;
    bool m_is_primary;
    std::unique_ptr<StreamAndEvent> m_stream;
    struct Event {
        Event() {
            CUDA_CHECK_THROW(cudaEventCreate(&event));
        }

        ~Event() {
            cudaEventDestroy(event);
        }

        Event(const Event&) = delete;
        Event& operator=(const Event&) = delete;
        Event(Event&& other) { *this = std::move(other); }
        Event& operator=(Event&& other) {
            std::swap(event, other.event);
            return *this;
        }

        cudaEvent_t event = {};
    };
    Event m_primary_device_event;
    std::unique_ptr<Data> m_data;
    CudaRenderBufferView m_render_buffer_view = {};

    bool m_dirty = true;

    std::unique_ptr<ThreadPool> m_render_worker;
};

inline CudaDevice::CudaDevice(int id, bool is_primary) : m_id{id}, m_is_primary{is_primary} {
	auto guard = device_guard();
	m_stream = std::make_unique<StreamAndEvent>();
	m_data = std::make_unique<Data>();
	m_render_worker = std::make_unique<ThreadPool>(is_primary ? 0u : 1u);
}

inline ScopeGuard CudaDevice::device_guard() {
	int prev_device = cuda_device();
	if (prev_device == m_id) {
		return {};
	}

	set_cuda_device(m_id);
	return ScopeGuard{[prev_device]() {
		set_cuda_device(prev_device);
	}};
}

inline int find_cuda_device() {
	int active_device;
    CUDA_CHECK_THROW(cudaGetDevice(&active_device));
	cudaDeviceProp props;
	CUDA_CHECK_THROW(cudaGetDeviceProperties(&props, active_device));
	int active_compute_capability = props.major * 10 + props.minor;
	tlog::success() << fmt::format(
		"Initialized CUDA {}. Active GPU is #{}: {} [{}]",
		cuda_runtime_version_string(),
		active_device,
		cuda_device_name(),
		active_compute_capability
	);

	if (active_compute_capability < MIN_GPU_ARCH) {
		tlog::warning() << "Insufficient compute capability " << active_compute_capability << " detected.";
		tlog::warning() << "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly.";
	}

    return active_device;
}

inline void sync_device(CudaRenderBuffer& render_buffer, CudaDevice& device, StreamAndEvent& stream) {
	if (!device.dirty()) {
		return;
	}

	if (device.is_primary()) {
		device.data().hidden_area_mask = render_buffer.hidden_area_mask();
		device.set_dirty(false);
		return;
	}

	stream.signal(device.stream());
	device.set_dirty(false);
	device.signal(stream.get());
}

// From https://stackoverflow.com/questions/20843271/passing-a-non-copyable-closure-object-to-stdfunction-parameter
template <class F>
auto create_copyable_function(F&& f) {
	using dF = std::decay_t<F>;
	auto spf = std::make_shared<dF>(std::forward<F>(f));
	return [spf](auto&&... args) -> decltype(auto) {
		return (*spf)( decltype(args)(args)... );
	};
}

inline ScopeGuard use_device(cudaStream_t stream, CudaRenderBuffer& render_buffer, CudaDevice& device) {
	device.wait_for(stream);

	if (device.is_primary()) {
		device.set_render_buffer_view(render_buffer.view());
		return ScopeGuard{[&device, stream]() {
			device.set_render_buffer_view({});
			device.signal(stream);
		}};
	}

	int active_device = cuda_device();
	auto guard = device.device_guard();

	size_t n_pixels = product(render_buffer.in_resolution());

	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<vec4, float>(device.stream(), &alloc, n_pixels, n_pixels);

	device.set_render_buffer_view({
		std::get<0>(scratch),
		std::get<1>(scratch),
		render_buffer.in_resolution(),
		render_buffer.spp(),
		device.data().hidden_area_mask,
	});

	return ScopeGuard{create_copyable_function([&render_buffer, &device, guard=std::move(guard), alloc=std::move(alloc), active_device, stream]() {
		// Copy device's render buffer's data onto the original render buffer
		CUDA_CHECK_THROW(cudaMemcpyPeerAsync(render_buffer.frame_buffer(), active_device, device.render_buffer_view().frame_buffer, device.id(), product(render_buffer.in_resolution()) * sizeof(vec4), device.stream()));
		CUDA_CHECK_THROW(cudaMemcpyPeerAsync(render_buffer.depth_buffer(), active_device, device.render_buffer_view().depth_buffer, device.id(), product(render_buffer.in_resolution()) * sizeof(float), device.stream()));

		device.set_render_buffer_view({});
		device.signal(stream);
	})};
}

}