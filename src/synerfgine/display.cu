#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/marching_cubes.h>

#include <tiny-cuda-nn/common.h>

#include <synerfgine/camera.h>
#include <synerfgine/display.h>
#include <synerfgine/file.h>
#include <synerfgine/virtual_object.h>

namespace sng {

namespace inputs {
using namespace tcnn;
static vec3 i_camera_eye = camera_default::position;
}

bool Display::m_is_init = false;

void glfw_error_callback(int error, const char* description) {
	tlog::error() << "GLFW error #" << error << ": " << description;
}

GLFWwindow* Display::init_window(int resw, int resh, bool hidden) {
    m_window_res = {resw, resh};

#ifdef NGP_GUI
	m_glfw_window = renderer.create_glfw_window(m_window_res);
    ui.init_imgui(m_glfw_window);
#endif
	init_buffers();
	Display::m_is_init = true;
	return m_glfw_window;
}

GLFWwindow* Renderer::create_glfw_window(const ivec2& m_window_res) {
	this->m_window_res = m_window_res;
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        throw std::runtime_error{"GLFW could not be initialized."};
    }
    
    std::string title = "Synthetic Object NeRF Engine";
    m_glfw_window = glfwCreateWindow(m_window_res.x, m_window_res.y, title.c_str(), NULL, NULL);
    if (m_glfw_window == NULL) {
        throw std::runtime_error{"GLFW window could not be created."};
    }
    glfwMakeContextCurrent(m_glfw_window);
#ifdef _WIN32
    if (gl3wInit()) {
        throw std::runtime_error{"GL3W could not be initialized."};
    }
#else
    glewExperimental = 1;
    if (glewInit()) {
        throw std::runtime_error{"GLEW could not be initialized."};
    }
#endif
    glfwSwapInterval(0); // Disable vsync

    GLint gl_version_minor, gl_version_major;
    glGetIntegerv(GL_MINOR_VERSION, &gl_version_minor);
    glGetIntegerv(GL_MAJOR_VERSION, &gl_version_major);

    if (gl_version_major < 3 || (gl_version_major == 3 && gl_version_minor < 1)) {
        throw std::runtime_error{fmt::format("Unsupported OpenGL version {}.{}. instant-ngp requires at least OpenGL 3.1", gl_version_major, gl_version_minor)};
    }

    tlog::success() << "Initialized OpenGL version " << glGetString(GL_VERSION);

	// TODO: Fix window size crashing issues
	// glfwSetWindowSizeCallback(m_glfw_window, );

	init_opengl_shaders();

	return m_glfw_window;
}

void Ui::init_imgui(GLFWwindow* m_glfw_window) {
	this->m_glfw_window = m_glfw_window;

	float xscale, yscale;
	glfwGetWindowContentScale(m_glfw_window, &xscale, &yscale);

	// IMGUI init
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	// By default, imgui places its configuration (state of the GUI -- size of windows,
	// which regions are expanded, etc.) in ./imgui.ini relative to the working directory.
	// Instead, we would like to place imgui.ini in the directory that instant-ngp project
	// resides in.
	static std::string ini_filename;
	ini_filename = (Utils::get_root_dir()/"imgui.ini").string();
	io.IniFilename = ini_filename.c_str();

	// New ImGui event handling seems to make camera controls laggy if input trickling is true.
	// So disable input trickling.
	io.ConfigInputTrickleEventQueue = false;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(m_glfw_window, true);
	ImGui_ImplOpenGL3_Init("#version 140");

	ImGui::GetStyle().ScaleAllSizes(xscale);
	ImFontConfig font_cfg;
	font_cfg.SizePixels = 13.0f * xscale;
	io.Fonts->AddFontDefault(&font_cfg);
}

void Ui::imgui(SyntheticWorld& syn_world, float frame_time) {
	static std::string imgui_error_string = "";

	if (ImGui::Begin("Load Virtual Object")) {
		ImGui::Text("Add Virtual Object (.obj only)");
		ImGui::InputText("##PathFile", sng::virtual_object_fp, 1024);
		ImGui::SameLine();
		static std::string vo_path_load_error_string = "";
		if (ImGui::Button("Load")) {
			try {
				syn_world.create_object(sng::virtual_object_fp);
			} catch (const std::exception& e) {
				ImGui::OpenPopup("Virtual object path load error");
				vo_path_load_error_string = std::string{"Failed to load object path: "} + e.what();
			}
		}
		if (ImGui::BeginPopupModal("Virtual object path load error", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
			ImGui::Text("%s", vo_path_load_error_string.c_str());
			if (ImGui::Button("OK", ImVec2(120, 0))) {
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}
		if (ImGui::CollapsingHeader("Virtual Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
			auto& objs = syn_world.objects();
			std::string to_remove;
			size_t k = 0;
			for (auto& vo : objs) {
				std::string delete_button_name = std::to_string(k) + ". Delete ";
				delete_button_name.append(vo.first);
				if (ImGui::Button(delete_button_name.c_str())) {
					to_remove = vo.first;
					break;
				}
				vo.second.imgui();
				++k;
			}
			if (!to_remove.empty()) {
				syn_world.delete_object(to_remove);
			}
		}
	}
	ImGui::End();
	if (ImGui::Begin("Camera")) {
		auto rd = syn_world.camera().view_pos();
		ImGui::Text("View Pos: %f, %f, %f", rd.r, rd.g, rd.b);
		rd = syn_world.camera().view_dir();
		ImGui::Text("View Dir: %f, %f, %f", rd.r, rd.g, rd.b);
		rd = syn_world.camera().look_at();
		ImGui::Text("Look At: %f, %f, %f", rd.r, rd.g, rd.b);
		rd = syn_world.camera().sun_pos();
		ImGui::Text("Sun Pos: %f, %f, %f", rd.r, rd.g, rd.b);
		float fps = !frame_time ? 1000.0f : (1000.0f / frame_time);
		ImGui::Text("Frame: %.2f ms (%.1f FPS)", frame_time, fps);
		if (ImGui::Button("Reset Camera")) {
			syn_world.mut_camera().reset_camera();
		}
	// 	if (ImGui::SliderFloat3("Camera Position", inputs::i_camera_eye.data(), -10.0, 10.0)) {
	// 		// syn_world.camera_position(inputs::i_camera_eye);
	// 	}
	// 	if (ImGui::Button("Reset Position")) {
	// 		inputs::i_camera_eye = camera_default::position;
	// 		// syn_world.camera_position(inputs::i_camera_eye);
	// 	}
	// 	if (ImGui::SliderFloat3("Camera Look At", inputs::i_camera_at.data(), -10.0, 10.0)) {
	// 		// syn_world.camera_look_at(inputs::i_camera_at);
	// 	}
	// 	if (ImGui::Button("Reset Look At")) {
	// 		inputs::i_camera_at = camera_default::lookat;
	// 		// syn_world.camera_look_at(inputs::i_camera_at);
	// 	}
	}
	ImGui::End();
}

void Renderer::init_opengl_shaders() {
	static const char* shader_vert = R"glsl(#version 140
		out vec2 UVs;
		void main() {
			UVs = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
			gl_Position = vec4(UVs * 2.0 - 1.0, 0.0, 1.0);
		})glsl";

	static const char* shader_frag = R"glsl(#version 140
		in vec2 UVs;
		out vec4 frag_color;
		uniform sampler2D rgba_texture;
		uniform sampler2D depth_texture;

		struct FoveationWarp {
			float al, bl, cl;
			float am, bm;
			float ar, br, cr;
			float switch_left, switch_right;
			float inv_switch_left, inv_switch_right;
		};

		uniform FoveationWarp warp_x;
		uniform FoveationWarp warp_y;

		float unwarp(in FoveationWarp warp, float y) {
			y = clamp(y, 0.0, 1.0);
			if (y < warp.inv_switch_left) {
				return (sqrt(-4.0 * warp.al * warp.cl + 4.0 * warp.al * y + warp.bl * warp.bl) - warp.bl) / (2.0 * warp.al);
			} else if (y > warp.inv_switch_right) {
				return (sqrt(-4.0 * warp.ar * warp.cr + 4.0 * warp.ar * y + warp.br * warp.br) - warp.br) / (2.0 * warp.ar);
			} else {
				return (y - warp.bm) / warp.am;
			}
		}

		vec2 unwarp(in vec2 pos) {
			return vec2(unwarp(warp_x, pos.x), unwarp(warp_y, pos.y));
		}

		void main() {
			vec2 tex_coords = UVs;
			tex_coords.y = 1.0 - tex_coords.y;
			tex_coords = unwarp(tex_coords);
			frag_color = texture(rgba_texture, tex_coords.xy);
			//Uncomment the following line of code to visualize debug the depth buffer for debugging.
			// frag_color = vec4(vec3(texture(depth_texture, tex_coords.xy).r), 1.0);
			gl_FragDepth = texture(depth_texture, tex_coords.xy).r;
		})glsl";

	GLuint vert = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vert, 1, &shader_vert, NULL);
	glCompileShader(vert);
	ngp::check_shader(vert, "Blit vertex shader", false);

	GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(frag, 1, &shader_frag, NULL);
	glCompileShader(frag);
	ngp::check_shader(frag, "Blit fragment shader", false);

	m_blit_program = glCreateProgram();
	glAttachShader(m_blit_program, vert);
	glAttachShader(m_blit_program, frag);
	glLinkProgram(m_blit_program);
	ngp::check_shader(m_blit_program, "Blit shader program", true);

	glDeleteShader(vert);
	glDeleteShader(frag);

	glGenVertexArrays(1, &m_blit_vao);
}

void Display::init_buffers() {
	// Make sure there's at least one usable render texture
	m_rgba_render_textures = std::make_shared<GLTexture>();
	m_depth_render_textures = std::make_shared<GLTexture>();

	m_render_buffer = std::make_shared<CudaRenderBuffer>(m_rgba_render_textures, m_depth_render_textures);
	m_render_buffer->resize(m_view_res);
	m_render_buffer->disable_dlss();
}

bool Display::begin_frame(CudaDevice& device, bool& is_dirty) {
	if (glfwWindowShouldClose(m_glfw_window) || ImGui::IsKeyPressed(GLFW_KEY_ESCAPE) || ImGui::IsKeyPressed(GLFW_KEY_Q)) {
		destroy();
		return false;
	}

	glfwPollEvents();
	glfwGetFramebufferSize(m_glfw_window, &m_window_res.x, &m_window_res.y);
	if (is_dirty) {
		device.device_guard();
		m_render_buffer->resize(m_window_res);
		is_dirty = false;
	}
	return ui.begin_frame() && renderer.begin_frame();
}

bool Ui::begin_frame() {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGuizmo::BeginFrame();

	return true;
}

bool Renderer::begin_frame() {
	return true;
}

void Display::end_frame() {
	renderer.end_frame();
	ui.end_frame();
	auto time_now = std::chrono::system_clock::now();
	m_last_frame_time = (float)std::chrono::duration_cast<std::chrono::milliseconds>(time_now - m_last_timestamp).count();
	m_last_timestamp = time_now;
 }

void Ui::end_frame() {
	ImGui::EndFrame();
}

void Renderer::end_frame() {
}

bool Display::present(CudaDevice& device, SyntheticWorld& syn_world) {
	ui.imgui(syn_world, m_last_frame_time);
	m_render_buffer->set_hidden_area_mask(nullptr);
	return renderer.present({1,1}, m_rgba_render_textures, m_depth_render_textures, device); 
}

bool Renderer::present(const ivec2& m_n_views, std::shared_ptr<ngp::GLTexture> rgba, std::shared_ptr<ngp::GLTexture> depth, CudaDevice& device) { 
	if (!m_glfw_window) {
		throw std::runtime_error{"Window must be initialized to be presented."};
	}

	// Make sure all the cuda code finished its business here
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
	glfwMakeContextCurrent(m_glfw_window);
	int display_w, display_h;
	glfwGetFramebufferSize(m_glfw_window, &display_w, &display_h);

	// IMAGE RENDER
	glViewport(0, 0, display_w, display_h);
	glClearColor(0.0f, 0.0f, 0.5f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_BLEND);
	glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
	glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	ivec2 extent = {(int)((float)display_w / m_n_views.x), (int)((float)display_h / m_n_views.y)};

    {
		auto n_elements = m_window_res.x * m_window_res.y;
		if (n_elements != m_cpu_frame_buffer.size()) m_cpu_frame_buffer.resize(n_elements);
		if (n_elements != m_cpu_depth_buffer.size()) m_cpu_depth_buffer.resize(n_elements);
        CUDA_CHECK_THROW(cudaMemcpyAsync(m_cpu_frame_buffer.data(), device.render_buffer_view().frame_buffer, n_elements * sizeof(vec4), cudaMemcpyDeviceToHost, device.stream()));
        CUDA_CHECK_THROW(cudaMemcpyAsync(m_cpu_depth_buffer.data(), device.render_buffer_view().depth_buffer, n_elements * sizeof(float), cudaMemcpyDeviceToHost, device.stream()));
		auto rgba_size = rgba->resolution();
		auto depth_size = depth->resolution();
		CUDA_CHECK_THROW(cudaStreamSynchronize(device.stream()));

		glBindTexture(GL_TEXTURE_2D, rgba->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, rgba_size.x, rgba_size.y, 0, GL_RGBA, GL_FLOAT, m_cpu_frame_buffer.data());
		glBindTexture(GL_TEXTURE_2D, depth->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, depth_size.x, depth_size.y, 0, GL_RED, GL_FLOAT, m_cpu_depth_buffer.data());
    }

	ivec2 top_left{0, display_h - extent.y};
	// blit_texture(m_foveated_rendering_visualize ? Foveation{} : view.foveation, m_rgba_render_textures.at(i)->texture(), m_foveated_rendering ? GL_LINEAR : GL_NEAREST, m_depth_render_textures.at(i)->texture(), 0, top_left, extent);
	// rgba->blit_from_cuda_mapping()
	blit_texture(ngp::Foveation{}, rgba->texture(), GL_LINEAR, depth->texture(), 0, top_left, extent);
	glFinish();

	// UI DRAWING
	glViewport(0, 0, display_w, display_h);

	ImDrawList* list = ImGui::GetBackgroundDrawList();
	list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);

	// Visualizations are only meaningful when rendering a single view
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glfwSwapBuffers(m_glfw_window);

	// Make sure all the OGL code finished its business here.
	// Any code outside of this function needs to be able to freely write to
	// textures without being worried about interfering with rendering.
	glFinish();

	return true;
}

void Renderer::blit_texture(const ngp::Foveation& foveation, GLint rgba_texture, GLint rgba_filter_mode, 
	GLint depth_texture, GLint framebuffer, const ivec2& offset, const ivec2& resolution) {
	if (m_blit_program == 0) {
		return;
	}

	// Blit image to OpenXR swapchain.
	// Note that the OpenXR swapchain is 8bit while the rendering is in a float texture.
	// As some XR runtimes do not support float swapchains, we can't render into it directly.

	bool tex = glIsEnabled(GL_TEXTURE_2D);
	bool depth = glIsEnabled(GL_DEPTH_TEST);
	bool cull = glIsEnabled(GL_CULL_FACE);

	if (!tex) 
		glEnable(GL_TEXTURE_2D);
	if (!depth) 
		glEnable(GL_DEPTH_TEST);
	if (cull) 
		glDisable(GL_CULL_FACE);

	glDepthFunc(GL_ALWAYS);
	glDepthMask(GL_TRUE);

	glBindVertexArray(m_blit_vao);
	glUseProgram(m_blit_program);
	auto rgba_uniform = glGetUniformLocation(m_blit_program, "rgba_texture");
	auto depth_uniform = glGetUniformLocation(m_blit_program, "depth_texture");
	glUniform1i(rgba_uniform, 0);
	glUniform1i(depth_uniform, 1);

	auto bind_warp = [&](const ngp::FoveationPiecewiseQuadratic& warp, const std::string& uniform_name) {
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".al").c_str()), warp.al);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".bl").c_str()), warp.bl);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".cl").c_str()), warp.cl);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".am").c_str()), warp.am);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".bm").c_str()), warp.bm);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".ar").c_str()), warp.ar);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".br").c_str()), warp.br);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".cr").c_str()), warp.cr);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".switch_left").c_str()), warp.switch_left);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".switch_right").c_str()), warp.switch_right);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".inv_switch_left").c_str()), warp.inv_switch_left);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".inv_switch_right").c_str()), warp.inv_switch_right);
	};

	bind_warp(foveation.warp_x, "warp_x");
	bind_warp(foveation.warp_y, "warp_y");

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, depth_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, rgba_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, rgba_filter_mode);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, rgba_filter_mode);

	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glViewport(offset.x, offset.y, resolution.x, resolution.y);

	glDrawArrays(GL_TRIANGLES, 0, 3);

	glBindVertexArray(0);
	glUseProgram(0);

	glDepthFunc(GL_LESS);

	// restore old state
	if (!tex) glDisable(GL_TEXTURE_2D);
	if (!depth) glDisable(GL_DEPTH_TEST);
	if (cull) glEnable(GL_CULL_FACE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Display::destroy() {
#ifndef NGP_GUI
	throw std::runtime_error{"destroy_window failed: NGP was built without GUI support"};
#else
	if (!Display::m_is_init) {
		return;
	}

	m_render_buffer = nullptr;

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(m_glfw_window);
	glfwTerminate();

	m_glfw_window = nullptr;
	m_is_init = false;
#endif //NGP_GUI
}

}