#include <neural-graphics-primitives/marching_cubes.h>

#include <synerfgine/display.h>

namespace sng {

void glfw_error_callback(int error, const char* description) {
	tlog::error() << "GLFW error #" << error << ": " << description;
}

void Display::init_window(int resw, int resh, bool hidden) {
    m_window_res = {resw, resh};

#ifdef NGP_GUI
	renderer.init_glfw(m_glfw_window);
    ui.init(m_glfw_window);
#endif
	renderer.init_buffers();
}

void Renderer::init_glfw(GLFWwindow* m_glfw_window, const ivec2& m_window_res) {
	this->m_window_res = m_window_res;
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        throw std::runtime_error{"GLFW could not be initialized."};
    }
    
    glfwWindowHint(GLFW_VISIBLE, hidden ? GLFW_FALSE : GLFW_TRUE);
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
}

void Ui::init_imgui(const GLFWwindow* m_glfw_window) {
	// IMGUI init
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	// By default, imgui places its configuration (state of the GUI -- size of windows,
	// which regions are expanded, etc.) in ./imgui.ini relative to the working directory.
	// Instead, we would like to place imgui.ini in the directory that instant-ngp project
	// resides in.
	static std::string ini_filename;
	ini_filename = (root_dir()/"imgui.ini").str();
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

	this->m_glfw_window = m_glfw_window;
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

void Renderer::init_buffers() {

}

}