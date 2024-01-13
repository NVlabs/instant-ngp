#include <synerfgine/engine.h>

namespace sng {

void Engine::init(int res_width, int res_height) {
    display.init_window(res_width, res_height, false);
}

bool Engine::frame() {
    if (!display.begin_frame()) return false;
    display.present();
    display.end_frame();
    return true;
}

}