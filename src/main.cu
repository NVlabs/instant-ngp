/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   main.cu
 *  @author Thomas Müller, NVIDIA
 */

#include <neural-graphics-primitives/testbed.h>

#include <tiny-cuda-nn/common.h>

#include <args/args.hxx>

#include <filesystem/path.h>

using namespace args;
using namespace ngp;
using namespace std;
using namespace tcnn;
namespace fs = ::filesystem;

int main(int argc, char** argv) {
	ArgumentParser parser{
		"neural graphics primitives\n"
		"version " NGP_VERSION,
		"",
	};

	HelpFlag help_flag{
		parser,
		"HELP",
		"Display this help menu.",
		{'h', "help"},
	};

	ValueFlag<string> mode_flag{
		parser,
		"MODE",
		"Mode can be 'nerf', 'sdf', or 'image' or 'volume'. Inferred from the scene if unspecified.",
		{'m', "mode"},
	};

	ValueFlag<string> network_config_flag{
		parser,
		"CONFIG",
		"Path to the network config. Uses the scene's default if unspecified.",
		{'n', 'c', "network", "config"},
	};

	Flag no_gui_flag{
		parser,
		"NO_GUI",
		"Disables the GUI and instead reports training progress on the command line.",
		{"no-gui"},
	};

	Flag no_train_flag{
		parser,
		"NO_TRAIN",
		"Disables training on startup.",
		{"no-train"},
	};

	ValueFlag<string> scene_flag{
		parser,
		"SCENE",
		"The scene to load. Can be NeRF dataset, a *.obj mesh for training a SDF, an image, or a *.nvdb volume.",
		{'s', "scene"},
	};

	ValueFlag<string> snapshot_flag{
		parser,
		"SNAPSHOT",
		"Optional snapshot to load upon startup.",
		{"snapshot"},
	};

	ValueFlag<uint32_t> width_flag{
		parser,
		"WIDTH",
		"Resolution width of the GUI.",
		{"width"},
	};

	ValueFlag<uint32_t> height_flag{
		parser,
		"HEIGHT",
		"Resolution height of the GUI.",
		{"height"},
	};

	Flag version_flag{
		parser,
		"VERSION",
		"Display the version of neural graphics primitives.",
		{'v', "version"},
	};

	// Parse command line arguments and react to parsing
	// errors using exceptions.
	try {
		parser.ParseCLI(argc, argv);
	} catch (const Help&) {
		cout << parser;
		return 0;
	} catch (const ParseError& e) {
		cerr << e.what() << endl;
		cerr << parser;
		return -1;
	} catch (const ValidationError& e) {
		cerr << e.what() << endl;
		cerr << parser;
		return -2;
	}

	if (version_flag) {
		tlog::none() << "neural graphics primitives version " NGP_VERSION;
		return 0;
	}

	try {
		ETestbedMode mode;
		if (!mode_flag) {
			if (!scene_flag) {
				tlog::error() << "Must specify either a mode or a scene";
				return 1;
			}

			fs::path scene_path = get(scene_flag);
			if (!scene_path.exists()) {
				tlog::error() << "Scene path " << scene_path << " does not exist.";
				return 1;
			}

			if (scene_path.is_directory() || equals_case_insensitive(scene_path.extension(), "json")) {
				mode = ETestbedMode::Nerf;
			} else if (equals_case_insensitive(scene_path.extension(), "obj") || equals_case_insensitive(scene_path.extension(), "stl")) {
				mode = ETestbedMode::Sdf;
			} else if (equals_case_insensitive(scene_path.extension(), "nvdb")) {
				mode = ETestbedMode::Volume;
			} else {
				mode = ETestbedMode::Image;
			}
		} else {
			auto mode_str = get(mode_flag);
			if (equals_case_insensitive(mode_str, "nerf")) {
				mode = ETestbedMode::Nerf;
			} else if (equals_case_insensitive(mode_str, "sdf")) {
				mode = ETestbedMode::Sdf;
			} else if (equals_case_insensitive(mode_str, "image")) {
				mode = ETestbedMode::Image;
			} else if (equals_case_insensitive(mode_str, "volume")) {
				mode = ETestbedMode::Volume;
			} else {
				tlog::error() << "Mode must be one of 'nerf', 'sdf', 'image', and 'volume'.";
				return 1;
			}
		}

		Testbed testbed{mode};

		if (scene_flag) {
			fs::path scene_path = get(scene_flag);
			if (!scene_path.exists()) {
				tlog::error() << "Scene path " << scene_path << " does not exist.";
				return 1;
			}
			testbed.load_training_data(scene_path.str());
		}

		std::string mode_str;
		switch (mode) {
			case ETestbedMode::Nerf:   mode_str = "nerf";   break;
			case ETestbedMode::Sdf:    mode_str = "sdf";    break;
			case ETestbedMode::Image:  mode_str = "image";  break;
			case ETestbedMode::Volume: mode_str = "volume"; break;
		}

		if (snapshot_flag) {
			// Load network from a snapshot if one is provided
			fs::path snapshot_path = get(snapshot_flag);
			if (!snapshot_path.exists()) {
				tlog::error() << "Snapshot path " << snapshot_path << " does not exist.";
				return 1;
			}

			testbed.load_snapshot(snapshot_path.str());
			testbed.m_train = false;
		} else {
			// Otherwise, load the network config and prepare for training
			fs::path network_config_path = fs::path{"configs"}/mode_str;
			if (network_config_flag) {
				auto network_config_str = get(network_config_flag);
				if ((network_config_path/network_config_str).exists()) {
					network_config_path = network_config_path/network_config_str;
				} else {
					network_config_path = network_config_str;
				}
			} else {
				network_config_path = network_config_path/"base.json";
			}

			if (!network_config_path.exists()) {
				tlog::error() << "Network config path " << network_config_path << " does not exist.";
				return 1;
			}

			testbed.reload_network_from_file(network_config_path.str());
			testbed.m_train = !no_train_flag;
		}

		bool gui = !no_gui_flag;
#ifndef NGP_GUI
		gui = false;
#endif

		if (gui) {
			testbed.init_window(width_flag ? get(width_flag) : 1920, height_flag ? get(height_flag) : 1080);
		}

		// Render/training loop
		while (testbed.frame()) {
			if (!gui) {
				tlog::info() << "iteration=" << testbed.m_training_step << " loss=" << testbed.m_loss_scalar.val();
			}
		}
	} catch (const exception& e) {
		tlog::error() << "Uncaught exception: " << e.what();
		return 1;
	}
}
