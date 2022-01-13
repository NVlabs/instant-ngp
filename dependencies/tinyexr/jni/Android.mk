# A simple test for the minimal standard C++ library
#

LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := tinyexr
LOCAL_SRC_FILES := ../tinyexr.cc

LOCAL_C_INCLUDES := ../

include $(BUILD_SHARED_LIBRARY)
