ifeq ($(ANDROID_HOME),)
NDK ?= $(ANDROID_NDK)
else
NDK ?= $(lastword $(sort $(wildcard $(dir $(ANDROID_HOME))/android-ndk*)))
endif

all:
	$(NDK)/ndk-build V=1
