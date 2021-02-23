add_library(common-native STATIC
	src/Shader.cpp
	src/Text.cpp
	src/Texture.cpp
	src/ETCHeader.cpp
	src/Matrix.cpp
	src/JavaClass.cpp
	src/AndroidPlatform.cpp
	src/Timer.cpp)

target_include_directories(common-native PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/inc
	${CMAKE_CURRENT_SOURCE_DIR}/inc/mali)

add_library( lib_opencv SHARED IMPORTED )
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libopencv_java3.so)

add_library(detectron SHARED IMPORTED)
set_target_properties(detectron PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libcaffe2_detectron_ops.so)

add_library(clog STATIC IMPORTED)
set_target_properties(clog PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libclog.a)

add_library(cpuinfo STATIC IMPORTED)
set_target_properties(cpuinfo PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libcpuinfo.a)

add_library(c10 STATIC IMPORTED)
set_target_properties(c10 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libc10.a)

#add_library(torch STATIC IMPORTED)
#set_target_properties(torch PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libtorch.a)

add_library(caffe2 STATIC IMPORTED)
set_target_properties(caffe2 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libcaffe2_protos.a)

add_library(thread_pool STATIC IMPORTED)
set_target_properties(thread_pool PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libpthreadpool.a)

add_library(protobuf STATIC IMPORTED)
set_target_properties(protobuf PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libprotobuf.a)

add_library(NNPACK STATIC IMPORTED)
set_target_properties(NNPACK PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libnnpack.a)

#add_library(QNNPACK STATIC IMPORTED)
#set_target_properties(QNNPACK PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libqnnpack.a)

target_compile_definitions(common-native PUBLIC GLES_VERSION=2)
target_link_libraries(common-native
                        lib_opencv
                        -Wl,--whole-archive
                        caffe2
                        detectron
                        -Wl,--no-whole-archive
                        NNPACK
                        #QNNPACK
                        #torch
                        thread_pool
                        c10
                        protobuf
                        cpuinfo
                        clog
                        log
                        GLESv2
                        EGL)

add_library(common-native-gles3 STATIC
	src/Shader.cpp
	src/Text.cpp
	src/Texture.cpp
	src/ETCHeader.cpp
	src/Matrix.cpp
	src/JavaClass.cpp
	src/AndroidPlatform.cpp
	src/Timer.cpp)

target_include_directories(common-native-gles3 PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/inc
	${CMAKE_CURRENT_SOURCE_DIR}/inc/mali)

target_compile_definitions(common-native-gles3 PUBLIC GLES_VERSION=3)
target_link_libraries(common-native-gles3
                        lib_opencv
                        -Wl,--whole-archive
                        caffe2
                        detectron
                        -Wl,--no-whole-archive
                        NNPACK
                        QNNPACK
                        torch
                        thread_pool
                        c10
                        protobuf
                        cpuinfo
                        clog
                        log
                        GLESv3
                        EGL)
