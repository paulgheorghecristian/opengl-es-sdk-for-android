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

target_compile_definitions(common-native PUBLIC GLES_VERSION=2)
target_link_libraries(common-native lib_opencv log GLESv2 EGL)

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
target_link_libraries(common-native-gles3 lib_opencv log GLESv3 EGL)
