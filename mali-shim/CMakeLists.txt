cmake_minimum_required(VERSION 3.18)
project(mali_vk_shim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(mali_vk_shim SHARED src/shim.cpp)

target_compile_options(mali_vk_shim PRIVATE
    -Wall -Wextra -fvisibility=hidden -O2
    -DVK_USE_PLATFORM_ANDROID_KHR
    -DVK_NO_PROTOTYPES
)

target_link_libraries(mali_vk_shim PRIVATE log dl)

# Export only the HAL symbol (HMI = HAL_MODULE_INFO_SYM)
set_target_properties(mali_vk_shim PROPERTIES
    OUTPUT_NAME "vulkan.rk356x_shim"
)
