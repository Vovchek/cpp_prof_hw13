cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

cmake_host_system_information(RESULT HOST QUERY HOSTNAME)

set(PATCH_VERSION "1" CACHE INTERNAL "Patch version")
set(PROJECT_VERSION 0.0.${PATCH_VERSION})

project(fashio_mnist VERSION ${PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

file(GLOB HDRS
    include/*.h
)
file(GLOB SRCS
    src/*.cpp
)

if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "/usr/src/app")

    include(${PROJECT_SOURCE_DIR}/cmake/conan.cmake)
    conan_cmake_run(
        CONANFILE conanfile.txt
        BASIC_SETUP
        CMAKE_TARGETS # individual targets to link to
        BUILD
            missing
    )
endif()

set(COMMON_HDRS
    include/helpers.h
    include/mlp_classifier.h
)

set(COMMON_SRCS
    src/helpers.cpp
    src/mlp_classifier.cpp
)

add_executable(${PROJECT_NAME}
    ${SRCS}
    ${HDRS}
)

if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "/usr/src/app")

    target_link_libraries(${PROJECT_NAME} PUBLIC
        CONAN_PKG::eigen
    )
    target_include_directories(${PROJECT_NAME} PUBLIC
        include
    )
    target_link_libraries(${PROJECT_NAME}
        eigen
    )

else()
    target_include_directories(${PROJECT_NAME} PUBLIC
        include
    )
endif()


file(GLOB TEST_SRCS
    tests/*.cpp
)

add_executable(test_${PROJECT_NAME}
    tests/test_mlp.cpp
    ${COMMON_SRCS}
    ${COMMON_HDRS}
)

if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "/usr/src/app")

    target_include_directories(test_${PROJECT_NAME} PUBLIC
        include
    )
    target_link_libraries(test_${PROJECT_NAME}
        CONAN_PKG::igen
        CONAN_PKG::gtest
    )
else()
# ++setup google test localy
    include(FetchContent)   # run ..\CMake\share\cmake-X.YY\Modules\FetchContent.cmake
    if("${HOST}" STREQUAL "PCVS012CHEKAL")
        set(GTESTS_URL G:/softdev/googletest-main.zip)
    else()
        set(GTESTS_URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip)
    endif()
    FetchContent_Declare(
        googletest
        URL ${GTESTS_URL}
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
    target_link_libraries(test_${PROJECT_NAME} GTest::gtest_main)

    enable_testing()    # Enables testing for this directory and below
# --setup google test localy

    target_include_directories(test_${PROJECT_NAME} PUBLIC
        include
    )

endif()

add_custom_target(${PROJECT_NAME}_test
    $<TARGET_FILE:test_${PROJECT_NAME}>
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

set(CPACK_GENERATOR DEB)

set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

set(CPACK_PACKAGE_CONTACT vovchek64@inbox.ru)

include(CPack)
