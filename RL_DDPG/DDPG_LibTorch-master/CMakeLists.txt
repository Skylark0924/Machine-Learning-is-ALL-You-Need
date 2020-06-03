CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

get_filename_component(APP_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${APP_NAME})
set(CMAKE_CXX_STANDARD 17)
# Find Qt5 to be able to link it
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC  ON)
find_package(jsoncpp REQUIRED)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Wno-unknown-pragmas" )

find_package(${TORCH_DIR}/Torch REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic  )

target_sources(${PROJECT_NAME}
        PRIVATE
        agent.h
        agent.cpp
        ddpg_model.h
        ddpg_model.cpp
        replayBuffer.h
        )

target_link_libraries(${PROJECT_NAME}
        curl
        ${TORCH_LIBRARIES}
        )
