cmake_minimum_required(VERSION 3.10)
project(disparity_calc CXX)

set(CXX_STANDARD 17)

find_package(OpenCV REQUIRED COMPONENTS ximgproc highgui)
find_package(Boost REQUIRED COMPONENTS filesystem)

add_executable(just_go main.cpp)
target_link_libraries(just_go ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
target_include_directories(just_go SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
