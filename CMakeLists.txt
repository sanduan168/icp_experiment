cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(pcl-interactive_icp)

find_package(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (fpfh FPFH.cpp)
add_executable (ndt ndt.cpp)
add_executable (bruteforce_icp bruteforce_icp.cpp)
add_executable (inter_icp interactive_icp.cpp)
target_link_libraries (fpfh ${PCL_LIBRARIES})
target_link_libraries (ndt ${PCL_LIBRARIES})
target_link_libraries (bruteforce_icp ${PCL_LIBRARIES})
target_link_libraries (inter_icp ${PCL_LIBRARIES})
