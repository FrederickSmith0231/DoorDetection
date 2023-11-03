cmake_minimum_required(VERSION 3.0.2)
project(doordetection)

# Find required ROS packages and libraries
find_package(catkin REQUIRED COMPONENTS
  std_msgs
  sensor_msgs
  detection_msgs
  roscpp
  pcl_conversions
  pcl_ros
  cv_bridge
)

find_package(OpenCV)

# Declare the catkin package
catkin_package(
  CATKIN_DEPENDS
    std_msgs
    sensor_msgs
    detection_msgs
    roscpp
)

# Specify the include directories for your package
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Declare a C++ executable
add_executable(doordetection_node src/doordetection_node.cpp)

# Link the executable with catkin libraries
target_link_libraries(doordetection_node
  ${catkin_LIBRARIES}
  ${OpenCV_INCLUDE_DIRS}
)

# Mark the executable for installation
install(
  TARGETS doordetection_node 
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
