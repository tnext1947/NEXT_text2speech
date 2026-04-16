cmake_minimum_required(VERSION 3.0.2)
project(next_text_ai)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  next_msgs
  actionlib_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/gtts_node.py
  scripts/vachanatts_node.py
  scripts/robot.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
