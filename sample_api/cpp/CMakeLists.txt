cmake_minimum_required(VERSION 3.7)
project("SAMPLE API")

find_package(OpenCV COMPONENTS core imgcodecs dnn REQUIRED)
find_package(cpprestsdk REQUIRED)
find_package(OpenSSL REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${cpprestsdk_INCLUDE_DIRS} ${OpenSSL_INCLUDE_DIRS})

file(GLOB API_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/source/api.cpp)

add_executable(api ${API_SOURCE})
target_link_libraries(api ${OpenCV_LIBRARIES} cpprestsdk::cpprest OpenSSL::SSL OpenSSL::Crypto)


