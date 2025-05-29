set(SOURCE_FILES
    build_intersection.cpp
    build_range.cpp
    search_intersection.cpp
    search_range.cpp
)

foreach(source_file ${SOURCE_FILES})
    get_filename_component(exec_name ${source_file} NAME_WE)
    add_executable(${exec_name} ${source_file})
    target_link_libraries(${exec_name} stdc++fs)
endforeach()
