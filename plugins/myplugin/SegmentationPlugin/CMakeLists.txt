# CloudCompare example for standard plugins

# REPLACE ALL 'ExamplePlugin' OCCURENCES BY YOUR PLUGIN NAME
# AND ADAPT THE CODE BELOW TO YOUR OWN NEEDS!

# Add an option to CMake to control whether we build this plugin or not
option( PLUGIN_SEGMENTATION_STANDARD "Install example plugin" OFF )

if ( PLUGIN_SEGMENTATION_STANDARD )
	project( Segmentation )
	 
	AddPlugin( NAME ${PROJECT_NAME} )

	add_subdirectory( include )
	add_subdirectory( src )

	#add_subdirectory( PclUtils ) # PCL bridge library
	#add_subdirectory( PclIO ) # PCL I/O plugin (to load PCD files)
	find_package(PCL 1.10 REQUIRED)
    if (PCL_FOUND) 
        include_directories(${PCL_INCLUDE_DIRS})                #4
        link_directories(${PCL_LIBRARY_DIRS})                   #5
        add_definitions(${PCL_DEFINITIONS})                     #6
    else()
       	message(STATUS "Do not found pcl")
    endif()

	target_link_libraries( ${PROJECT_NAME}  ${PCL_LIBRARIES})
		
	# Import PCL dlls (if any, WIN32 only)
	# include( ../cmake/ExportPCLDlls.cmake )
	# export_PCL_dlls( "${CLOUDCOMPARE_DEST_FOLDER}" )

	# set dependencies to necessary libraries
	# target_link_libraries( ${PROJECT_NAME} LIB1 )
endif()
