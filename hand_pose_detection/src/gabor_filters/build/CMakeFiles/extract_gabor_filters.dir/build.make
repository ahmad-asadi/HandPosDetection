# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/build

# Include any dependencies generated for this target.
include CMakeFiles/extract_gabor_filters.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/extract_gabor_filters.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/extract_gabor_filters.dir/flags.make

CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o: CMakeFiles/extract_gabor_filters.dir/flags.make
CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o: ../EGF/extract_gabor_filters.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o -c /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/EGF/extract_gabor_filters.cpp

CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/EGF/extract_gabor_filters.cpp > CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.i

CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/EGF/extract_gabor_filters.cpp -o CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.s

CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o.requires:
.PHONY : CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o.requires

CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o.provides: CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o.requires
	$(MAKE) -f CMakeFiles/extract_gabor_filters.dir/build.make CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o.provides.build
.PHONY : CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o.provides

CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o.provides.build: CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o

# Object files for target extract_gabor_filters
extract_gabor_filters_OBJECTS = \
"CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o"

# External object files for target extract_gabor_filters
extract_gabor_filters_EXTERNAL_OBJECTS =

libextract_gabor_filters.a: CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o
libextract_gabor_filters.a: CMakeFiles/extract_gabor_filters.dir/build.make
libextract_gabor_filters.a: CMakeFiles/extract_gabor_filters.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libextract_gabor_filters.a"
	$(CMAKE_COMMAND) -P CMakeFiles/extract_gabor_filters.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/extract_gabor_filters.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/extract_gabor_filters.dir/build: libextract_gabor_filters.a
.PHONY : CMakeFiles/extract_gabor_filters.dir/build

CMakeFiles/extract_gabor_filters.dir/requires: CMakeFiles/extract_gabor_filters.dir/EGF/extract_gabor_filters.cpp.o.requires
.PHONY : CMakeFiles/extract_gabor_filters.dir/requires

CMakeFiles/extract_gabor_filters.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/extract_gabor_filters.dir/cmake_clean.cmake
.PHONY : CMakeFiles/extract_gabor_filters.dir/clean

CMakeFiles/extract_gabor_filters.dir/depend:
	cd /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/build /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/build /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/build/CMakeFiles/extract_gabor_filters.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/extract_gabor_filters.dir/depend
