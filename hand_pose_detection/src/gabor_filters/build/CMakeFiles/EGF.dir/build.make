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
include CMakeFiles/EGF.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/EGF.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/EGF.dir/flags.make

CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o: CMakeFiles/EGF.dir/flags.make
CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o: ../EGF/extract_gabor_filters.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o -c /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/EGF/extract_gabor_filters.cpp

CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/EGF/extract_gabor_filters.cpp > CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.i

CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/EGF/extract_gabor_filters.cpp -o CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.s

CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o.requires:
.PHONY : CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o.requires

CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o.provides: CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o.requires
	$(MAKE) -f CMakeFiles/EGF.dir/build.make CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o.provides.build
.PHONY : CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o.provides

CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o.provides.build: CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o

# Object files for target EGF
EGF_OBJECTS = \
"CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o"

# External object files for target EGF
EGF_EXTERNAL_OBJECTS =

libEGF.a: CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o
libEGF.a: CMakeFiles/EGF.dir/build.make
libEGF.a: CMakeFiles/EGF.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libEGF.a"
	$(CMAKE_COMMAND) -P CMakeFiles/EGF.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/EGF.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/EGF.dir/build: libEGF.a
.PHONY : CMakeFiles/EGF.dir/build

CMakeFiles/EGF.dir/requires: CMakeFiles/EGF.dir/EGF/extract_gabor_filters.cpp.o.requires
.PHONY : CMakeFiles/EGF.dir/requires

CMakeFiles/EGF.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/EGF.dir/cmake_clean.cmake
.PHONY : CMakeFiles/EGF.dir/clean

CMakeFiles/EGF.dir/depend:
	cd /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/build /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/build /home/asadi/GIT/Vision/hand_pose_detection/src/gabor_filters/build/CMakeFiles/EGF.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/EGF.dir/depend
