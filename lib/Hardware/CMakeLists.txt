add_subdirectory(Configuration)
if(LOOM_HAVE_CIRCT)
  add_subdirectory(RTL)
endif()
