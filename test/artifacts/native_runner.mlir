// RUN: %python %S/test_native_runner.py %S/../..
// RUN: %python %S/../app/native_runner.py --all --build-root %t.dir --cc %loom-cc --cxx %loom-c++
