#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "bool", sizeInBits = 8, encoding = DW_ATE_boolean>
#di_file = #llvm.di_file<"tests/app/axpy/axpy.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/axpy/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/axpy/axpy.cpp":8:0)
#loc3 = loc("tests/app/axpy/axpy.cpp":12:0)
#loc6 = loc("tests/app/axpy/axpy.cpp":19:0)
#loc17 = loc("tests/app/axpy/main.cpp":28:0)
#loc20 = loc("tests/app/axpy/main.cpp":31:0)
#loc23 = loc("tests/app/axpy/main.cpp":34:0)
#loc25 = loc("tests/app/axpy/main.cpp":39:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_label = #llvm.di_label<scope = #di_subprogram1, name = "compute_loop", file = #di_file, line = 23>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 12>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 26>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 28>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 31>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 34>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 39>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram2, name = "passed", file = #di_file1, line = 38, type = #di_basic_type2>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type2, sizeInBits = 256, elements = #llvm.di_subrange<count = 8 : i64>>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type2>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram, name = "alpha", file = #di_file, line = 11, arg = 4, type = #di_derived_type2>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 11, arg = 5, type = #di_derived_type2>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 12, type = #di_derived_type2>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram1, name = "alpha", file = #di_file, line = 22, arg = 4, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 22, arg = 5, type = #di_derived_type2>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 26, type = #di_derived_type2>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 28, type = #di_derived_type2>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 31, type = #di_derived_type2>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file1, line = 34, type = #di_derived_type2>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block5, name = "i", file = #di_file1, line = 39, type = #di_derived_type2>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type4, sizeInBits = 64>
#di_derived_type8 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type5>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 8, type = #di_derived_type4>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram2, name = "alpha", file = #di_file1, line = 9, type = #di_derived_type4>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram2, name = "x", file = #di_file1, line = 12, type = #di_composite_type>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram2, name = "y", file = #di_file1, line = 13, type = #di_composite_type>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "cpu_out", file = #di_file1, line = 16, type = #di_composite_type>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "dsa_out", file = #di_file1, line = 17, type = #di_composite_type>
#di_derived_type9 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type7>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_y", file = #di_file, line = 10, arg = 3, type = #di_derived_type8>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_y", file = #di_file, line = 21, arg = 3, type = #di_derived_type8>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 7, scopeLine = 7, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable11, #di_local_variable12, #di_local_variable13, #di_local_variable14, #di_local_variable15, #di_local_variable16, #di_local_variable7, #di_local_variable8, #di_local_variable9, #di_local_variable, #di_local_variable10>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 28>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 31>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 34>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 39>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_x", file = #di_file, line = 8, arg = 1, type = #di_derived_type9>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_y", file = #di_file, line = 9, arg = 2, type = #di_derived_type9>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_x", file = #di_file, line = 19, arg = 1, type = #di_derived_type9>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_y", file = #di_file, line = 20, arg = 2, type = #di_derived_type9>
#di_subroutine_type2 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type9, #di_derived_type9, #di_derived_type8, #di_derived_type2, #di_derived_type2>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "axpy_cpu", linkageName = "_Z8axpy_cpuPKjS0_Pjjj", file = #di_file, line = 8, scopeLine = 11, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable19, #di_local_variable20, #di_local_variable17, #di_local_variable1, #di_local_variable2, #di_local_variable3>
#di_subprogram6 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "axpy_dsa", linkageName = "_Z8axpy_dsaPKjS0_Pjjj", file = #di_file, line = 19, scopeLine = 22, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable21, #di_local_variable22, #di_local_variable18, #di_local_variable4, #di_local_variable5, #di_label, #di_local_variable6>
#loc46 = loc(fused<#di_lexical_block6>[#loc17])
#loc47 = loc(fused<#di_lexical_block7>[#loc20])
#loc48 = loc(fused<#di_lexical_block8>[#loc23])
#loc49 = loc(fused<#di_lexical_block9>[#loc25])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 12>
#loc51 = loc(fused<#di_subprogram5>[#loc1])
#loc53 = loc(fused<#di_subprogram6>[#loc6])
#loc60 = loc(fused<#di_lexical_block16>[#loc3])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<16xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 61, 97, 120, 112, 121, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<24xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 97, 120, 112, 121, 47, 97, 120, 112, 121, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @__const.main.x : memref<8xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @__const.main.y : memref<8xi32> = dense<[10, 20, 30, 40, 50, 60, 70, 80]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @".str.2" : memref<43xi8> = dense<[65, 88, 80, 89, 32, 82, 101, 115, 117, 108, 116, 115, 32, 40, 121, 32, 61, 32, 97, 108, 112, 104, 97, 42, 120, 32, 43, 32, 121, 44, 32, 97, 108, 112, 104, 97, 61, 37, 117, 41, 58, 10, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.1.3" : memref<8xi8> = dense<[120, 32, 32, 32, 61, 32, 91, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.2.4" : memref<5xi8> = dense<[37, 117, 37, 115, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.3" : memref<3xi8> = dense<[44, 32, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.4" : memref<1xi8> = dense<0> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.6" : memref<8xi8> = dense<[121, 32, 32, 32, 61, 32, 91, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.7" : memref<8xi8> = dense<[111, 117, 116, 32, 61, 32, 91, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.8" : memref<36xi8> = dense<[70, 65, 73, 76, 69, 68, 32, 97, 116, 32, 105, 110, 100, 101, 120, 32, 37, 117, 58, 32, 99, 112, 117, 61, 37, 117, 44, 32, 100, 115, 97, 61, 37, 117, 10, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.11 : memref<2xi8> = dense<[93, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.12 : memref<29xi8> = dense<[80, 65, 83, 83, 69, 68, 58, 32, 65, 108, 108, 32, 114, 101, 115, 117, 108, 116, 115, 32, 99, 111, 114, 114, 101, 99, 116, 33, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z8axpy_cpuPKjS0_Pjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg3: i32 loc(fused<#di_subprogram5>[#loc1]), %arg4: i32 loc(fused<#di_subprogram5>[#loc1])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg4, %c0_i32 : i32 loc(#loc66)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg4 : i32 to i64 loc(#loc66)
      %2 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg5 : i64 to index loc(#loc69)
        %4 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc69)
        %5 = arith.muli %4, %arg3 : i32 loc(#loc69)
        %6 = memref.load %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc69)
        %7 = arith.addi %5, %6 : i32 loc(#loc69)
        memref.store %7, %arg2[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc69)
        %8 = arith.addi %arg5, %c1_i64 : i64 loc(#loc66)
        %9 = arith.cmpi ne, %8, %1 : i64 loc(#loc70)
        scf.condition(%9) %8 : i64 loc(#loc60)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block16>[#loc3])):
        scf.yield %arg5 : i64 loc(#loc60)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc60)
    } loc(#loc60)
    return loc(#loc52)
  } loc(#loc51)
  handshake.func @_Z8axpy_dsaPKjS0_Pjjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc6]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc6]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc6]), %arg3: i32 loc(fused<#di_subprogram6>[#loc6]), %arg4: i32 loc(fused<#di_subprogram6>[#loc6]), %arg5: i1 loc(fused<#di_subprogram6>[#loc6]), ...) -> i1 attributes {argNames = ["input_x", "input_y", "output_y", "alpha", "N", "start_token"], loom.annotations = ["loom.accel=axpy"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : i1 loc(#loc53)
    %1 = handshake.join %0 : none loc(#loc53)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg4, %2 : i32 loc(#loc67)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc61)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc61)
    %6 = arith.index_cast %3 : i64 to index loc(#loc61)
    %7 = arith.index_cast %arg4 : i32 to index loc(#loc61)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.parallel degree=4 schedule=1", "loom.loop.tripcount typical=256 avg=256 min=1 max=1024"], step_op = "+=", stop_cond = "!="} loc(#loc61)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc61)
    %dataResult, %addressResults = handshake.load [%afterValue] %11#0, %13 : index, i32 loc(#loc71)
    %8 = arith.muli %dataResult, %arg3 : i32 loc(#loc71)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %10#0, %20 : index, i32 loc(#loc71)
    %9 = arith.addi %8, %dataResult_0 : i32 loc(#loc71)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %9, %18 : index, i32 loc(#loc71)
    %10:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 0 : i32} : (index) -> (i32, none) loc(#loc53)
    %11:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc53)
    %12 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc53)
    %13 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc61)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %11#1 : none loc(#loc61)
    %14 = handshake.constant %1 {value = 0 : index} : index loc(#loc61)
    %15 = handshake.constant %1 {value = 1 : index} : index loc(#loc61)
    %16 = arith.select %4, %15, %14 : index loc(#loc61)
    %17 = handshake.mux %16 [%falseResult_4, %trueResult] : index, none loc(#loc61)
    %18 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc61)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %12 : none loc(#loc61)
    %19 = handshake.mux %16 [%falseResult_6, %trueResult] : index, none loc(#loc61)
    %20 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc61)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %10#1 : none loc(#loc61)
    %21 = handshake.mux %16 [%falseResult_8, %trueResult] : index, none loc(#loc61)
    %22 = handshake.join %17, %19, %21 : none, none, none loc(#loc53)
    %23 = handshake.constant %22 {value = true} : i1 loc(#loc53)
    handshake.return %23 : i1 loc(#loc53)
  } loc(#loc53)
  handshake.func @_Z8axpy_dsaPKjS0_Pjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc6]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc6]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc6]), %arg3: i32 loc(fused<#di_subprogram6>[#loc6]), %arg4: i32 loc(fused<#di_subprogram6>[#loc6]), %arg5: none loc(fused<#di_subprogram6>[#loc6]), ...) -> none attributes {argNames = ["input_x", "input_y", "output_y", "alpha", "N", "start_token"], loom.annotations = ["loom.accel=axpy"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : none loc(#loc53)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi eq, %arg4, %1 : i32 loc(#loc67)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc61)
    %4 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc61)
    %5 = arith.index_cast %2 : i64 to index loc(#loc61)
    %6 = arith.index_cast %arg4 : i32 to index loc(#loc61)
    %index, %willContinue = dataflow.stream %5, %4, %6 {loom.annotations = ["loom.loop.parallel degree=4 schedule=1", "loom.loop.tripcount typical=256 avg=256 min=1 max=1024"], step_op = "+=", stop_cond = "!="} loc(#loc61)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc61)
    %dataResult, %addressResults = handshake.load [%afterValue] %10#0, %12 : index, i32 loc(#loc71)
    %7 = arith.muli %dataResult, %arg3 : i32 loc(#loc71)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %9#0, %19 : index, i32 loc(#loc71)
    %8 = arith.addi %7, %dataResult_0 : i32 loc(#loc71)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %8, %17 : index, i32 loc(#loc71)
    %9:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 0 : i32} : (index) -> (i32, none) loc(#loc53)
    %10:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc53)
    %11 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc53)
    %12 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc61)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %10#1 : none loc(#loc61)
    %13 = handshake.constant %0 {value = 0 : index} : index loc(#loc61)
    %14 = handshake.constant %0 {value = 1 : index} : index loc(#loc61)
    %15 = arith.select %3, %14, %13 : index loc(#loc61)
    %16 = handshake.mux %15 [%falseResult_4, %trueResult] : index, none loc(#loc61)
    %17 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc61)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %11 : none loc(#loc61)
    %18 = handshake.mux %15 [%falseResult_6, %trueResult] : index, none loc(#loc61)
    %19 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc61)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %9#1 : none loc(#loc61)
    %20 = handshake.mux %15 [%falseResult_8, %trueResult] : index, none loc(#loc61)
    %21 = handshake.join %16, %18, %20 : none, none, none loc(#loc53)
    handshake.return %21 : none loc(#loc54)
  } loc(#loc53)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc2)
    %false = arith.constant false loc(#loc32)
    %0 = seq.const_clock  low loc(#loc32)
    %c8 = arith.constant 8 : index loc(#loc2)
    %c1 = arith.constant 1 : index loc(#loc2)
    %c0_i8 = arith.constant 0 : i8 loc(#loc2)
    %c1_i8 = arith.constant 1 : i8 loc(#loc2)
    %c8_i64 = arith.constant 8 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c7_i64 = arith.constant 7 : i64 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c8_i32 = arith.constant 8 : i32 loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %1 = memref.get_global @__const.main.x : memref<8xi32> loc(#loc2)
    %2 = memref.get_global @__const.main.y : memref<8xi32> loc(#loc2)
    %3 = memref.get_global @".str.2" : memref<43xi8> loc(#loc2)
    %4 = memref.get_global @".str.1.3" : memref<8xi8> loc(#loc2)
    %5 = memref.get_global @".str.4" : memref<1xi8> loc(#loc2)
    %6 = memref.get_global @".str.3" : memref<3xi8> loc(#loc2)
    %7 = memref.get_global @".str.2.4" : memref<5xi8> loc(#loc2)
    %8 = memref.get_global @str.11 : memref<2xi8> loc(#loc2)
    %9 = memref.get_global @".str.6" : memref<8xi8> loc(#loc2)
    %10 = memref.get_global @".str.7" : memref<8xi8> loc(#loc2)
    %11 = memref.get_global @".str.8" : memref<36xi8> loc(#loc2)
    %12 = memref.get_global @str.12 : memref<29xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<8xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<8xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<8xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<8xi32> loc(#loc2)
    scf.for %arg0 = %c0 to %c8 step %c1 {
      %40 = memref.load %1[%arg0] : memref<8xi32> loc(#loc33)
      memref.store %40, %alloca[%arg0] : memref<8xi32> loc(#loc33)
    } loc(#loc33)
    scf.for %arg0 = %c0 to %c8 step %c1 {
      %40 = memref.load %2[%arg0] : memref<8xi32> loc(#loc34)
      memref.store %40, %alloca_0[%arg0] : memref<8xi32> loc(#loc34)
    } loc(#loc34)
    %cast = memref.cast %alloca : memref<8xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc35)
    %cast_3 = memref.cast %alloca_0 : memref<8xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc35)
    %cast_4 = memref.cast %alloca_1 : memref<8xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc35)
    call @_Z8axpy_cpuPKjS0_Pjjj(%cast, %cast_3, %cast_4, %c3_i32, %c8_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc35)
    %cast_5 = memref.cast %alloca_2 : memref<8xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc36)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc36)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc36)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc36)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c3_i32, %true : i32 loc(#loc36)
    %chanOutput_12, %ready_13 = esi.wrap.vr %c8_i32, %true : i32 loc(#loc36)
    %chanOutput_14, %ready_15 = esi.wrap.vr %true, %true : i1 loc(#loc36)
    %13 = handshake.esi_instance @_Z8axpy_dsaPKjS0_Pjjj_esi "_Z8axpy_dsaPKjS0_Pjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12, %chanOutput_14) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc36)
    %rawOutput, %valid = esi.unwrap.vr %13, %true : i1 loc(#loc36)
    %intptr = memref.extract_aligned_pointer_as_index %3 : memref<43xi8> -> index loc(#loc37)
    %14 = arith.index_cast %intptr : index to i64 loc(#loc37)
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr loc(#loc37)
    %16 = llvm.call @printf(%15, %c3_i32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32 loc(#loc37)
    %intptr_16 = memref.extract_aligned_pointer_as_index %4 : memref<8xi8> -> index loc(#loc38)
    %17 = arith.index_cast %intptr_16 : index to i64 loc(#loc38)
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr loc(#loc38)
    %19 = llvm.call @printf(%18) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32 loc(#loc38)
    %cast_17 = memref.cast %5 : memref<1xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc55)
    %cast_18 = memref.cast %6 : memref<3xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc55)
    %intptr_19 = memref.extract_aligned_pointer_as_index %7 : memref<5xi8> -> index loc(#loc55)
    %20 = arith.index_cast %intptr_19 : index to i64 loc(#loc55)
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr loc(#loc55)
    %22 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %40 = arith.index_cast %arg0 : i64 to index loc(#loc55)
      %41 = memref.load %alloca[%40] : memref<8xi32> loc(#loc55)
      %42 = arith.cmpi eq, %arg0, %c7_i64 : i64 loc(#loc55)
      %43 = arith.select %42, %cast_17, %cast_18 : memref<?xi8, strided<[1], offset: ?>> loc(#loc55)
      %intptr_23 = memref.extract_aligned_pointer_as_index %43 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc55)
      %44 = arith.index_cast %intptr_23 : index to i64 loc(#loc55)
      %45 = llvm.inttoptr %44 : i64 to !llvm.ptr loc(#loc55)
      %46 = llvm.call @printf(%21, %41, %45) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr) -> i32 loc(#loc55)
      %47 = arith.addi %arg0, %c1_i64 : i64 loc(#loc55)
      %48 = arith.cmpi ne, %47, %c8_i64 : i64 loc(#loc62)
      scf.condition(%48) %47 : i64 loc(#loc46)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block6>[#loc17])):
      scf.yield %arg0 : i64 loc(#loc46)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc46)
    %intptr_20 = memref.extract_aligned_pointer_as_index %8 : memref<2xi8> -> index loc(#loc39)
    %23 = arith.index_cast %intptr_20 : index to i64 loc(#loc39)
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr loc(#loc39)
    %25 = llvm.call @puts(%24) : (!llvm.ptr) -> i32 loc(#loc39)
    %intptr_21 = memref.extract_aligned_pointer_as_index %9 : memref<8xi8> -> index loc(#loc40)
    %26 = arith.index_cast %intptr_21 : index to i64 loc(#loc40)
    %27 = llvm.inttoptr %26 : i64 to !llvm.ptr loc(#loc40)
    %28 = llvm.call @printf(%27) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32 loc(#loc40)
    %29 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %40 = arith.index_cast %arg0 : i64 to index loc(#loc56)
      %41 = memref.load %alloca_0[%40] : memref<8xi32> loc(#loc56)
      %42 = arith.cmpi eq, %arg0, %c7_i64 : i64 loc(#loc56)
      %43 = arith.select %42, %cast_17, %cast_18 : memref<?xi8, strided<[1], offset: ?>> loc(#loc56)
      %intptr_23 = memref.extract_aligned_pointer_as_index %43 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc56)
      %44 = arith.index_cast %intptr_23 : index to i64 loc(#loc56)
      %45 = llvm.inttoptr %44 : i64 to !llvm.ptr loc(#loc56)
      %46 = llvm.call @printf(%21, %41, %45) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr) -> i32 loc(#loc56)
      %47 = arith.addi %arg0, %c1_i64 : i64 loc(#loc56)
      %48 = arith.cmpi ne, %47, %c8_i64 : i64 loc(#loc63)
      scf.condition(%48) %47 : i64 loc(#loc47)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block7>[#loc20])):
      scf.yield %arg0 : i64 loc(#loc47)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc47)
    %30 = llvm.call @puts(%24) : (!llvm.ptr) -> i32 loc(#loc41)
    %intptr_22 = memref.extract_aligned_pointer_as_index %10 : memref<8xi8> -> index loc(#loc42)
    %31 = arith.index_cast %intptr_22 : index to i64 loc(#loc42)
    %32 = llvm.inttoptr %31 : i64 to !llvm.ptr loc(#loc42)
    %33 = llvm.call @printf(%32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32 loc(#loc42)
    %34 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %40 = arith.index_cast %arg0 : i64 to index loc(#loc57)
      %41 = memref.load %alloca_2[%40] : memref<8xi32> loc(#loc57)
      %42 = arith.cmpi eq, %arg0, %c7_i64 : i64 loc(#loc57)
      %43 = arith.select %42, %cast_17, %cast_18 : memref<?xi8, strided<[1], offset: ?>> loc(#loc57)
      %intptr_23 = memref.extract_aligned_pointer_as_index %43 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc57)
      %44 = arith.index_cast %intptr_23 : index to i64 loc(#loc57)
      %45 = llvm.inttoptr %44 : i64 to !llvm.ptr loc(#loc57)
      %46 = llvm.call @printf(%21, %41, %45) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr) -> i32 loc(#loc57)
      %47 = arith.addi %arg0, %c1_i64 : i64 loc(#loc57)
      %48 = arith.cmpi ne, %47, %c8_i64 : i64 loc(#loc64)
      scf.condition(%48) %47 : i64 loc(#loc48)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block8>[#loc23])):
      scf.yield %arg0 : i64 loc(#loc48)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc48)
    %35 = llvm.call @puts(%24) : (!llvm.ptr) -> i32 loc(#loc43)
    %36:2 = scf.while (%arg0 = %c0_i64, %arg1 = %c1_i8) : (i64, i8) -> (i64, i8) {
      %40 = arith.index_cast %arg0 : i64 to index loc(#loc68)
      %41 = memref.load %alloca_1[%40] : memref<8xi32> loc(#loc68)
      %42 = memref.load %alloca_2[%40] : memref<8xi32> loc(#loc68)
      %43 = arith.cmpi eq, %41, %42 : i32 loc(#loc68)
      %44 = arith.select %43, %arg1, %c0_i8 : i8 loc(#loc68)
      scf.if %43 {
      } else {
        %47 = arith.trunci %arg0 : i64 to i32 loc(#loc72)
        %intptr_23 = memref.extract_aligned_pointer_as_index %11 : memref<36xi8> -> index loc(#loc72)
        %48 = arith.index_cast %intptr_23 : index to i64 loc(#loc72)
        %49 = llvm.inttoptr %48 : i64 to !llvm.ptr loc(#loc72)
        %50 = llvm.call @printf(%49, %47, %41, %42) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, i32) -> i32 loc(#loc72)
      } loc(#loc68)
      %45 = arith.addi %arg0, %c1_i64 : i64 loc(#loc58)
      %46 = arith.cmpi ne, %45, %c8_i64 : i64 loc(#loc65)
      scf.condition(%46) %45, %44 : i64, i8 loc(#loc49)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc25]), %arg1: i8 loc(fused<#di_lexical_block9>[#loc25])):
      scf.yield %arg0, %arg1 : i64, i8 loc(#loc49)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc49)
    %37 = arith.trunci %36#1 : i8 to i1 loc(#loc50)
    scf.if %37 {
      %intptr_23 = memref.extract_aligned_pointer_as_index %12 : memref<29xi8> -> index loc(#loc59)
      %40 = arith.index_cast %intptr_23 : index to i64 loc(#loc59)
      %41 = llvm.inttoptr %40 : i64 to !llvm.ptr loc(#loc59)
      %42 = llvm.call @puts(%41) : (!llvm.ptr) -> i32 loc(#loc59)
    } loc(#loc50)
    %38 = arith.xori %37, %true : i1 loc(#loc44)
    %39 = arith.extui %38 : i1 to i32 loc(#loc44)
    return %39 : i32 loc(#loc44)
  } loc(#loc32)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.memcpy.p0.p0.i64(memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, i64, i1) loc(#loc2)
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} loc(#loc45)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#di_basic_type3 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "char", sizeInBits = 8, encoding = DW_ATE_signed_char>
#di_file2 = #llvm.di_file<"/usr/include/stdio.h" in "">
#loc = loc("tests/app/axpy/axpy.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/axpy/axpy.cpp":13:0)
#loc5 = loc("tests/app/axpy/axpy.cpp":15:0)
#loc7 = loc("tests/app/axpy/axpy.cpp":26:0)
#loc8 = loc("tests/app/axpy/axpy.cpp":27:0)
#loc9 = loc("tests/app/axpy/axpy.cpp":29:0)
#loc10 = loc("tests/app/axpy/main.cpp":7:0)
#loc11 = loc("tests/app/axpy/main.cpp":12:0)
#loc12 = loc("tests/app/axpy/main.cpp":13:0)
#loc13 = loc("tests/app/axpy/main.cpp":20:0)
#loc14 = loc("tests/app/axpy/main.cpp":23:0)
#loc15 = loc("tests/app/axpy/main.cpp":26:0)
#loc16 = loc("tests/app/axpy/main.cpp":27:0)
#loc18 = loc("tests/app/axpy/main.cpp":29:0)
#loc19 = loc("tests/app/axpy/main.cpp":30:0)
#loc21 = loc("tests/app/axpy/main.cpp":32:0)
#loc22 = loc("tests/app/axpy/main.cpp":33:0)
#loc24 = loc("tests/app/axpy/main.cpp":35:0)
#loc26 = loc("tests/app/axpy/main.cpp":40:0)
#loc27 = loc("tests/app/axpy/main.cpp":41:0)
#loc28 = loc("tests/app/axpy/main.cpp":46:0)
#loc29 = loc("tests/app/axpy/main.cpp":47:0)
#loc30 = loc("tests/app/axpy/main.cpp":50:0)
#loc31 = loc("/usr/include/stdio.h":363:0)
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type3>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_basic_type1, #di_derived_type6, #di_null_type>
#di_subprogram4 = #llvm.di_subprogram<scope = #di_file2, name = "printf", file = #di_file2, line = 363, subprogramFlags = Optimized, type = #di_subroutine_type1>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 46>
#loc32 = loc(fused<#di_subprogram3>[#loc10])
#loc33 = loc(fused<#di_subprogram3>[#loc11])
#loc34 = loc(fused<#di_subprogram3>[#loc12])
#loc35 = loc(fused<#di_subprogram3>[#loc13])
#loc36 = loc(fused<#di_subprogram3>[#loc14])
#loc37 = loc(fused<#di_subprogram3>[#loc15])
#loc38 = loc(fused<#di_subprogram3>[#loc16])
#loc39 = loc(fused<#di_subprogram3>[#loc18])
#loc40 = loc(fused<#di_subprogram3>[#loc19])
#loc41 = loc(fused<#di_subprogram3>[#loc21])
#loc42 = loc(fused<#di_subprogram3>[#loc22])
#loc43 = loc(fused<#di_subprogram3>[#loc24])
#loc44 = loc(fused<#di_subprogram3>[#loc30])
#loc45 = loc(fused<#di_subprogram4>[#loc31])
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 28>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 31>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 34>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file1, line = 39>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 46>
#loc50 = loc(fused<#di_lexical_block10>[#loc28])
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_subprogram6, file = #di_file, line = 26>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 39>
#loc52 = loc(fused<#di_subprogram5>[#loc5])
#loc54 = loc(fused<#di_subprogram6>[#loc9])
#loc55 = loc(fused<#di_lexical_block11>[#loc17])
#loc56 = loc(fused<#di_lexical_block12>[#loc20])
#loc57 = loc(fused<#di_lexical_block13>[#loc23])
#loc58 = loc(fused<#di_lexical_block14>[#loc25])
#loc59 = loc(fused<#di_lexical_block15>[#loc29])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 12>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 26>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 40>
#loc61 = loc(fused<#di_lexical_block17>[#loc7])
#loc62 = loc(fused[#loc46, #loc55])
#loc63 = loc(fused[#loc47, #loc56])
#loc64 = loc(fused[#loc48, #loc57])
#loc65 = loc(fused[#loc49, #loc58])
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 12>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 26>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file1, line = 40>
#loc66 = loc(fused<#di_lexical_block19>[#loc3])
#loc67 = loc(fused<#di_lexical_block20>[#loc7])
#loc68 = loc(fused<#di_lexical_block21>[#loc26])
#loc69 = loc(fused<#di_lexical_block22>[#loc4])
#loc70 = loc(fused[#loc60, #loc66])
#loc71 = loc(fused<#di_lexical_block23>[#loc8])
#loc72 = loc(fused<#di_lexical_block24>[#loc27])
