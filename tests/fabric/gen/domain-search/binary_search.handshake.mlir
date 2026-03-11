#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/binary_search/binary_search.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/binary_search/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/binary_search/binary_search.cpp":12:0)
#loc2 = loc(unknown)
#loc3 = loc("tests/app/binary_search/binary_search.cpp":18:0)
#loc11 = loc("tests/app/binary_search/binary_search.cpp":44:0)
#loc25 = loc("tests/app/binary_search/main.cpp":26:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 320, elements = #llvm.di_subrange<count = 10 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 160, elements = #llvm.di_subrange<count = 5 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__int32_t", baseType = #di_basic_type2>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 18>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 52>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 26>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type2>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type, sizeInBits = 64>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type1>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "int32_t", baseType = #di_derived_type2>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 18>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 52>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_sorted", file = #di_file1, line = 10, type = #di_composite_type>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_targets", file = #di_file1, line = 13, type = #di_composite_type1>
#di_composite_type2 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type4, sizeInBits = 160, elements = #llvm.di_subrange<count = 5 : i64>>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type4, sizeInBits = 64>
#di_derived_type8 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type4>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file, line = 18>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 52>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block, name = "t", file = #di_file, line = 18, type = #di_derived_type4>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "t", file = #di_file, line = 52, type = #di_derived_type4>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 26, type = #di_derived_type4>
#di_derived_type9 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type7>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 24>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 58>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_sorted", file = #di_file, line = 12, arg = 1, type = #di_derived_type6>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_targets", file = #di_file, line = 13, arg = 2, type = #di_derived_type6>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 15, arg = 4, type = #di_derived_type8>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "M", file = #di_file, line = 16, arg = 5, type = #di_derived_type8>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block5, name = "target", file = #di_file, line = 19, type = #di_basic_type>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block5, name = "left", file = #di_file, line = 20, type = #di_derived_type5>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block5, name = "right", file = #di_file, line = 21, type = #di_derived_type5>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block5, name = "result", file = #di_file, line = 22, type = #di_derived_type5>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_sorted", file = #di_file, line = 44, arg = 1, type = #di_derived_type6>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_targets", file = #di_file, line = 45, arg = 2, type = #di_derived_type6>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 47, arg = 4, type = #di_derived_type8>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram1, name = "M", file = #di_file, line = 48, arg = 5, type = #di_derived_type8>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "target", file = #di_file, line = 53, type = #di_basic_type>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "left", file = #di_file, line = 54, type = #di_derived_type5>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "right", file = #di_file, line = 55, type = #di_derived_type5>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "result", file = #di_file, line = 56, type = #di_derived_type5>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 6, type = #di_derived_type8>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram2, name = "M", file = #di_file1, line = 7, type = #di_derived_type8>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_indices", file = #di_file1, line = 16, type = #di_composite_type2>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_indices", file = #di_file1, line = 17, type = #di_composite_type2>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_indices", file = #di_file, line = 14, arg = 3, type = #di_derived_type9>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "mid", file = #di_file, line = 25, type = #di_derived_type5>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_indices", file = #di_file, line = 46, arg = 3, type = #di_derived_type9>
#di_local_variable28 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "mid", file = #di_file, line = 59, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable21, #di_local_variable22, #di_local_variable, #di_local_variable1, #di_local_variable23, #di_local_variable24, #di_local_variable4>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type9, #di_derived_type8, #di_derived_type8>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 26>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "binary_search_cpu", linkageName = "_Z17binary_search_cpuPKfS0_Pjjj", file = #di_file, line = 12, scopeLine = 16, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable5, #di_local_variable6, #di_local_variable25, #di_local_variable7, #di_local_variable8, #di_local_variable2, #di_local_variable9, #di_local_variable10, #di_local_variable11, #di_local_variable12, #di_local_variable26>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "binary_search_dsa", linkageName = "_Z17binary_search_dsaPKfS0_Pjjj", file = #di_file, line = 44, scopeLine = 48, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable13, #di_local_variable14, #di_local_variable27, #di_local_variable15, #di_local_variable16, #di_local_variable3, #di_local_variable17, #di_local_variable18, #di_local_variable19, #di_local_variable20, #di_local_variable28>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 18>
#loc38 = loc(fused<#di_subprogram4>[#loc1])
#loc40 = loc(fused<#di_subprogram5>[#loc11])
#loc42 = loc(fused<#di_lexical_block9>[#loc25])
#loc43 = loc(fused<#di_lexical_block10>[#loc3])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<42xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 98, 105, 110, 97, 114, 121, 95, 115, 101, 97, 114, 99, 104, 47, 98, 105, 110, 97, 114, 121, 95, 115, 101, 97, 114, 99, 104, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @__const.main.input_sorted : memref<10xf32> = dense<[1.000000e+00, 3.000000e+00, 5.000000e+00, 7.000000e+00, 9.000000e+00, 1.100000e+01, 1.300000e+01, 1.500000e+01, 1.700000e+01, 1.900000e+01]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @__const.main.input_targets : memref<5xf32> = dense<[7.000000e+00, 2.000000e+00, 1.500000e+01, 2.000000e+01, 1.000000e+00]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @str : memref<22xi8> = dense<[98, 105, 110, 97, 114, 121, 95, 115, 101, 97, 114, 99, 104, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<22xi8> = dense<[98, 105, 110, 97, 114, 121, 95, 115, 101, 97, 114, 99, 104, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z17binary_search_cpuPKfS0_Pjjj(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg3: i32 loc(fused<#di_subprogram4>[#loc1]), %arg4: i32 loc(fused<#di_subprogram4>[#loc1])) {
    %0 = ub.poison : i32 loc(#loc38)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %1 = arith.cmpi eq, %arg4, %c0_i32 : i32 loc(#loc46)
    scf.if %1 {
    } else {
      %2 = arith.addi %arg3, %c-1_i32 : i32 loc(#loc2)
      %3 = arith.extui %arg4 : i32 to i64 loc(#loc46)
      %4 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        %5 = arith.index_cast %arg5 : i64 to index loc(#loc49)
        %6 = memref.load %arg1[%5] : memref<?xf32, strided<[1], offset: ?>> loc(#loc49)
        %7:4 = scf.while (%arg6 = %c0_i32, %arg7 = %2, %arg8 = %c-1_i32) : (i32, i32, i32) -> (i32, i32, i32, i32) {
          %10 = arith.cmpi sgt, %arg6, %arg7 : i32 loc(#loc50)
          %11:5 = scf.if %10 -> (i32, i32, i32, i32, i32) {
            scf.yield %0, %0, %0, %arg8, %c0_i32 : i32, i32, i32, i32, i32 loc(#loc50)
          } else {
            %13 = arith.subi %arg7, %arg6 : i32 loc(#loc57)
            %14 = arith.shrui %13, %c1_i32 : i32 loc(#loc57)
            %15 = arith.addi %14, %arg6 : i32 loc(#loc57)
            %16 = arith.extui %15 : i32 to i64 loc(#loc61)
            %17 = arith.index_cast %16 : i64 to index loc(#loc61)
            %18 = memref.load %arg0[%17] : memref<?xf32, strided<[1], offset: ?>> loc(#loc61)
            %19 = arith.cmpf une, %18, %6 : f32 loc(#loc61)
            %20 = arith.select %19, %arg8, %15 : i32 loc(#loc61)
            %21 = arith.extui %19 : i1 to i32 loc(#loc2)
            %22:2 = scf.if %19 -> (i32, i32) {
              %23 = arith.cmpf olt, %18, %6 : f32 loc(#loc63)
              %24 = arith.addi %15, %c1_i32 : i32 loc(#loc63)
              %25 = arith.addi %15, %c-1_i32 : i32 loc(#loc63)
              %26 = arith.select %23, %24, %arg6 : i32 loc(#loc63)
              %27 = arith.select %23, %arg7, %25 : i32 loc(#loc63)
              scf.yield %26, %27 : i32, i32 loc(#loc2)
            } else {
              scf.yield %0, %0 : i32, i32 loc(#loc2)
            } loc(#loc2)
            scf.yield %22#0, %22#1, %20, %20, %21 : i32, i32, i32, i32, i32 loc(#loc50)
          } loc(#loc50)
          %12 = arith.trunci %11#4 : i32 to i1 loc(#loc2)
          scf.condition(%12) %11#0, %11#1, %11#2, %11#3 : i32, i32, i32, i32 loc(#loc2)
        } do {
        ^bb0(%arg6: i32 loc(unknown), %arg7: i32 loc(unknown), %arg8: i32 loc(unknown), %arg9: i32 loc(unknown)):
          scf.yield %arg6, %arg7, %arg8 : i32, i32, i32 loc(#loc2)
        } loc(#loc2)
        memref.store %7#3, %arg2[%5] : memref<?xi32, strided<[1], offset: ?>> loc(#loc51)
        %8 = arith.addi %arg5, %c1_i64 : i64 loc(#loc46)
        %9 = arith.cmpi ne, %8, %3 : i64 loc(#loc52)
        scf.condition(%9) %8 : i64 loc(#loc43)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block10>[#loc3])):
        scf.yield %arg5 : i64 loc(#loc43)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc43)
    } loc(#loc43)
    return loc(#loc39)
  } loc(#loc38)
  handshake.func @_Z17binary_search_dsaPKfS0_Pjjj_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc11]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc11]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc11]), %arg3: i32 loc(fused<#di_subprogram5>[#loc11]), %arg4: i32 loc(fused<#di_subprogram5>[#loc11]), %arg5: i1 loc(fused<#di_subprogram5>[#loc11]), ...) -> i1 attributes {argNames = ["input_sorted", "input_targets", "output_indices", "N", "M", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = ub.poison : i32 loc(#loc40)
    %1 = handshake.join %arg5 : i1 loc(#loc40)
    %2 = handshake.join %1 : none loc(#loc40)
    %3 = handshake.constant %2 {value = 1 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %2 {value = 0 : i32} : i32 loc(#loc2)
    %5 = handshake.constant %2 {value = -1 : i32} : i32 loc(#loc2)
    %6 = handshake.constant %2 {value = 0 : i64} : i64 loc(#loc2)
    %7 = arith.cmpi eq, %arg4, %4 : i32 loc(#loc47)
    %trueResult, %falseResult = handshake.cond_br %7, %2 : none loc(#loc44)
    %8 = arith.addi %arg3, %5 : i32 loc(#loc2)
    %9 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc44)
    %10 = arith.index_cast %6 : i64 to index loc(#loc44)
    %11 = arith.index_cast %arg4 : i32 to index loc(#loc44)
    %index, %willContinue = dataflow.stream %10, %9, %11 {loom.annotations = ["loom.loop.no_parallel", "loom.loop.no_unroll"], step_op = "+=", stop_cond = "!="} loc(#loc44)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc44)
    %12 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc44)
    %dataResult, %addressResults = handshake.load [%afterValue] %47#0, %50 : index, f32 loc(#loc53)
    %13 = dataflow.invariant %afterCond, %8 : i1, i32 -> i32 loc(#loc2)
    %14 = handshake.constant %2 {value = true} : i1 loc(#loc2)
    %15 = dataflow.carry %14, %4, %trueResult_4 : i1, i32, i32 -> i32 loc(#loc2)
    %16 = dataflow.carry %14, %13, %trueResult_6 : i1, i32, i32 -> i32 loc(#loc2)
    %17 = dataflow.carry %14, %5, %trueResult_8 : i1, i32, i32 -> i32 loc(#loc2)
    %18 = dataflow.invariant %14, %12 : i1, none -> none loc(#loc2)
    %19 = arith.cmpi sgt, %15, %16 : i32 loc(#loc54)
    %trueResult_0, %falseResult_1 = handshake.cond_br %19, %18 : none loc(#loc54)
    %20 = arith.subi %16, %15 : i32 loc(#loc58)
    %21 = arith.shrui %20, %3 : i32 loc(#loc58)
    %22 = arith.addi %21, %15 : i32 loc(#loc58)
    %23 = arith.extui %22 : i32 to i64 loc(#loc62)
    %24 = arith.index_cast %23 : i64 to index loc(#loc62)
    %dataResult_2, %addressResults_3 = handshake.load [%24] %48#0, %falseResult_18 : index, f32 loc(#loc62)
    %25 = arith.cmpf une, %dataResult_2, %dataResult : f32 loc(#loc62)
    %26 = arith.select %25, %17, %22 : i32 loc(#loc62)
    %27 = arith.extui %25 : i1 to i32 loc(#loc2)
    %28 = arith.cmpf olt, %dataResult_2, %dataResult : f32 loc(#loc64)
    %29 = arith.addi %22, %3 : i32 loc(#loc64)
    %30 = arith.addi %22, %5 : i32 loc(#loc64)
    %31 = arith.select %28, %29, %15 : i32 loc(#loc64)
    %32 = arith.select %28, %16, %30 : i32 loc(#loc64)
    %33 = handshake.constant %falseResult_1 {value = 0 : index} : index loc(#loc2)
    %34 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc2)
    %35 = arith.select %25, %34, %33 : index loc(#loc2)
    %36 = handshake.mux %35 [%0, %31] : index, i32 loc(#loc2)
    %37 = handshake.mux %35 [%0, %32] : index, i32 loc(#loc2)
    %38 = handshake.constant %18 {value = 0 : index} : index loc(#loc54)
    %39 = handshake.constant %18 {value = 1 : index} : index loc(#loc54)
    %40 = arith.select %19, %39, %38 : index loc(#loc54)
    %41 = handshake.mux %40 [%36, %0] : index, i32 loc(#loc54)
    %42 = handshake.mux %40 [%37, %0] : index, i32 loc(#loc54)
    %43 = handshake.mux %40 [%26, %0] : index, i32 loc(#loc54)
    %44 = handshake.mux %40 [%26, %17] : index, i32 loc(#loc54)
    %45 = handshake.mux %40 [%27, %4] : index, i32 loc(#loc54)
    %46 = arith.trunci %45 : i32 to i1 loc(#loc2)
    %trueResult_4, %falseResult_5 = handshake.cond_br %46, %41 : i32 loc(#loc2)
    handshake.sink %falseResult_5 : i32 loc(#loc2)
    %trueResult_6, %falseResult_7 = handshake.cond_br %46, %42 : i32 loc(#loc2)
    handshake.sink %falseResult_7 : i32 loc(#loc2)
    %trueResult_8, %falseResult_9 = handshake.cond_br %46, %43 : i32 loc(#loc2)
    handshake.sink %falseResult_9 : i32 loc(#loc2)
    %trueResult_10, %falseResult_11 = handshake.cond_br %46, %44 : i32 loc(#loc2)
    %dataResult_12, %addressResult = handshake.store [%afterValue] %falseResult_11, %55 : index, i32 loc(#loc55)
    %47:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc40)
    %48:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_3) {id = 1 : i32} : (index) -> (f32, none) loc(#loc40)
    %49 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_12, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc40)
    %50 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc44)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %47#1 : none loc(#loc44)
    %51 = handshake.constant %2 {value = 0 : index} : index loc(#loc44)
    %52 = handshake.constant %2 {value = 1 : index} : index loc(#loc44)
    %53 = arith.select %7, %52, %51 : index loc(#loc44)
    %54 = handshake.mux %53 [%falseResult_14, %trueResult] : index, none loc(#loc44)
    %55 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc44)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %49 : none loc(#loc44)
    %56 = handshake.mux %53 [%falseResult_16, %trueResult] : index, none loc(#loc44)
    %57 = dataflow.carry %willContinue, %falseResult, %trueResult_21 : i1, none, none -> none loc(#loc44)
    %58 = dataflow.carry %46, %57, %trueResult_19 : i1, none, none -> none loc(#loc2)
    %trueResult_17, %falseResult_18 = handshake.cond_br %19, %58 : none loc(#loc54)
    %59 = handshake.constant %58 {value = 0 : index} : index loc(#loc54)
    %60 = handshake.constant %58 {value = 1 : index} : index loc(#loc54)
    %61 = arith.select %19, %60, %59 : index loc(#loc54)
    %62 = handshake.mux %61 [%48#1, %trueResult_17] : index, none loc(#loc54)
    %trueResult_19, %falseResult_20 = handshake.cond_br %46, %62 : none loc(#loc2)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue, %falseResult_20 : none loc(#loc44)
    %63 = handshake.mux %53 [%falseResult_22, %trueResult] : index, none loc(#loc44)
    %64 = handshake.join %54, %56, %63 : none, none, none loc(#loc40)
    %65 = handshake.constant %64 {value = true} : i1 loc(#loc40)
    handshake.return %65 : i1 loc(#loc40)
  } loc(#loc40)
  handshake.func @_Z17binary_search_dsaPKfS0_Pjjj(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc11]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc11]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc11]), %arg3: i32 loc(fused<#di_subprogram5>[#loc11]), %arg4: i32 loc(fused<#di_subprogram5>[#loc11]), %arg5: none loc(fused<#di_subprogram5>[#loc11]), ...) -> none attributes {argNames = ["input_sorted", "input_targets", "output_indices", "N", "M", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = ub.poison : i32 loc(#loc40)
    %1 = handshake.join %arg5 : none loc(#loc40)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = -1 : i32} : i32 loc(#loc2)
    %5 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %6 = arith.cmpi eq, %arg4, %3 : i32 loc(#loc47)
    %trueResult, %falseResult = handshake.cond_br %6, %1 : none loc(#loc44)
    %7 = arith.addi %arg3, %4 : i32 loc(#loc2)
    %8 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc44)
    %9 = arith.index_cast %5 : i64 to index loc(#loc44)
    %10 = arith.index_cast %arg4 : i32 to index loc(#loc44)
    %index, %willContinue = dataflow.stream %9, %8, %10 {loom.annotations = ["loom.loop.no_parallel", "loom.loop.no_unroll"], step_op = "+=", stop_cond = "!="} loc(#loc44)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc44)
    %11 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc44)
    %dataResult, %addressResults = handshake.load [%afterValue] %46#0, %49 : index, f32 loc(#loc53)
    %12 = dataflow.invariant %afterCond, %7 : i1, i32 -> i32 loc(#loc2)
    %13 = handshake.constant %1 {value = true} : i1 loc(#loc2)
    %14 = dataflow.carry %13, %3, %trueResult_4 : i1, i32, i32 -> i32 loc(#loc2)
    %15 = dataflow.carry %13, %12, %trueResult_6 : i1, i32, i32 -> i32 loc(#loc2)
    %16 = dataflow.carry %13, %4, %trueResult_8 : i1, i32, i32 -> i32 loc(#loc2)
    %17 = dataflow.invariant %13, %11 : i1, none -> none loc(#loc2)
    %18 = arith.cmpi sgt, %14, %15 : i32 loc(#loc54)
    %trueResult_0, %falseResult_1 = handshake.cond_br %18, %17 : none loc(#loc54)
    %19 = arith.subi %15, %14 : i32 loc(#loc58)
    %20 = arith.shrui %19, %2 : i32 loc(#loc58)
    %21 = arith.addi %20, %14 : i32 loc(#loc58)
    %22 = arith.extui %21 : i32 to i64 loc(#loc62)
    %23 = arith.index_cast %22 : i64 to index loc(#loc62)
    %dataResult_2, %addressResults_3 = handshake.load [%23] %47#0, %falseResult_18 : index, f32 loc(#loc62)
    %24 = arith.cmpf une, %dataResult_2, %dataResult : f32 loc(#loc62)
    %25 = arith.select %24, %16, %21 : i32 loc(#loc62)
    %26 = arith.extui %24 : i1 to i32 loc(#loc2)
    %27 = arith.cmpf olt, %dataResult_2, %dataResult : f32 loc(#loc64)
    %28 = arith.addi %21, %2 : i32 loc(#loc64)
    %29 = arith.addi %21, %4 : i32 loc(#loc64)
    %30 = arith.select %27, %28, %14 : i32 loc(#loc64)
    %31 = arith.select %27, %15, %29 : i32 loc(#loc64)
    %32 = handshake.constant %falseResult_1 {value = 0 : index} : index loc(#loc2)
    %33 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc2)
    %34 = arith.select %24, %33, %32 : index loc(#loc2)
    %35 = handshake.mux %34 [%0, %30] : index, i32 loc(#loc2)
    %36 = handshake.mux %34 [%0, %31] : index, i32 loc(#loc2)
    %37 = handshake.constant %17 {value = 0 : index} : index loc(#loc54)
    %38 = handshake.constant %17 {value = 1 : index} : index loc(#loc54)
    %39 = arith.select %18, %38, %37 : index loc(#loc54)
    %40 = handshake.mux %39 [%35, %0] : index, i32 loc(#loc54)
    %41 = handshake.mux %39 [%36, %0] : index, i32 loc(#loc54)
    %42 = handshake.mux %39 [%25, %0] : index, i32 loc(#loc54)
    %43 = handshake.mux %39 [%25, %16] : index, i32 loc(#loc54)
    %44 = handshake.mux %39 [%26, %3] : index, i32 loc(#loc54)
    %45 = arith.trunci %44 : i32 to i1 loc(#loc2)
    %trueResult_4, %falseResult_5 = handshake.cond_br %45, %40 : i32 loc(#loc2)
    handshake.sink %falseResult_5 : i32 loc(#loc2)
    %trueResult_6, %falseResult_7 = handshake.cond_br %45, %41 : i32 loc(#loc2)
    handshake.sink %falseResult_7 : i32 loc(#loc2)
    %trueResult_8, %falseResult_9 = handshake.cond_br %45, %42 : i32 loc(#loc2)
    handshake.sink %falseResult_9 : i32 loc(#loc2)
    %trueResult_10, %falseResult_11 = handshake.cond_br %45, %43 : i32 loc(#loc2)
    %dataResult_12, %addressResult = handshake.store [%afterValue] %falseResult_11, %54 : index, i32 loc(#loc55)
    %46:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc40)
    %47:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_3) {id = 1 : i32} : (index) -> (f32, none) loc(#loc40)
    %48 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_12, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc40)
    %49 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc44)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %46#1 : none loc(#loc44)
    %50 = handshake.constant %1 {value = 0 : index} : index loc(#loc44)
    %51 = handshake.constant %1 {value = 1 : index} : index loc(#loc44)
    %52 = arith.select %6, %51, %50 : index loc(#loc44)
    %53 = handshake.mux %52 [%falseResult_14, %trueResult] : index, none loc(#loc44)
    %54 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc44)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %48 : none loc(#loc44)
    %55 = handshake.mux %52 [%falseResult_16, %trueResult] : index, none loc(#loc44)
    %56 = dataflow.carry %willContinue, %falseResult, %trueResult_21 : i1, none, none -> none loc(#loc44)
    %57 = dataflow.carry %45, %56, %trueResult_19 : i1, none, none -> none loc(#loc2)
    %trueResult_17, %falseResult_18 = handshake.cond_br %18, %57 : none loc(#loc54)
    %58 = handshake.constant %57 {value = 0 : index} : index loc(#loc54)
    %59 = handshake.constant %57 {value = 1 : index} : index loc(#loc54)
    %60 = arith.select %18, %59, %58 : index loc(#loc54)
    %61 = handshake.mux %60 [%47#1, %trueResult_17] : index, none loc(#loc54)
    %trueResult_19, %falseResult_20 = handshake.cond_br %45, %61 : none loc(#loc2)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue, %falseResult_20 : none loc(#loc44)
    %62 = handshake.mux %52 [%falseResult_22, %trueResult] : index, none loc(#loc44)
    %63 = handshake.join %53, %55, %62 : none, none, none loc(#loc40)
    handshake.return %63 : none loc(#loc41)
  } loc(#loc40)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc31)
    %false = arith.constant false loc(#loc31)
    %0 = seq.const_clock  low loc(#loc31)
    %c2_i32 = arith.constant 2 : i32 loc(#loc31)
    %1 = ub.poison : i64 loc(#loc31)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c5 = arith.constant 5 : index loc(#loc32)
    %c10 = arith.constant 10 : index loc(#loc33)
    %c1 = arith.constant 1 : index loc(#loc2)
    %c5_i64 = arith.constant 5 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c5_i32 = arith.constant 5 : i32 loc(#loc2)
    %c10_i32 = arith.constant 10 : i32 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %2 = memref.get_global @__const.main.input_sorted : memref<10xf32> loc(#loc2)
    %3 = memref.get_global @__const.main.input_targets : memref<5xf32> loc(#loc2)
    %4 = memref.get_global @str : memref<22xi8> loc(#loc2)
    %5 = memref.get_global @str.2 : memref<22xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<10xf32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<5xf32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<5xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<5xi32> loc(#loc2)
    scf.for %arg0 = %c0 to %c10 step %c1 {
      %11 = memref.load %2[%arg0] : memref<10xf32> loc(#loc33)
      memref.store %11, %alloca[%arg0] : memref<10xf32> loc(#loc33)
    } loc(#loc33)
    scf.for %arg0 = %c0 to %c5 step %c1 {
      %11 = memref.load %3[%arg0] : memref<5xf32> loc(#loc32)
      memref.store %11, %alloca_0[%arg0] : memref<5xf32> loc(#loc32)
    } loc(#loc32)
    %cast = memref.cast %alloca : memref<10xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc34)
    %cast_3 = memref.cast %alloca_0 : memref<5xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc34)
    %cast_4 = memref.cast %alloca_1 : memref<5xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc34)
    call @_Z17binary_search_cpuPKfS0_Pjjj(%cast, %cast_3, %cast_4, %c10_i32, %c5_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc34)
    %cast_5 = memref.cast %alloca_2 : memref<5xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc35)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc35)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc35)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc35)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c10_i32, %true : i32 loc(#loc35)
    %chanOutput_12, %ready_13 = esi.wrap.vr %c5_i32, %true : i32 loc(#loc35)
    %chanOutput_14, %ready_15 = esi.wrap.vr %true, %true : i1 loc(#loc35)
    %6 = handshake.esi_instance @_Z17binary_search_dsaPKfS0_Pjjj_esi "_Z17binary_search_dsaPKfS0_Pjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12, %chanOutput_14) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc35)
    %rawOutput, %valid = esi.unwrap.vr %6, %true : i1 loc(#loc35)
    %7:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc56)
      %12 = memref.load %alloca_1[%11] : memref<5xi32> loc(#loc56)
      %13 = memref.load %alloca_2[%11] : memref<5xi32> loc(#loc56)
      %14 = arith.cmpi eq, %12, %13 : i32 loc(#loc56)
      %15:3 = scf.if %14 -> (i64, i32, i32) {
        %17 = arith.addi %arg0, %c1_i64 : i64 loc(#loc45)
        %18 = arith.cmpi eq, %17, %c5_i64 : i64 loc(#loc45)
        %19 = arith.extui %18 : i1 to i32 loc(#loc42)
        %20 = arith.cmpi ne, %17, %c5_i64 : i64 loc(#loc48)
        %21 = arith.extui %20 : i1 to i32 loc(#loc42)
        scf.yield %17, %19, %21 : i64, i32, i32 loc(#loc56)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc56)
      } loc(#loc56)
      %16 = arith.trunci %15#2 : i32 to i1 loc(#loc42)
      scf.condition(%16) %15#0, %14, %15#1 : i64, i1, i32 loc(#loc42)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc25]), %arg1: i1 loc(fused<#di_lexical_block9>[#loc25]), %arg2: i32 loc(fused<#di_lexical_block9>[#loc25])):
      scf.yield %arg0 : i64 loc(#loc42)
    } loc(#loc42)
    %8 = arith.index_castui %7#2 : i32 to index loc(#loc42)
    %9 = scf.index_switch %8 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc42)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %4 : memref<22xi8> -> index loc(#loc59)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc59)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc59)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc59)
      scf.yield %c1_i32 : i32 loc(#loc60)
    } loc(#loc42)
    %10 = arith.select %7#1, %c0_i32, %9 : i32 loc(#loc2)
    scf.if %7#1 {
      %intptr = memref.extract_aligned_pointer_as_index %5 : memref<22xi8> -> index loc(#loc36)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc36)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc36)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc36)
    } loc(#loc2)
    return %10 : i32 loc(#loc37)
  } loc(#loc31)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.memcpy.p0.p0.i64(memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, i64, i1) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/binary_search/binary_search.cpp":0:0)
#loc4 = loc("tests/app/binary_search/binary_search.cpp":19:0)
#loc5 = loc("tests/app/binary_search/binary_search.cpp":24:0)
#loc6 = loc("tests/app/binary_search/binary_search.cpp":25:0)
#loc7 = loc("tests/app/binary_search/binary_search.cpp":27:0)
#loc8 = loc("tests/app/binary_search/binary_search.cpp":30:0)
#loc9 = loc("tests/app/binary_search/binary_search.cpp":38:0)
#loc10 = loc("tests/app/binary_search/binary_search.cpp":40:0)
#loc12 = loc("tests/app/binary_search/binary_search.cpp":52:0)
#loc13 = loc("tests/app/binary_search/binary_search.cpp":53:0)
#loc14 = loc("tests/app/binary_search/binary_search.cpp":58:0)
#loc15 = loc("tests/app/binary_search/binary_search.cpp":59:0)
#loc16 = loc("tests/app/binary_search/binary_search.cpp":61:0)
#loc17 = loc("tests/app/binary_search/binary_search.cpp":64:0)
#loc18 = loc("tests/app/binary_search/binary_search.cpp":72:0)
#loc19 = loc("tests/app/binary_search/binary_search.cpp":74:0)
#loc20 = loc("tests/app/binary_search/main.cpp":5:0)
#loc21 = loc("tests/app/binary_search/main.cpp":13:0)
#loc22 = loc("tests/app/binary_search/main.cpp":10:0)
#loc23 = loc("tests/app/binary_search/main.cpp":20:0)
#loc24 = loc("tests/app/binary_search/main.cpp":23:0)
#loc26 = loc("tests/app/binary_search/main.cpp":27:0)
#loc27 = loc("tests/app/binary_search/main.cpp":28:0)
#loc28 = loc("tests/app/binary_search/main.cpp":29:0)
#loc29 = loc("tests/app/binary_search/main.cpp":33:0)
#loc30 = loc("tests/app/binary_search/main.cpp":35:0)
#loc31 = loc(fused<#di_subprogram3>[#loc20])
#loc32 = loc(fused<#di_subprogram3>[#loc21])
#loc33 = loc(fused<#di_subprogram3>[#loc22])
#loc34 = loc(fused<#di_subprogram3>[#loc23])
#loc35 = loc(fused<#di_subprogram3>[#loc24])
#loc36 = loc(fused<#di_subprogram3>[#loc29])
#loc37 = loc(fused<#di_subprogram3>[#loc30])
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 52>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file1, line = 26>
#loc39 = loc(fused<#di_subprogram4>[#loc10])
#loc41 = loc(fused<#di_subprogram5>[#loc19])
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 18>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 52>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file1, line = 26>
#loc44 = loc(fused<#di_lexical_block11>[#loc12])
#loc45 = loc(fused<#di_lexical_block12>[#loc25])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 18>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 52>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 27>
#loc46 = loc(fused<#di_lexical_block13>[#loc3])
#loc47 = loc(fused<#di_lexical_block14>[#loc12])
#loc48 = loc(fused[#loc42, #loc45])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 24>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 58>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 27>
#loc49 = loc(fused<#di_lexical_block16>[#loc4])
#loc50 = loc(fused<#di_lexical_block16>[#loc5])
#loc51 = loc(fused<#di_lexical_block16>[#loc9])
#loc52 = loc(fused[#loc43, #loc46])
#loc53 = loc(fused<#di_lexical_block17>[#loc13])
#loc54 = loc(fused<#di_lexical_block17>[#loc14])
#loc55 = loc(fused<#di_lexical_block17>[#loc18])
#loc56 = loc(fused<#di_lexical_block18>[#loc26])
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 27>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 61>
#loc57 = loc(fused<#di_lexical_block19>[#loc6])
#loc58 = loc(fused<#di_lexical_block20>[#loc15])
#loc59 = loc(fused<#di_lexical_block21>[#loc27])
#loc60 = loc(fused<#di_lexical_block21>[#loc28])
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file, line = 30>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file, line = 64>
#loc61 = loc(fused<#di_lexical_block22>[#loc7])
#loc62 = loc(fused<#di_lexical_block23>[#loc16])
#loc63 = loc(fused<#di_lexical_block24>[#loc8])
#loc64 = loc(fused<#di_lexical_block25>[#loc17])
