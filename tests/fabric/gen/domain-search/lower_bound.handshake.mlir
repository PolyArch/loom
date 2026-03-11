#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/lower_bound/lower_bound.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/lower_bound/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/lower_bound/lower_bound.cpp":12:0)
#loc3 = loc("tests/app/lower_bound/lower_bound.cpp":18:0)
#loc5 = loc("tests/app/lower_bound/lower_bound.cpp":23:0)
#loc10 = loc("tests/app/lower_bound/lower_bound.cpp":39:0)
#loc23 = loc("tests/app/lower_bound/main.cpp":26:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 320, elements = #llvm.di_subrange<count = 10 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 192, elements = #llvm.di_subrange<count = 6 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 18>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 47>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 26>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type2>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type, sizeInBits = 64>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type1>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 18>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 47>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_sorted", file = #di_file1, line = 10, type = #di_composite_type>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_targets", file = #di_file1, line = 13, type = #di_composite_type1>
#di_composite_type2 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type3, sizeInBits = 192, elements = #llvm.di_subrange<count = 6 : i64>>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type2>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type3, sizeInBits = 64>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type3>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file, line = 18>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 47>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block, name = "t", file = #di_file, line = 18, type = #di_derived_type3>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "t", file = #di_file, line = 47, type = #di_derived_type3>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 26, type = #di_derived_type3>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type5>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 23>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 52>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_sorted", file = #di_file, line = 12, arg = 1, type = #di_derived_type4>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_targets", file = #di_file, line = 13, arg = 2, type = #di_derived_type4>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 15, arg = 4, type = #di_derived_type6>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "M", file = #di_file, line = 16, arg = 5, type = #di_derived_type6>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block5, name = "target", file = #di_file, line = 19, type = #di_basic_type>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block5, name = "left", file = #di_file, line = 20, type = #di_derived_type3>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block5, name = "right", file = #di_file, line = 21, type = #di_derived_type3>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_sorted", file = #di_file, line = 39, arg = 1, type = #di_derived_type4>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_targets", file = #di_file, line = 40, arg = 2, type = #di_derived_type4>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 42, arg = 4, type = #di_derived_type6>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram1, name = "M", file = #di_file, line = 43, arg = 5, type = #di_derived_type6>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "target", file = #di_file, line = 48, type = #di_basic_type>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "left", file = #di_file, line = 49, type = #di_derived_type3>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "right", file = #di_file, line = 50, type = #di_derived_type3>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 6, type = #di_derived_type6>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram2, name = "M", file = #di_file1, line = 7, type = #di_derived_type6>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_indices", file = #di_file1, line = 16, type = #di_composite_type2>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_indices", file = #di_file1, line = 17, type = #di_composite_type2>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_indices", file = #di_file, line = 14, arg = 3, type = #di_derived_type7>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "mid", file = #di_file, line = 24, type = #di_derived_type3>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_indices", file = #di_file, line = 41, arg = 3, type = #di_derived_type7>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "mid", file = #di_file, line = 53, type = #di_derived_type3>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable19, #di_local_variable20, #di_local_variable, #di_local_variable1, #di_local_variable21, #di_local_variable22, #di_local_variable4>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type4, #di_derived_type4, #di_derived_type7, #di_derived_type6, #di_derived_type6>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 26>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "lower_bound_cpu", linkageName = "_Z15lower_bound_cpuPKfS0_Pjjj", file = #di_file, line = 12, scopeLine = 16, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable5, #di_local_variable6, #di_local_variable23, #di_local_variable7, #di_local_variable8, #di_local_variable2, #di_local_variable9, #di_local_variable10, #di_local_variable11, #di_local_variable24>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "lower_bound_dsa", linkageName = "_Z15lower_bound_dsaPKfS0_Pjjj", file = #di_file, line = 39, scopeLine = 43, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable12, #di_local_variable13, #di_local_variable25, #di_local_variable14, #di_local_variable15, #di_local_variable3, #di_local_variable16, #di_local_variable17, #di_local_variable18, #di_local_variable26>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 18>
#loc36 = loc(fused<#di_subprogram4>[#loc1])
#loc38 = loc(fused<#di_subprogram5>[#loc10])
#loc40 = loc(fused<#di_lexical_block9>[#loc23])
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 18>
#loc41 = loc(fused<#di_lexical_block10>[#loc3])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 18>
#loc48 = loc(fused<#di_lexical_block16>[#loc5])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<38xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 108, 111, 119, 101, 114, 95, 98, 111, 117, 110, 100, 47, 108, 111, 119, 101, 114, 95, 98, 111, 117, 110, 100, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @__const.main.input_sorted : memref<10xf32> = dense<[1.000000e+00, 3.000000e+00, 3.000000e+00, 5.000000e+00, 7.000000e+00, 9.000000e+00, 1.100000e+01, 1.300000e+01, 1.500000e+01, 1.700000e+01]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @__const.main.input_targets : memref<6xf32> = dense<[3.000000e+00, 0.000000e+00, 8.000000e+00, 2.000000e+01, 5.000000e+00, 1.100000e+01]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @str : memref<20xi8> = dense<[108, 111, 119, 101, 114, 95, 98, 111, 117, 110, 100, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<20xi8> = dense<[108, 111, 119, 101, 114, 95, 98, 111, 117, 110, 100, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z15lower_bound_cpuPKfS0_Pjjj(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg3: i32 loc(fused<#di_subprogram4>[#loc1]), %arg4: i32 loc(fused<#di_subprogram4>[#loc1])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg4, %c0_i32 : i32 loc(#loc44)
    scf.if %0 {
    } else {
      %1 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc2)
      %2 = arith.extui %arg4 : i32 to i64 loc(#loc44)
      %3 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        %4 = arith.index_cast %arg5 : i64 to index loc(#loc47)
        %5 = memref.load %arg1[%4] : memref<?xf32, strided<[1], offset: ?>> loc(#loc47)
        %6 = scf.if %1 -> (i32) {
          scf.yield %c0_i32 : i32 loc(#loc48)
        } else {
          %9:2 = scf.while (%arg6 = %arg3, %arg7 = %c0_i32) : (i32, i32) -> (i32, i32) {
            %10 = arith.subi %arg6, %arg7 : i32 loc(#loc55)
            %11 = arith.shrui %10, %c1_i32 : i32 loc(#loc55)
            %12 = arith.addi %11, %arg7 : i32 loc(#loc55)
            %13 = arith.extui %12 : i32 to i64 loc(#loc59)
            %14 = arith.index_cast %13 : i64 to index loc(#loc59)
            %15 = memref.load %arg0[%14] : memref<?xf32, strided<[1], offset: ?>> loc(#loc59)
            %16 = arith.cmpf olt, %15, %5 : f32 loc(#loc59)
            %17 = arith.addi %12, %c1_i32 : i32 loc(#loc59)
            %18 = arith.select %16, %17, %arg7 : i32 loc(#loc59)
            %19 = arith.select %16, %arg6, %12 : i32 loc(#loc59)
            %20 = arith.cmpi ult, %18, %19 : i32 loc(#loc48)
            scf.condition(%20) %19, %18 : i32, i32 loc(#loc48)
          } do {
          ^bb0(%arg6: i32 loc(fused<#di_lexical_block16>[#loc5]), %arg7: i32 loc(fused<#di_lexical_block16>[#loc5])):
            scf.yield %arg6, %arg7 : i32, i32 loc(#loc48)
          } loc(#loc48)
          scf.yield %9#1 : i32 loc(#loc48)
        } loc(#loc48)
        memref.store %6, %arg2[%4] : memref<?xi32, strided<[1], offset: ?>> loc(#loc49)
        %7 = arith.addi %arg5, %c1_i64 : i64 loc(#loc44)
        %8 = arith.cmpi ne, %7, %2 : i64 loc(#loc50)
        scf.condition(%8) %7 : i64 loc(#loc41)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block10>[#loc3])):
        scf.yield %arg5 : i64 loc(#loc41)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc41)
    } loc(#loc41)
    return loc(#loc37)
  } loc(#loc36)
  handshake.func @_Z15lower_bound_dsaPKfS0_Pjjj_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg3: i32 loc(fused<#di_subprogram5>[#loc10]), %arg4: i32 loc(fused<#di_subprogram5>[#loc10]), %arg5: i1 loc(fused<#di_subprogram5>[#loc10]), ...) -> i1 attributes {argNames = ["input_sorted", "input_targets", "output_indices", "N", "M", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : i1 loc(#loc38)
    %1 = handshake.join %0 : none loc(#loc38)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.cmpi eq, %arg4, %3 : i32 loc(#loc45)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc42)
    %6 = arith.cmpi eq, %arg3, %3 : i32 loc(#loc2)
    %7 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc42)
    %8 = arith.index_cast %4 : i64 to index loc(#loc42)
    %9 = arith.index_cast %arg4 : i32 to index loc(#loc42)
    %index, %willContinue = dataflow.stream %8, %7, %9 {loom.annotations = ["loom.loop.no_parallel", "loom.loop.no_unroll"], step_op = "+=", stop_cond = "!="} loc(#loc42)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc42)
    %10 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc42)
    %dataResult, %addressResults = handshake.load [%afterValue] %29#0, %32 : index, f32 loc(#loc51)
    %11 = dataflow.invariant %afterCond, %6 : i1, i1 -> i1 loc(#loc52)
    %12 = handshake.constant %1 {value = true} : i1 loc(#loc52)
    %13 = dataflow.carry %12, %arg3, %trueResult_2 : i1, i32, i32 -> i32 loc(#loc52)
    %14 = dataflow.carry %12, %3, %trueResult_4 : i1, i32, i32 -> i32 loc(#loc52)
    %15 = arith.subi %13, %14 : i32 loc(#loc56)
    %16 = arith.shrui %15, %2 : i32 loc(#loc56)
    %17 = arith.addi %16, %14 : i32 loc(#loc56)
    %18 = arith.extui %17 : i32 to i64 loc(#loc60)
    %19 = arith.index_cast %18 : i64 to index loc(#loc60)
    %dataResult_0, %addressResults_1 = handshake.load [%19] %30#0, %40 : index, f32 loc(#loc60)
    %20 = arith.cmpf olt, %dataResult_0, %dataResult : f32 loc(#loc60)
    %21 = arith.addi %17, %2 : i32 loc(#loc60)
    %22 = arith.select %20, %21, %14 : i32 loc(#loc60)
    %23 = arith.select %20, %13, %17 : i32 loc(#loc60)
    %24 = arith.cmpi ult, %22, %23 : i32 loc(#loc52)
    %trueResult_2, %falseResult_3 = handshake.cond_br %24, %23 : i32 loc(#loc52)
    handshake.sink %falseResult_3 : i32 loc(#loc52)
    %trueResult_4, %falseResult_5 = handshake.cond_br %24, %22 : i32 loc(#loc52)
    %25 = handshake.constant %10 {value = 0 : index} : index loc(#loc52)
    %26 = handshake.constant %10 {value = 1 : index} : index loc(#loc52)
    %27 = arith.select %11, %26, %25 : index loc(#loc52)
    %28 = handshake.mux %27 [%falseResult_5, %3] : index, i32 loc(#loc52)
    %dataResult_6, %addressResult = handshake.store [%afterValue] %28, %37 : index, i32 loc(#loc53)
    %29:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc38)
    %30:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (f32, none) loc(#loc38)
    %31 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_6, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc38)
    %32 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc42)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %29#1 : none loc(#loc42)
    %33 = handshake.constant %1 {value = 0 : index} : index loc(#loc42)
    %34 = handshake.constant %1 {value = 1 : index} : index loc(#loc42)
    %35 = arith.select %5, %34, %33 : index loc(#loc42)
    %36 = handshake.mux %35 [%falseResult_8, %trueResult] : index, none loc(#loc42)
    %37 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc42)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %31 : none loc(#loc42)
    %38 = handshake.mux %35 [%falseResult_10, %trueResult] : index, none loc(#loc42)
    %39 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc42)
    %trueResult_11, %falseResult_12 = handshake.cond_br %11, %39 : none loc(#loc52)
    %40 = dataflow.carry %24, %falseResult_12, %trueResult_13 : i1, none, none -> none loc(#loc52)
    %trueResult_13, %falseResult_14 = handshake.cond_br %24, %30#1 : none loc(#loc52)
    %41 = handshake.constant %39 {value = 0 : index} : index loc(#loc52)
    %42 = handshake.constant %39 {value = 1 : index} : index loc(#loc52)
    %43 = arith.select %11, %42, %41 : index loc(#loc52)
    %44 = handshake.mux %43 [%falseResult_14, %trueResult_11] : index, none loc(#loc52)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %44 : none loc(#loc42)
    %45 = handshake.mux %35 [%falseResult_16, %trueResult] : index, none loc(#loc42)
    %46 = handshake.join %36, %38, %45 : none, none, none loc(#loc38)
    %47 = handshake.constant %46 {value = true} : i1 loc(#loc38)
    handshake.return %47 : i1 loc(#loc38)
  } loc(#loc38)
  handshake.func @_Z15lower_bound_dsaPKfS0_Pjjj(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg3: i32 loc(fused<#di_subprogram5>[#loc10]), %arg4: i32 loc(fused<#di_subprogram5>[#loc10]), %arg5: none loc(fused<#di_subprogram5>[#loc10]), ...) -> none attributes {argNames = ["input_sorted", "input_targets", "output_indices", "N", "M", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : none loc(#loc38)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg4, %2 : i32 loc(#loc45)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc42)
    %5 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc2)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc42)
    %7 = arith.index_cast %3 : i64 to index loc(#loc42)
    %8 = arith.index_cast %arg4 : i32 to index loc(#loc42)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.no_parallel", "loom.loop.no_unroll"], step_op = "+=", stop_cond = "!="} loc(#loc42)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc42)
    %9 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc42)
    %dataResult, %addressResults = handshake.load [%afterValue] %28#0, %31 : index, f32 loc(#loc51)
    %10 = dataflow.invariant %afterCond, %5 : i1, i1 -> i1 loc(#loc52)
    %11 = handshake.constant %0 {value = true} : i1 loc(#loc52)
    %12 = dataflow.carry %11, %arg3, %trueResult_2 : i1, i32, i32 -> i32 loc(#loc52)
    %13 = dataflow.carry %11, %2, %trueResult_4 : i1, i32, i32 -> i32 loc(#loc52)
    %14 = arith.subi %12, %13 : i32 loc(#loc56)
    %15 = arith.shrui %14, %1 : i32 loc(#loc56)
    %16 = arith.addi %15, %13 : i32 loc(#loc56)
    %17 = arith.extui %16 : i32 to i64 loc(#loc60)
    %18 = arith.index_cast %17 : i64 to index loc(#loc60)
    %dataResult_0, %addressResults_1 = handshake.load [%18] %29#0, %39 : index, f32 loc(#loc60)
    %19 = arith.cmpf olt, %dataResult_0, %dataResult : f32 loc(#loc60)
    %20 = arith.addi %16, %1 : i32 loc(#loc60)
    %21 = arith.select %19, %20, %13 : i32 loc(#loc60)
    %22 = arith.select %19, %12, %16 : i32 loc(#loc60)
    %23 = arith.cmpi ult, %21, %22 : i32 loc(#loc52)
    %trueResult_2, %falseResult_3 = handshake.cond_br %23, %22 : i32 loc(#loc52)
    handshake.sink %falseResult_3 : i32 loc(#loc52)
    %trueResult_4, %falseResult_5 = handshake.cond_br %23, %21 : i32 loc(#loc52)
    %24 = handshake.constant %9 {value = 0 : index} : index loc(#loc52)
    %25 = handshake.constant %9 {value = 1 : index} : index loc(#loc52)
    %26 = arith.select %10, %25, %24 : index loc(#loc52)
    %27 = handshake.mux %26 [%falseResult_5, %2] : index, i32 loc(#loc52)
    %dataResult_6, %addressResult = handshake.store [%afterValue] %27, %36 : index, i32 loc(#loc53)
    %28:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc38)
    %29:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (f32, none) loc(#loc38)
    %30 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_6, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc38)
    %31 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc42)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %28#1 : none loc(#loc42)
    %32 = handshake.constant %0 {value = 0 : index} : index loc(#loc42)
    %33 = handshake.constant %0 {value = 1 : index} : index loc(#loc42)
    %34 = arith.select %4, %33, %32 : index loc(#loc42)
    %35 = handshake.mux %34 [%falseResult_8, %trueResult] : index, none loc(#loc42)
    %36 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc42)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %30 : none loc(#loc42)
    %37 = handshake.mux %34 [%falseResult_10, %trueResult] : index, none loc(#loc42)
    %38 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc42)
    %trueResult_11, %falseResult_12 = handshake.cond_br %10, %38 : none loc(#loc52)
    %39 = dataflow.carry %23, %falseResult_12, %trueResult_13 : i1, none, none -> none loc(#loc52)
    %trueResult_13, %falseResult_14 = handshake.cond_br %23, %29#1 : none loc(#loc52)
    %40 = handshake.constant %38 {value = 0 : index} : index loc(#loc52)
    %41 = handshake.constant %38 {value = 1 : index} : index loc(#loc52)
    %42 = arith.select %10, %41, %40 : index loc(#loc52)
    %43 = handshake.mux %42 [%falseResult_14, %trueResult_11] : index, none loc(#loc52)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %43 : none loc(#loc42)
    %44 = handshake.mux %34 [%falseResult_16, %trueResult] : index, none loc(#loc42)
    %45 = handshake.join %35, %37, %44 : none, none, none loc(#loc38)
    handshake.return %45 : none loc(#loc39)
  } loc(#loc38)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc29)
    %false = arith.constant false loc(#loc29)
    %0 = seq.const_clock  low loc(#loc29)
    %c2_i32 = arith.constant 2 : i32 loc(#loc29)
    %1 = ub.poison : i64 loc(#loc29)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c6 = arith.constant 6 : index loc(#loc30)
    %c10 = arith.constant 10 : index loc(#loc31)
    %c1 = arith.constant 1 : index loc(#loc2)
    %c6_i64 = arith.constant 6 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c6_i32 = arith.constant 6 : i32 loc(#loc2)
    %c10_i32 = arith.constant 10 : i32 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %2 = memref.get_global @__const.main.input_sorted : memref<10xf32> loc(#loc2)
    %3 = memref.get_global @__const.main.input_targets : memref<6xf32> loc(#loc2)
    %4 = memref.get_global @str : memref<20xi8> loc(#loc2)
    %5 = memref.get_global @str.2 : memref<20xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<10xf32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<6xf32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<6xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<6xi32> loc(#loc2)
    scf.for %arg0 = %c0 to %c10 step %c1 {
      %11 = memref.load %2[%arg0] : memref<10xf32> loc(#loc31)
      memref.store %11, %alloca[%arg0] : memref<10xf32> loc(#loc31)
    } loc(#loc31)
    scf.for %arg0 = %c0 to %c6 step %c1 {
      %11 = memref.load %3[%arg0] : memref<6xf32> loc(#loc30)
      memref.store %11, %alloca_0[%arg0] : memref<6xf32> loc(#loc30)
    } loc(#loc30)
    %cast = memref.cast %alloca : memref<10xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc32)
    %cast_3 = memref.cast %alloca_0 : memref<6xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc32)
    %cast_4 = memref.cast %alloca_1 : memref<6xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc32)
    call @_Z15lower_bound_cpuPKfS0_Pjjj(%cast, %cast_3, %cast_4, %c10_i32, %c6_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc32)
    %cast_5 = memref.cast %alloca_2 : memref<6xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c10_i32, %true : i32 loc(#loc33)
    %chanOutput_12, %ready_13 = esi.wrap.vr %c6_i32, %true : i32 loc(#loc33)
    %chanOutput_14, %ready_15 = esi.wrap.vr %true, %true : i1 loc(#loc33)
    %6 = handshake.esi_instance @_Z15lower_bound_dsaPKfS0_Pjjj_esi "_Z15lower_bound_dsaPKfS0_Pjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12, %chanOutput_14) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc33)
    %rawOutput, %valid = esi.unwrap.vr %6, %true : i1 loc(#loc33)
    %7:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc54)
      %12 = memref.load %alloca_1[%11] : memref<6xi32> loc(#loc54)
      %13 = memref.load %alloca_2[%11] : memref<6xi32> loc(#loc54)
      %14 = arith.cmpi eq, %12, %13 : i32 loc(#loc54)
      %15:3 = scf.if %14 -> (i64, i32, i32) {
        %17 = arith.addi %arg0, %c1_i64 : i64 loc(#loc43)
        %18 = arith.cmpi eq, %17, %c6_i64 : i64 loc(#loc43)
        %19 = arith.extui %18 : i1 to i32 loc(#loc40)
        %20 = arith.cmpi ne, %17, %c6_i64 : i64 loc(#loc46)
        %21 = arith.extui %20 : i1 to i32 loc(#loc40)
        scf.yield %17, %19, %21 : i64, i32, i32 loc(#loc54)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc54)
      } loc(#loc54)
      %16 = arith.trunci %15#2 : i32 to i1 loc(#loc40)
      scf.condition(%16) %15#0, %14, %15#1 : i64, i1, i32 loc(#loc40)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc23]), %arg1: i1 loc(fused<#di_lexical_block9>[#loc23]), %arg2: i32 loc(fused<#di_lexical_block9>[#loc23])):
      scf.yield %arg0 : i64 loc(#loc40)
    } loc(#loc40)
    %8 = arith.index_castui %7#2 : i32 to index loc(#loc40)
    %9 = scf.index_switch %8 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc40)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %4 : memref<20xi8> -> index loc(#loc57)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc57)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc57)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc57)
      scf.yield %c1_i32 : i32 loc(#loc58)
    } loc(#loc40)
    %10 = arith.select %7#1, %c0_i32, %9 : i32 loc(#loc2)
    scf.if %7#1 {
      %intptr = memref.extract_aligned_pointer_as_index %5 : memref<20xi8> -> index loc(#loc34)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc34)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc34)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc34)
    } loc(#loc2)
    return %10 : i32 loc(#loc35)
  } loc(#loc29)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.memcpy.p0.p0.i64(memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, i64, i1) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/lower_bound/lower_bound.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/lower_bound/lower_bound.cpp":19:0)
#loc6 = loc("tests/app/lower_bound/lower_bound.cpp":24:0)
#loc7 = loc("tests/app/lower_bound/lower_bound.cpp":26:0)
#loc8 = loc("tests/app/lower_bound/lower_bound.cpp":33:0)
#loc9 = loc("tests/app/lower_bound/lower_bound.cpp":35:0)
#loc11 = loc("tests/app/lower_bound/lower_bound.cpp":47:0)
#loc12 = loc("tests/app/lower_bound/lower_bound.cpp":48:0)
#loc13 = loc("tests/app/lower_bound/lower_bound.cpp":52:0)
#loc14 = loc("tests/app/lower_bound/lower_bound.cpp":53:0)
#loc15 = loc("tests/app/lower_bound/lower_bound.cpp":55:0)
#loc16 = loc("tests/app/lower_bound/lower_bound.cpp":62:0)
#loc17 = loc("tests/app/lower_bound/lower_bound.cpp":64:0)
#loc18 = loc("tests/app/lower_bound/main.cpp":5:0)
#loc19 = loc("tests/app/lower_bound/main.cpp":13:0)
#loc20 = loc("tests/app/lower_bound/main.cpp":10:0)
#loc21 = loc("tests/app/lower_bound/main.cpp":20:0)
#loc22 = loc("tests/app/lower_bound/main.cpp":23:0)
#loc24 = loc("tests/app/lower_bound/main.cpp":27:0)
#loc25 = loc("tests/app/lower_bound/main.cpp":28:0)
#loc26 = loc("tests/app/lower_bound/main.cpp":29:0)
#loc27 = loc("tests/app/lower_bound/main.cpp":33:0)
#loc28 = loc("tests/app/lower_bound/main.cpp":35:0)
#loc29 = loc(fused<#di_subprogram3>[#loc18])
#loc30 = loc(fused<#di_subprogram3>[#loc19])
#loc31 = loc(fused<#di_subprogram3>[#loc20])
#loc32 = loc(fused<#di_subprogram3>[#loc21])
#loc33 = loc(fused<#di_subprogram3>[#loc22])
#loc34 = loc(fused<#di_subprogram3>[#loc27])
#loc35 = loc(fused<#di_subprogram3>[#loc28])
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 47>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file1, line = 26>
#loc37 = loc(fused<#di_subprogram4>[#loc9])
#loc39 = loc(fused<#di_subprogram5>[#loc17])
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 47>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file1, line = 26>
#loc42 = loc(fused<#di_lexical_block11>[#loc11])
#loc43 = loc(fused<#di_lexical_block12>[#loc23])
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 47>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 27>
#loc44 = loc(fused<#di_lexical_block13>[#loc3])
#loc45 = loc(fused<#di_lexical_block14>[#loc11])
#loc46 = loc(fused[#loc40, #loc43])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 23>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 52>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 27>
#loc47 = loc(fused<#di_lexical_block16>[#loc4])
#loc49 = loc(fused<#di_lexical_block16>[#loc8])
#loc50 = loc(fused[#loc41, #loc44])
#loc51 = loc(fused<#di_lexical_block17>[#loc12])
#loc52 = loc(fused<#di_lexical_block17>[#loc13])
#loc53 = loc(fused<#di_lexical_block17>[#loc16])
#loc54 = loc(fused<#di_lexical_block18>[#loc24])
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 26>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 55>
#loc55 = loc(fused<#di_lexical_block19>[#loc6])
#loc56 = loc(fused<#di_lexical_block20>[#loc14])
#loc57 = loc(fused<#di_lexical_block21>[#loc25])
#loc58 = loc(fused<#di_lexical_block21>[#loc26])
#loc59 = loc(fused<#di_lexical_block22>[#loc7])
#loc60 = loc(fused<#di_lexical_block23>[#loc15])
