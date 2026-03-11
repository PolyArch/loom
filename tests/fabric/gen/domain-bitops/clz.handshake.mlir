#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/clz/clz.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/clz/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/clz/clz.cpp":14:0)
#loc3 = loc("tests/app/clz/clz.cpp":17:0)
#loc6 = loc("tests/app/clz/clz.cpp":26:0)
#loc10 = loc("tests/app/clz/clz.cpp":39:0)
#loc24 = loc("tests/app/clz/main.cpp":16:0)
#loc28 = loc("tests/app/clz/main.cpp":31:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 17>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 44>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 16>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 31>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 17>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 44>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 8192, elements = #llvm.di_subrange<count = 256 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 17>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 44>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 17, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 44, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 16, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 31, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 20>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 47>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 16, arg = 3, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "value", file = #di_file, line = 18, type = #di_derived_type1>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 41, arg = 3, type = #di_derived_type2>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "value", file = #di_file, line = 45, type = #di_derived_type1>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 6, type = #di_derived_type2>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file1, line = 9, type = #di_composite_type>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_count", file = #di_file1, line = 21, type = #di_composite_type>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_count", file = #di_file1, line = 22, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 22>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 49>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_count", file = #di_file, line = 15, arg = 2, type = #di_derived_type5>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_count", file = #di_file, line = 40, arg = 2, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable8, #di_local_variable9, #di_local_variable2, #di_local_variable10, #di_local_variable11, #di_local_variable3>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 16>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 31>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_data", file = #di_file, line = 14, arg = 1, type = #di_derived_type6>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block10, name = "count", file = #di_file, line = 23, type = #di_derived_type1>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_lexical_block10, name = "mask", file = #di_file, line = 24, type = #di_derived_type1>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_data", file = #di_file, line = 39, arg = 1, type = #di_derived_type6>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block11, name = "count", file = #di_file, line = 50, type = #di_derived_type1>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_lexical_block11, name = "mask", file = #di_file, line = 51, type = #di_derived_type1>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "clz_cpu", linkageName = "_Z7clz_cpuPKjPjj", file = #di_file, line = 14, scopeLine = 16, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable14, #di_local_variable12, #di_local_variable4, #di_local_variable, #di_local_variable5, #di_local_variable15, #di_local_variable16>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "clz_dsa", linkageName = "_Z7clz_dsaPKjPjj", file = #di_file, line = 39, scopeLine = 41, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable17, #di_local_variable13, #di_local_variable6, #di_local_variable1, #di_local_variable7, #di_local_variable18, #di_local_variable19>
#loc44 = loc(fused<#di_lexical_block12>[#loc24])
#loc45 = loc(fused<#di_lexical_block13>[#loc28])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 17>
#loc46 = loc(fused<#di_subprogram4>[#loc1])
#loc48 = loc(fused<#di_subprogram5>[#loc10])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 17>
#loc52 = loc(fused<#di_lexical_block16>[#loc3])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 17>
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file, line = 20>
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file, line = 22>
#loc69 = loc(fused<#di_lexical_block28>[#loc6])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<22xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 99, 108, 122, 47, 99, 108, 122, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @str : memref<12xi8> = dense<[99, 108, 122, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<12xi8> = dense<[99, 108, 122, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z7clz_cpuPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: i32 loc(fused<#di_subprogram4>[#loc1])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c32_i32 = arith.constant 32 : i32 loc(#loc2)
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %c-2147483648_i32 = arith.constant -2147483648 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg2, %c0_i32 : i32 loc(#loc57)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg2 : i32 to i64 loc(#loc57)
      %2 = scf.while (%arg3 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg3 : i64 to index loc(#loc60)
        %4 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc60)
        %5 = arith.cmpi eq, %4, %c0_i32 : i32 loc(#loc65)
        %6 = scf.if %5 -> (i32) {
          scf.yield %c32_i32 : i32 loc(#loc65)
        } else {
          %9 = arith.cmpi sgt, %4, %c-1_i32 : i32 loc(#loc69)
          %10 = scf.if %9 -> (i32) {
            %11:2 = scf.while (%arg4 = %c-2147483648_i32, %arg5 = %c0_i32) : (i32, i32) -> (i32, i32) {
              %12 = arith.addi %arg5, %c1_i32 : i32 loc(#loc71)
              %13 = arith.shrui %arg4, %c1_i32 : i32 loc(#loc72)
              %14 = arith.andi %13, %4 : i32 loc(#loc69)
              %15 = arith.cmpi eq, %14, %c0_i32 : i32 loc(#loc69)
              scf.condition(%15) %13, %12 : i32, i32 loc(#loc69)
            } do {
            ^bb0(%arg4: i32 loc(fused<#di_lexical_block28>[#loc6]), %arg5: i32 loc(fused<#di_lexical_block28>[#loc6])):
              scf.yield %arg4, %arg5 : i32, i32 loc(#loc69)
            } loc(#loc69)
            scf.yield %11#1 : i32 loc(#loc69)
          } else {
            scf.yield %c0_i32 : i32 loc(#loc69)
          } loc(#loc69)
          scf.yield %10 : i32 loc(#loc65)
        } loc(#loc65)
        memref.store %6, %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc66)
        %7 = arith.addi %arg3, %c1_i64 : i64 loc(#loc57)
        %8 = arith.cmpi ne, %7, %1 : i64 loc(#loc61)
        scf.condition(%8) %7 : i64 loc(#loc52)
      } do {
      ^bb0(%arg3: i64 loc(fused<#di_lexical_block16>[#loc3])):
        scf.yield %arg3 : i64 loc(#loc52)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc52)
    } loc(#loc52)
    return loc(#loc47)
  } loc(#loc46)
  handshake.func @_Z7clz_dsaPKjPjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg2: i32 loc(fused<#di_subprogram5>[#loc10]), %arg3: i1 loc(fused<#di_subprogram5>[#loc10]), ...) -> i1 attributes {argNames = ["input_data", "output_count", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc48)
    %1 = handshake.join %0 : none loc(#loc48)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %1 {value = 32 : i32} : i32 loc(#loc2)
    %6 = handshake.constant %1 {value = -1 : i32} : i32 loc(#loc2)
    %7 = handshake.constant %1 {value = -2147483648 : i32} : i32 loc(#loc2)
    %8 = arith.cmpi eq, %arg2, %3 : i32 loc(#loc58)
    %trueResult, %falseResult = handshake.cond_br %8, %1 : none loc(#loc53)
    %9 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc53)
    %10 = arith.index_cast %4 : i64 to index loc(#loc53)
    %11 = arith.index_cast %arg2 : i32 to index loc(#loc53)
    %index, %willContinue = dataflow.stream %10, %9, %11 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc53)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc53)
    %12 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc53)
    %dataResult, %addressResults = handshake.load [%afterValue] %30#0, %32 : index, i32 loc(#loc62)
    %13 = arith.cmpi eq, %dataResult, %3 : i32 loc(#loc67)
    %trueResult_0, %falseResult_1 = handshake.cond_br %13, %12 : none loc(#loc67)
    %14 = arith.cmpi sgt, %dataResult, %6 : i32 loc(#loc70)
    %15 = handshake.constant %1 {value = true} : i1 loc(#loc70)
    %16 = dataflow.carry %15, %7, %trueResult_2 : i1, i32, i32 -> i32 loc(#loc70)
    %17 = dataflow.carry %15, %3, %trueResult_4 : i1, i32, i32 -> i32 loc(#loc70)
    %18 = arith.addi %17, %2 : i32 loc(#loc73)
    %19 = arith.shrui %16, %2 : i32 loc(#loc74)
    %20 = arith.andi %19, %dataResult : i32 loc(#loc70)
    %21 = arith.cmpi eq, %20, %3 : i32 loc(#loc70)
    %trueResult_2, %falseResult_3 = handshake.cond_br %21, %19 : i32 loc(#loc70)
    handshake.sink %falseResult_3 : i32 loc(#loc70)
    %trueResult_4, %falseResult_5 = handshake.cond_br %21, %18 : i32 loc(#loc70)
    %22 = handshake.constant %falseResult_1 {value = 0 : index} : index loc(#loc70)
    %23 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc70)
    %24 = arith.select %14, %23, %22 : index loc(#loc70)
    %25 = handshake.mux %24 [%3, %falseResult_5] : index, i32 loc(#loc70)
    %26 = handshake.constant %12 {value = 0 : index} : index loc(#loc67)
    %27 = handshake.constant %12 {value = 1 : index} : index loc(#loc67)
    %28 = arith.select %13, %27, %26 : index loc(#loc67)
    %29 = handshake.mux %28 [%25, %5] : index, i32 loc(#loc67)
    %dataResult_6, %addressResult = handshake.store [%afterValue] %29, %37 : index, i32 loc(#loc68)
    %30:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc48)
    %31 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_6, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc48)
    %32 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc53)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %30#1 : none loc(#loc53)
    %33 = handshake.constant %1 {value = 0 : index} : index loc(#loc53)
    %34 = handshake.constant %1 {value = 1 : index} : index loc(#loc53)
    %35 = arith.select %8, %34, %33 : index loc(#loc53)
    %36 = handshake.mux %35 [%falseResult_8, %trueResult] : index, none loc(#loc53)
    %37 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc53)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %31 : none loc(#loc53)
    %38 = handshake.mux %35 [%falseResult_10, %trueResult] : index, none loc(#loc53)
    %39 = handshake.join %36, %38 : none, none loc(#loc48)
    %40 = handshake.constant %39 {value = true} : i1 loc(#loc48)
    handshake.return %40 : i1 loc(#loc48)
  } loc(#loc48)
  handshake.func @_Z7clz_dsaPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg2: i32 loc(fused<#di_subprogram5>[#loc10]), %arg3: none loc(fused<#di_subprogram5>[#loc10]), ...) -> none attributes {argNames = ["input_data", "output_count", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc48)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %0 {value = 32 : i32} : i32 loc(#loc2)
    %5 = handshake.constant %0 {value = -1 : i32} : i32 loc(#loc2)
    %6 = handshake.constant %0 {value = -2147483648 : i32} : i32 loc(#loc2)
    %7 = arith.cmpi eq, %arg2, %2 : i32 loc(#loc58)
    %trueResult, %falseResult = handshake.cond_br %7, %0 : none loc(#loc53)
    %8 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc53)
    %9 = arith.index_cast %3 : i64 to index loc(#loc53)
    %10 = arith.index_cast %arg2 : i32 to index loc(#loc53)
    %index, %willContinue = dataflow.stream %9, %8, %10 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc53)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc53)
    %11 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc53)
    %dataResult, %addressResults = handshake.load [%afterValue] %29#0, %31 : index, i32 loc(#loc62)
    %12 = arith.cmpi eq, %dataResult, %2 : i32 loc(#loc67)
    %trueResult_0, %falseResult_1 = handshake.cond_br %12, %11 : none loc(#loc67)
    %13 = arith.cmpi sgt, %dataResult, %5 : i32 loc(#loc70)
    %14 = handshake.constant %0 {value = true} : i1 loc(#loc70)
    %15 = dataflow.carry %14, %6, %trueResult_2 : i1, i32, i32 -> i32 loc(#loc70)
    %16 = dataflow.carry %14, %2, %trueResult_4 : i1, i32, i32 -> i32 loc(#loc70)
    %17 = arith.addi %16, %1 : i32 loc(#loc73)
    %18 = arith.shrui %15, %1 : i32 loc(#loc74)
    %19 = arith.andi %18, %dataResult : i32 loc(#loc70)
    %20 = arith.cmpi eq, %19, %2 : i32 loc(#loc70)
    %trueResult_2, %falseResult_3 = handshake.cond_br %20, %18 : i32 loc(#loc70)
    handshake.sink %falseResult_3 : i32 loc(#loc70)
    %trueResult_4, %falseResult_5 = handshake.cond_br %20, %17 : i32 loc(#loc70)
    %21 = handshake.constant %falseResult_1 {value = 0 : index} : index loc(#loc70)
    %22 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc70)
    %23 = arith.select %13, %22, %21 : index loc(#loc70)
    %24 = handshake.mux %23 [%2, %falseResult_5] : index, i32 loc(#loc70)
    %25 = handshake.constant %11 {value = 0 : index} : index loc(#loc67)
    %26 = handshake.constant %11 {value = 1 : index} : index loc(#loc67)
    %27 = arith.select %12, %26, %25 : index loc(#loc67)
    %28 = handshake.mux %27 [%24, %4] : index, i32 loc(#loc67)
    %dataResult_6, %addressResult = handshake.store [%afterValue] %28, %36 : index, i32 loc(#loc68)
    %29:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc48)
    %30 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_6, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc48)
    %31 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc53)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %29#1 : none loc(#loc53)
    %32 = handshake.constant %0 {value = 0 : index} : index loc(#loc53)
    %33 = handshake.constant %0 {value = 1 : index} : index loc(#loc53)
    %34 = arith.select %7, %33, %32 : index loc(#loc53)
    %35 = handshake.mux %34 [%falseResult_8, %trueResult] : index, none loc(#loc53)
    %36 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc53)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %30 : none loc(#loc53)
    %37 = handshake.mux %34 [%falseResult_10, %trueResult] : index, none loc(#loc53)
    %38 = handshake.join %35, %37 : none, none loc(#loc48)
    handshake.return %38 : none loc(#loc49)
  } loc(#loc48)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc34)
    %false = arith.constant false loc(#loc34)
    %0 = seq.const_clock  low loc(#loc34)
    %c2_i32 = arith.constant 2 : i32 loc(#loc34)
    %1 = ub.poison : i64 loc(#loc34)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c4 = arith.constant 4 : index loc(#loc35)
    %c3 = arith.constant 3 : index loc(#loc36)
    %c2 = arith.constant 2 : index loc(#loc37)
    %c1 = arith.constant 1 : index loc(#loc38)
    %c0 = arith.constant 0 : index loc(#loc2)
    %c-2147483648_i32 = arith.constant -2147483648 : i32 loc(#loc2)
    %c1073741824_i32 = arith.constant 1073741824 : i32 loc(#loc2)
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %c5_i64 = arith.constant 5 : i64 loc(#loc2)
    %c4660_i32 = arith.constant 4660 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c256_i64 = arith.constant 256 : i64 loc(#loc2)
    %c256_i32 = arith.constant 256 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %2 = memref.get_global @str : memref<12xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<12xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<256xi32> loc(#loc2)
    memref.store %c0_i32, %alloca[%c0] : memref<256xi32> loc(#loc39)
    memref.store %c-2147483648_i32, %alloca[%c1] : memref<256xi32> loc(#loc38)
    memref.store %c1073741824_i32, %alloca[%c2] : memref<256xi32> loc(#loc37)
    memref.store %c1_i32, %alloca[%c3] : memref<256xi32> loc(#loc36)
    memref.store %c-1_i32, %alloca[%c4] : memref<256xi32> loc(#loc35)
    %4 = scf.while (%arg0 = %c5_i64) : (i64) -> i64 {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc54)
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc54)
      %12 = arith.muli %11, %c4660_i32 : i32 loc(#loc54)
      memref.store %12, %alloca[%10] : memref<256xi32> loc(#loc54)
      %13 = arith.addi %arg0, %c1_i64 : i64 loc(#loc50)
      %14 = arith.cmpi ne, %13, %c256_i64 : i64 loc(#loc55)
      scf.condition(%14) %13 : i64 loc(#loc44)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block12>[#loc24])):
      scf.yield %arg0 : i64 loc(#loc44)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc44)
    %cast = memref.cast %alloca : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc40)
    %cast_2 = memref.cast %alloca_0 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc40)
    call @_Z7clz_cpuPKjPjj(%cast, %cast_2, %c256_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc40)
    %cast_3 = memref.cast %alloca_1 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc41)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc41)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc41)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c256_i32, %true : i32 loc(#loc41)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc41)
    %5 = handshake.esi_instance @_Z7clz_dsaPKjPjj_esi "_Z7clz_dsaPKjPjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc41)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc41)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc59)
      %11 = memref.load %alloca_0[%10] : memref<256xi32> loc(#loc59)
      %12 = memref.load %alloca_1[%10] : memref<256xi32> loc(#loc59)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc59)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc51)
        %17 = arith.cmpi eq, %16, %c256_i64 : i64 loc(#loc51)
        %18 = arith.extui %17 : i1 to i32 loc(#loc45)
        %19 = arith.cmpi ne, %16, %c256_i64 : i64 loc(#loc56)
        %20 = arith.extui %19 : i1 to i32 loc(#loc45)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc59)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc59)
      } loc(#loc59)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc45)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc45)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block13>[#loc28]), %arg1: i1 loc(fused<#di_lexical_block13>[#loc28]), %arg2: i32 loc(fused<#di_lexical_block13>[#loc28])):
      scf.yield %arg0 : i64 loc(#loc45)
    } loc(#loc45)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc45)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc45)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<12xi8> -> index loc(#loc63)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc63)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc63)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc63)
      scf.yield %c1_i32 : i32 loc(#loc64)
    } loc(#loc45)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<12xi8> -> index loc(#loc42)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc42)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc42)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc42)
    } loc(#loc2)
    return %9 : i32 loc(#loc43)
  } loc(#loc34)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/clz/clz.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/clz/clz.cpp":18:0)
#loc5 = loc("tests/app/clz/clz.cpp":20:0)
#loc7 = loc("tests/app/clz/clz.cpp":27:0)
#loc8 = loc("tests/app/clz/clz.cpp":28:0)
#loc9 = loc("tests/app/clz/clz.cpp":34:0)
#loc11 = loc("tests/app/clz/clz.cpp":44:0)
#loc12 = loc("tests/app/clz/clz.cpp":45:0)
#loc13 = loc("tests/app/clz/clz.cpp":47:0)
#loc14 = loc("tests/app/clz/clz.cpp":53:0)
#loc15 = loc("tests/app/clz/clz.cpp":54:0)
#loc16 = loc("tests/app/clz/clz.cpp":55:0)
#loc17 = loc("tests/app/clz/clz.cpp":61:0)
#loc18 = loc("tests/app/clz/main.cpp":5:0)
#loc19 = loc("tests/app/clz/main.cpp":14:0)
#loc20 = loc("tests/app/clz/main.cpp":13:0)
#loc21 = loc("tests/app/clz/main.cpp":12:0)
#loc22 = loc("tests/app/clz/main.cpp":11:0)
#loc23 = loc("tests/app/clz/main.cpp":10:0)
#loc25 = loc("tests/app/clz/main.cpp":17:0)
#loc26 = loc("tests/app/clz/main.cpp":25:0)
#loc27 = loc("tests/app/clz/main.cpp":28:0)
#loc29 = loc("tests/app/clz/main.cpp":32:0)
#loc30 = loc("tests/app/clz/main.cpp":33:0)
#loc31 = loc("tests/app/clz/main.cpp":34:0)
#loc32 = loc("tests/app/clz/main.cpp":38:0)
#loc33 = loc("tests/app/clz/main.cpp":40:0)
#loc34 = loc(fused<#di_subprogram3>[#loc18])
#loc35 = loc(fused<#di_subprogram3>[#loc19])
#loc36 = loc(fused<#di_subprogram3>[#loc20])
#loc37 = loc(fused<#di_subprogram3>[#loc21])
#loc38 = loc(fused<#di_subprogram3>[#loc22])
#loc39 = loc(fused<#di_subprogram3>[#loc23])
#loc40 = loc(fused<#di_subprogram3>[#loc26])
#loc41 = loc(fused<#di_subprogram3>[#loc27])
#loc42 = loc(fused<#di_subprogram3>[#loc32])
#loc43 = loc(fused<#di_subprogram3>[#loc33])
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file1, line = 16>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file1, line = 31>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 44>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 16>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 31>
#loc47 = loc(fused<#di_subprogram4>[#loc9])
#loc49 = loc(fused<#di_subprogram5>[#loc17])
#loc50 = loc(fused<#di_lexical_block14>[#loc24])
#loc51 = loc(fused<#di_lexical_block15>[#loc28])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 44>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 32>
#loc53 = loc(fused<#di_lexical_block17>[#loc11])
#loc54 = loc(fused<#di_lexical_block18>[#loc25])
#loc55 = loc(fused[#loc44, #loc50])
#loc56 = loc(fused[#loc45, #loc51])
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file, line = 44>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 32>
#loc57 = loc(fused<#di_lexical_block20>[#loc3])
#loc58 = loc(fused<#di_lexical_block21>[#loc11])
#loc59 = loc(fused<#di_lexical_block22>[#loc29])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file, line = 47>
#loc60 = loc(fused<#di_lexical_block23>[#loc4])
#loc61 = loc(fused[#loc52, #loc57])
#loc62 = loc(fused<#di_lexical_block24>[#loc12])
#loc63 = loc(fused<#di_lexical_block25>[#loc30])
#loc64 = loc(fused<#di_lexical_block25>[#loc31])
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file, line = 49>
#loc65 = loc(fused<#di_lexical_block26>[#loc5])
#loc66 = loc(fused<#di_lexical_block26>[#loc])
#loc67 = loc(fused<#di_lexical_block27>[#loc13])
#loc68 = loc(fused<#di_lexical_block27>[#loc])
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file, line = 26>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block29, file = #di_file, line = 53>
#loc70 = loc(fused<#di_lexical_block29>[#loc14])
#loc71 = loc(fused<#di_lexical_block30>[#loc7])
#loc72 = loc(fused<#di_lexical_block30>[#loc8])
#loc73 = loc(fused<#di_lexical_block31>[#loc15])
#loc74 = loc(fused<#di_lexical_block31>[#loc16])
