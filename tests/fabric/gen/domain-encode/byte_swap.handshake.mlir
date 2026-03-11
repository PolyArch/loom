#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/byte_swap/byte_swap.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/byte_swap/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/byte_swap/byte_swap.cpp":15:0)
#loc3 = loc("tests/app/byte_swap/byte_swap.cpp":18:0)
#loc7 = loc("tests/app/byte_swap/byte_swap.cpp":32:0)
#loc20 = loc("tests/app/byte_swap/main.cpp":18:0)
#loc24 = loc("tests/app/byte_swap/main.cpp":33:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 18>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 37>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 18>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 33>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 18>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 37>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 8192, elements = #llvm.di_subrange<count = 256 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 18>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 37>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 18, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 37, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 18, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 33, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 17, arg = 3, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "value", file = #di_file, line = 19, type = #di_derived_type1>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "byte0", file = #di_file, line = 21, type = #di_derived_type1>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "byte1", file = #di_file, line = 22, type = #di_derived_type1>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "byte2", file = #di_file, line = 23, type = #di_derived_type1>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "byte3", file = #di_file, line = 24, type = #di_derived_type1>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 34, arg = 3, type = #di_derived_type2>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "value", file = #di_file, line = 38, type = #di_derived_type1>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "byte0", file = #di_file, line = 40, type = #di_derived_type1>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "byte1", file = #di_file, line = 41, type = #di_derived_type1>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "byte2", file = #di_file, line = 42, type = #di_derived_type1>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "byte3", file = #di_file, line = 43, type = #di_derived_type1>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 6, type = #di_derived_type2>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file1, line = 9, type = #di_composite_type>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_swapped", file = #di_file1, line = 23, type = #di_composite_type>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_swapped", file = #di_file1, line = 24, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_swapped", file = #di_file, line = 16, arg = 2, type = #di_derived_type5>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_swapped", file = #di_file, line = 33, arg = 2, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable16, #di_local_variable17, #di_local_variable2, #di_local_variable18, #di_local_variable19, #di_local_variable3>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 18>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 33>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_data", file = #di_file, line = 15, arg = 1, type = #di_derived_type6>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_data", file = #di_file, line = 32, arg = 1, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "byte_swap_cpu", linkageName = "_Z13byte_swap_cpuPKjPjj", file = #di_file, line = 15, scopeLine = 17, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable22, #di_local_variable20, #di_local_variable4, #di_local_variable, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable9>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "byte_swap_dsa", linkageName = "_Z13byte_swap_dsaPKjPjj", file = #di_file, line = 32, scopeLine = 34, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable23, #di_local_variable21, #di_local_variable10, #di_local_variable1, #di_local_variable11, #di_local_variable12, #di_local_variable13, #di_local_variable14, #di_local_variable15>
#loc42 = loc(fused<#di_lexical_block8>[#loc20])
#loc43 = loc(fused<#di_lexical_block9>[#loc24])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 18>
#loc44 = loc(fused<#di_subprogram4>[#loc1])
#loc46 = loc(fused<#di_subprogram5>[#loc7])
#loc50 = loc(fused<#di_lexical_block12>[#loc3])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<34xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 98, 121, 116, 101, 95, 115, 119, 97, 112, 47, 98, 121, 116, 101, 95, 115, 119, 97, 112, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @str : memref<18xi8> = dense<[98, 121, 116, 101, 95, 115, 119, 97, 112, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<18xi8> = dense<[98, 121, 116, 101, 95, 115, 119, 97, 112, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z13byte_swap_cpuPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: i32 loc(fused<#di_subprogram4>[#loc1])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c16_i32 = arith.constant 16 : i32 loc(#loc2)
    %c8_i32 = arith.constant 8 : i32 loc(#loc2)
    %c255_i32 = arith.constant 255 : i32 loc(#loc2)
    %c24_i32 = arith.constant 24 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg2, %c0_i32 : i32 loc(#loc55)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg2 : i32 to i64 loc(#loc55)
      %2 = scf.while (%arg3 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg3 : i64 to index loc(#loc58)
        %4 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc58)
        %5 = arith.andi %4, %c255_i32 : i32 loc(#loc59)
        %6 = arith.shli %5, %c24_i32 : i32 loc(#loc59)
        %7 = arith.shrui %4, %c8_i32 : i32 loc(#loc59)
        %8 = arith.andi %7, %c255_i32 : i32 loc(#loc59)
        %9 = arith.shli %8, %c16_i32 : i32 loc(#loc59)
        %10 = arith.ori %6, %9 : i32 loc(#loc59)
        %11 = arith.shrui %4, %c16_i32 : i32 loc(#loc59)
        %12 = arith.andi %11, %c255_i32 : i32 loc(#loc59)
        %13 = arith.shli %12, %c8_i32 : i32 loc(#loc59)
        %14 = arith.ori %10, %13 : i32 loc(#loc59)
        %15 = arith.shrui %4, %c24_i32 : i32 loc(#loc59)
        %16 = arith.andi %15, %c255_i32 : i32 loc(#loc59)
        %17 = arith.ori %14, %16 : i32 loc(#loc59)
        memref.store %17, %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc59)
        %18 = arith.addi %arg3, %c1_i64 : i64 loc(#loc55)
        %19 = arith.cmpi ne, %18, %1 : i64 loc(#loc60)
        scf.condition(%19) %18 : i64 loc(#loc50)
      } do {
      ^bb0(%arg3: i64 loc(fused<#di_lexical_block12>[#loc3])):
        scf.yield %arg3 : i64 loc(#loc50)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc50)
    } loc(#loc50)
    return loc(#loc45)
  } loc(#loc44)
  handshake.func @_Z13byte_swap_dsaPKjPjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc7]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc7]), %arg2: i32 loc(fused<#di_subprogram5>[#loc7]), %arg3: i1 loc(fused<#di_subprogram5>[#loc7]), ...) -> i1 attributes {argNames = ["input_data", "output_swapped", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc46)
    %1 = handshake.join %0 : none loc(#loc46)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 16 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 255 : i32} : i32 loc(#loc2)
    %5 = handshake.constant %1 {value = 24 : i32} : i32 loc(#loc2)
    %6 = handshake.constant %1 {value = 8 : i32} : i32 loc(#loc2)
    %7 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %8 = arith.cmpi eq, %arg2, %2 : i32 loc(#loc56)
    %trueResult, %falseResult = handshake.cond_br %8, %1 : none loc(#loc51)
    %9 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc51)
    %10 = arith.index_cast %7 : i64 to index loc(#loc51)
    %11 = arith.index_cast %arg2 : i32 to index loc(#loc51)
    %index, %willContinue = dataflow.stream %10, %9, %11 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc51)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc51)
    %dataResult, %addressResults = handshake.load [%afterValue] %25#0, %27 : index, i32 loc(#loc61)
    %12 = arith.andi %dataResult, %4 : i32 loc(#loc62)
    %13 = arith.shli %12, %5 : i32 loc(#loc62)
    %14 = arith.shrui %dataResult, %6 : i32 loc(#loc62)
    %15 = arith.andi %14, %4 : i32 loc(#loc62)
    %16 = arith.shli %15, %3 : i32 loc(#loc62)
    %17 = arith.ori %13, %16 : i32 loc(#loc62)
    %18 = arith.shrui %dataResult, %3 : i32 loc(#loc62)
    %19 = arith.andi %18, %4 : i32 loc(#loc62)
    %20 = arith.shli %19, %6 : i32 loc(#loc62)
    %21 = arith.ori %17, %20 : i32 loc(#loc62)
    %22 = arith.shrui %dataResult, %5 : i32 loc(#loc62)
    %23 = arith.andi %22, %4 : i32 loc(#loc62)
    %24 = arith.ori %21, %23 : i32 loc(#loc62)
    %dataResult_0, %addressResult = handshake.store [%afterValue] %24, %32 : index, i32 loc(#loc62)
    %25:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc46)
    %26 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_0, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc46)
    %27 = dataflow.carry %willContinue, %falseResult, %trueResult_1 : i1, none, none -> none loc(#loc51)
    %trueResult_1, %falseResult_2 = handshake.cond_br %willContinue, %25#1 : none loc(#loc51)
    %28 = handshake.constant %1 {value = 0 : index} : index loc(#loc51)
    %29 = handshake.constant %1 {value = 1 : index} : index loc(#loc51)
    %30 = arith.select %8, %29, %28 : index loc(#loc51)
    %31 = handshake.mux %30 [%falseResult_2, %trueResult] : index, none loc(#loc51)
    %32 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc51)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %26 : none loc(#loc51)
    %33 = handshake.mux %30 [%falseResult_4, %trueResult] : index, none loc(#loc51)
    %34 = handshake.join %31, %33 : none, none loc(#loc46)
    %35 = handshake.constant %34 {value = true} : i1 loc(#loc46)
    handshake.return %35 : i1 loc(#loc46)
  } loc(#loc46)
  handshake.func @_Z13byte_swap_dsaPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc7]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc7]), %arg2: i32 loc(fused<#di_subprogram5>[#loc7]), %arg3: none loc(fused<#di_subprogram5>[#loc7]), ...) -> none attributes {argNames = ["input_data", "output_swapped", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc46)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 16 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 255 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %0 {value = 24 : i32} : i32 loc(#loc2)
    %5 = handshake.constant %0 {value = 8 : i32} : i32 loc(#loc2)
    %6 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %7 = arith.cmpi eq, %arg2, %1 : i32 loc(#loc56)
    %trueResult, %falseResult = handshake.cond_br %7, %0 : none loc(#loc51)
    %8 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc51)
    %9 = arith.index_cast %6 : i64 to index loc(#loc51)
    %10 = arith.index_cast %arg2 : i32 to index loc(#loc51)
    %index, %willContinue = dataflow.stream %9, %8, %10 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc51)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc51)
    %dataResult, %addressResults = handshake.load [%afterValue] %24#0, %26 : index, i32 loc(#loc61)
    %11 = arith.andi %dataResult, %3 : i32 loc(#loc62)
    %12 = arith.shli %11, %4 : i32 loc(#loc62)
    %13 = arith.shrui %dataResult, %5 : i32 loc(#loc62)
    %14 = arith.andi %13, %3 : i32 loc(#loc62)
    %15 = arith.shli %14, %2 : i32 loc(#loc62)
    %16 = arith.ori %12, %15 : i32 loc(#loc62)
    %17 = arith.shrui %dataResult, %2 : i32 loc(#loc62)
    %18 = arith.andi %17, %3 : i32 loc(#loc62)
    %19 = arith.shli %18, %5 : i32 loc(#loc62)
    %20 = arith.ori %16, %19 : i32 loc(#loc62)
    %21 = arith.shrui %dataResult, %4 : i32 loc(#loc62)
    %22 = arith.andi %21, %3 : i32 loc(#loc62)
    %23 = arith.ori %20, %22 : i32 loc(#loc62)
    %dataResult_0, %addressResult = handshake.store [%afterValue] %23, %31 : index, i32 loc(#loc62)
    %24:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc46)
    %25 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_0, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc46)
    %26 = dataflow.carry %willContinue, %falseResult, %trueResult_1 : i1, none, none -> none loc(#loc51)
    %trueResult_1, %falseResult_2 = handshake.cond_br %willContinue, %24#1 : none loc(#loc51)
    %27 = handshake.constant %0 {value = 0 : index} : index loc(#loc51)
    %28 = handshake.constant %0 {value = 1 : index} : index loc(#loc51)
    %29 = arith.select %7, %28, %27 : index loc(#loc51)
    %30 = handshake.mux %29 [%falseResult_2, %trueResult] : index, none loc(#loc51)
    %31 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc51)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %25 : none loc(#loc51)
    %32 = handshake.mux %29 [%falseResult_4, %trueResult] : index, none loc(#loc51)
    %33 = handshake.join %30, %32 : none, none loc(#loc46)
    handshake.return %33 : none loc(#loc47)
  } loc(#loc46)
  func.func private @llvm.bswap.i32(i32) -> i32 loc(#loc2)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc30)
    %false = arith.constant false loc(#loc30)
    %0 = seq.const_clock  low loc(#loc30)
    %c2_i32 = arith.constant 2 : i32 loc(#loc30)
    %1 = ub.poison : i64 loc(#loc30)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c6 = arith.constant 6 : index loc(#loc31)
    %c5 = arith.constant 5 : index loc(#loc32)
    %c4 = arith.constant 4 : index loc(#loc33)
    %c3 = arith.constant 3 : index loc(#loc34)
    %c2 = arith.constant 2 : index loc(#loc35)
    %c1 = arith.constant 1 : index loc(#loc36)
    %c0 = arith.constant 0 : index loc(#loc2)
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %c305419896_i32 = arith.constant 305419896 : i32 loc(#loc2)
    %c287454020_i32 = arith.constant 287454020 : i32 loc(#loc2)
    %c-16777216_i32 = arith.constant -16777216 : i32 loc(#loc2)
    %c255_i32 = arith.constant 255 : i32 loc(#loc2)
    %c-1412567295_i32 = arith.constant -1412567295 : i32 loc(#loc2)
    %c7_i64 = arith.constant 7 : i64 loc(#loc2)
    %c16909060_i32 = arith.constant 16909060 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c256_i64 = arith.constant 256 : i64 loc(#loc2)
    %c256_i32 = arith.constant 256 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %2 = memref.get_global @str : memref<18xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<18xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<256xi32> loc(#loc2)
    memref.store %c0_i32, %alloca[%c0] : memref<256xi32> loc(#loc37)
    memref.store %c-1_i32, %alloca[%c1] : memref<256xi32> loc(#loc36)
    memref.store %c305419896_i32, %alloca[%c2] : memref<256xi32> loc(#loc35)
    memref.store %c287454020_i32, %alloca[%c3] : memref<256xi32> loc(#loc34)
    memref.store %c-16777216_i32, %alloca[%c4] : memref<256xi32> loc(#loc33)
    memref.store %c255_i32, %alloca[%c5] : memref<256xi32> loc(#loc32)
    memref.store %c-1412567295_i32, %alloca[%c6] : memref<256xi32> loc(#loc31)
    %4 = scf.while (%arg0 = %c7_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc52)
      %11 = arith.muli %10, %c16909060_i32 : i32 loc(#loc52)
      %12 = arith.index_cast %arg0 : i64 to index loc(#loc52)
      memref.store %11, %alloca[%12] : memref<256xi32> loc(#loc52)
      %13 = arith.addi %arg0, %c1_i64 : i64 loc(#loc48)
      %14 = arith.cmpi ne, %13, %c256_i64 : i64 loc(#loc53)
      scf.condition(%14) %13 : i64 loc(#loc42)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block8>[#loc20])):
      scf.yield %arg0 : i64 loc(#loc42)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc42)
    %cast = memref.cast %alloca : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %cast_2 = memref.cast %alloca_0 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    call @_Z13byte_swap_cpuPKjPjj(%cast, %cast_2, %c256_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc38)
    %cast_3 = memref.cast %alloca_1 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c256_i32, %true : i32 loc(#loc39)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc39)
    %5 = handshake.esi_instance @_Z13byte_swap_dsaPKjPjj_esi "_Z13byte_swap_dsaPKjPjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc39)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc39)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc57)
      %11 = memref.load %alloca_0[%10] : memref<256xi32> loc(#loc57)
      %12 = memref.load %alloca_1[%10] : memref<256xi32> loc(#loc57)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc57)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc49)
        %17 = arith.cmpi eq, %16, %c256_i64 : i64 loc(#loc49)
        %18 = arith.extui %17 : i1 to i32 loc(#loc43)
        %19 = arith.cmpi ne, %16, %c256_i64 : i64 loc(#loc54)
        %20 = arith.extui %19 : i1 to i32 loc(#loc43)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc57)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc57)
      } loc(#loc57)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc43)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc43)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc24]), %arg1: i1 loc(fused<#di_lexical_block9>[#loc24]), %arg2: i32 loc(fused<#di_lexical_block9>[#loc24])):
      scf.yield %arg0 : i64 loc(#loc43)
    } loc(#loc43)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc43)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc43)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<18xi8> -> index loc(#loc63)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc63)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc63)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc63)
      scf.yield %c1_i32 : i32 loc(#loc64)
    } loc(#loc43)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<18xi8> -> index loc(#loc40)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc40)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc40)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc40)
    } loc(#loc2)
    return %9 : i32 loc(#loc41)
  } loc(#loc30)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/byte_swap/byte_swap.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/byte_swap/byte_swap.cpp":19:0)
#loc5 = loc("tests/app/byte_swap/byte_swap.cpp":26:0)
#loc6 = loc("tests/app/byte_swap/byte_swap.cpp":28:0)
#loc8 = loc("tests/app/byte_swap/byte_swap.cpp":37:0)
#loc9 = loc("tests/app/byte_swap/byte_swap.cpp":38:0)
#loc10 = loc("tests/app/byte_swap/byte_swap.cpp":45:0)
#loc11 = loc("tests/app/byte_swap/byte_swap.cpp":47:0)
#loc12 = loc("tests/app/byte_swap/main.cpp":5:0)
#loc13 = loc("tests/app/byte_swap/main.cpp":16:0)
#loc14 = loc("tests/app/byte_swap/main.cpp":15:0)
#loc15 = loc("tests/app/byte_swap/main.cpp":14:0)
#loc16 = loc("tests/app/byte_swap/main.cpp":13:0)
#loc17 = loc("tests/app/byte_swap/main.cpp":12:0)
#loc18 = loc("tests/app/byte_swap/main.cpp":11:0)
#loc19 = loc("tests/app/byte_swap/main.cpp":10:0)
#loc21 = loc("tests/app/byte_swap/main.cpp":19:0)
#loc22 = loc("tests/app/byte_swap/main.cpp":27:0)
#loc23 = loc("tests/app/byte_swap/main.cpp":30:0)
#loc25 = loc("tests/app/byte_swap/main.cpp":34:0)
#loc26 = loc("tests/app/byte_swap/main.cpp":35:0)
#loc27 = loc("tests/app/byte_swap/main.cpp":36:0)
#loc28 = loc("tests/app/byte_swap/main.cpp":40:0)
#loc29 = loc("tests/app/byte_swap/main.cpp":42:0)
#loc30 = loc(fused<#di_subprogram3>[#loc12])
#loc31 = loc(fused<#di_subprogram3>[#loc13])
#loc32 = loc(fused<#di_subprogram3>[#loc14])
#loc33 = loc(fused<#di_subprogram3>[#loc15])
#loc34 = loc(fused<#di_subprogram3>[#loc16])
#loc35 = loc(fused<#di_subprogram3>[#loc17])
#loc36 = loc(fused<#di_subprogram3>[#loc18])
#loc37 = loc(fused<#di_subprogram3>[#loc19])
#loc38 = loc(fused<#di_subprogram3>[#loc22])
#loc39 = loc(fused<#di_subprogram3>[#loc23])
#loc40 = loc(fused<#di_subprogram3>[#loc28])
#loc41 = loc(fused<#di_subprogram3>[#loc29])
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 18>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file1, line = 33>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 37>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 18>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 33>
#loc45 = loc(fused<#di_subprogram4>[#loc6])
#loc47 = loc(fused<#di_subprogram5>[#loc11])
#loc48 = loc(fused<#di_lexical_block10>[#loc20])
#loc49 = loc(fused<#di_lexical_block11>[#loc24])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 18>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 37>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 34>
#loc51 = loc(fused<#di_lexical_block13>[#loc8])
#loc52 = loc(fused<#di_lexical_block14>[#loc21])
#loc53 = loc(fused[#loc42, #loc48])
#loc54 = loc(fused[#loc43, #loc49])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 18>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 37>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 34>
#loc55 = loc(fused<#di_lexical_block16>[#loc3])
#loc56 = loc(fused<#di_lexical_block17>[#loc8])
#loc57 = loc(fused<#di_lexical_block18>[#loc25])
#loc58 = loc(fused<#di_lexical_block19>[#loc4])
#loc59 = loc(fused<#di_lexical_block19>[#loc5])
#loc60 = loc(fused[#loc50, #loc55])
#loc61 = loc(fused<#di_lexical_block20>[#loc9])
#loc62 = loc(fused<#di_lexical_block20>[#loc10])
#loc63 = loc(fused<#di_lexical_block21>[#loc26])
#loc64 = loc(fused<#di_lexical_block21>[#loc27])
