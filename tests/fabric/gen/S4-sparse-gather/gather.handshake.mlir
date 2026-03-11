#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/gather/gather.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/gather/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/gather/gather.cpp":16:0)
#loc3 = loc("tests/app/gather/gather.cpp":21:0)
#loc9 = loc("tests/app/gather/gather.cpp":34:0)
#loc16 = loc("tests/app/gather/main.cpp":20:0)
#loc18 = loc("tests/app/gather/main.cpp":25:0)
#loc22 = loc("tests/app/gather/main.cpp":36:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 21>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 39>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 20>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 25>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 36>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 21>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 39>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 8192, elements = #llvm.di_subrange<count = 256 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 32768, elements = #llvm.di_subrange<count = 1024 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 21>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 39>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 21, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 39, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 20, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 25, type = #di_derived_type1>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file1, line = 36, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 19, arg = 4, type = #di_derived_type2>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "src_size", file = #di_file, line = 20, arg = 5, type = #di_derived_type2>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "idx", file = #di_file, line = 22, type = #di_derived_type1>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 37, arg = 4, type = #di_derived_type2>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram1, name = "src_size", file = #di_file, line = 38, arg = 5, type = #di_derived_type2>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "idx", file = #di_file, line = 40, type = #di_derived_type1>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 6, type = #di_derived_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram2, name = "src_size", file = #di_file1, line = 7, type = #di_derived_type2>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram2, name = "src", file = #di_file1, line = 10, type = #di_composite_type>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram2, name = "indices", file = #di_file1, line = 13, type = #di_composite_type1>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_dst", file = #di_file1, line = 16, type = #di_composite_type1>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_dst", file = #di_file1, line = 17, type = #di_composite_type1>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram, name = "dst", file = #di_file, line = 18, arg = 3, type = #di_derived_type5>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram1, name = "dst", file = #di_file, line = 36, arg = 3, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable11, #di_local_variable12, #di_local_variable13, #di_local_variable14, #di_local_variable15, #di_local_variable16, #di_local_variable2, #di_local_variable3, #di_local_variable4>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 20>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 25>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 36>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram, name = "src", file = #di_file, line = 16, arg = 1, type = #di_derived_type6>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram, name = "indices", file = #di_file, line = 17, arg = 2, type = #di_derived_type6>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram1, name = "src", file = #di_file, line = 34, arg = 1, type = #di_derived_type6>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram1, name = "indices", file = #di_file, line = 35, arg = 2, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type5, #di_derived_type2, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "gather_cpu", linkageName = "_Z10gather_cpuPKjS0_Pjjj", file = #di_file, line = 16, scopeLine = 20, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable19, #di_local_variable20, #di_local_variable17, #di_local_variable5, #di_local_variable6, #di_local_variable, #di_local_variable7>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "gather_dsa", linkageName = "_Z10gather_dsaPKjS0_Pjjj", file = #di_file, line = 34, scopeLine = 38, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable21, #di_local_variable22, #di_local_variable18, #di_local_variable8, #di_local_variable9, #di_local_variable1, #di_local_variable10>
#loc33 = loc(fused<#di_lexical_block9>[#loc16])
#loc34 = loc(fused<#di_lexical_block10>[#loc18])
#loc35 = loc(fused<#di_lexical_block11>[#loc22])
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 21>
#loc36 = loc(fused<#di_subprogram4>[#loc1])
#loc38 = loc(fused<#di_subprogram5>[#loc9])
#loc43 = loc(fused<#di_lexical_block15>[#loc3])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<19xi8> = dense<[108, 111, 111, 109, 46, 109, 101, 109, 111, 114, 121, 95, 98, 97, 110, 107, 61, 56, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<28xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 103, 97, 116, 104, 101, 114, 47, 103, 97, 116, 104, 101, 114, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<12xi8> = dense<[108, 111, 111, 109, 46, 115, 116, 114, 101, 97, 109, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @str : memref<15xi8> = dense<[103, 97, 116, 104, 101, 114, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<15xi8> = dense<[103, 97, 116, 104, 101, 114, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z10gather_cpuPKjS0_Pjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg3: i32 loc(fused<#di_subprogram4>[#loc1]), %arg4: i32 loc(fused<#di_subprogram4>[#loc1])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc50)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg3 : i32 to i64 loc(#loc50)
      %2 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg5 : i64 to index loc(#loc53)
        %4 = memref.load %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc53)
        %5 = arith.cmpi ult, %4, %arg4 : i32 loc(#loc58)
        %6 = scf.if %5 -> (i32) {
          %9 = arith.extui %4 : i32 to i64 loc(#loc62)
          %10 = arith.index_cast %9 : i64 to index loc(#loc62)
          %11 = memref.load %arg0[%10] : memref<?xi32, strided<[1], offset: ?>> loc(#loc62)
          scf.yield %11 : i32 loc(#loc63)
        } else {
          scf.yield %c0_i32 : i32 loc(#loc58)
        } loc(#loc58)
        memref.store %6, %arg2[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc59)
        %7 = arith.addi %arg5, %c1_i64 : i64 loc(#loc50)
        %8 = arith.cmpi ne, %7, %1 : i64 loc(#loc54)
        scf.condition(%8) %7 : i64 loc(#loc43)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block15>[#loc3])):
        scf.yield %arg5 : i64 loc(#loc43)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc43)
    } loc(#loc43)
    return loc(#loc37)
  } loc(#loc36)
  handshake.func @_Z10gather_dsaPKjS0_Pjjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg3: i32 loc(fused<#di_subprogram5>[#loc9]), %arg4: i32 loc(fused<#di_subprogram5>[#loc9]), %arg5: i1 loc(fused<#di_subprogram5>[#loc9]), ...) -> i1 attributes {argNames = ["src", "indices", "dst", "N", "src_size", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : i1 loc(#loc38)
    %1 = handshake.join %0 : none loc(#loc38)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc51)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc44)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc44)
    %6 = arith.index_cast %3 : i64 to index loc(#loc44)
    %7 = arith.index_cast %arg3 : i32 to index loc(#loc44)
    %index, %willContinue = dataflow.stream %6, %5, %7 {step_op = "+=", stop_cond = "!="} loc(#loc44)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc44)
    %8 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc44)
    %dataResult, %addressResults = handshake.load [%afterValue] %16#0, %19 : index, i32 loc(#loc55)
    %9 = arith.cmpi ult, %dataResult, %arg4 : i32 loc(#loc60)
    %10 = arith.extui %dataResult : i32 to i64 loc(#loc64)
    %11 = arith.index_cast %10 : i64 to index loc(#loc64)
    %dataResult_0, %addressResults_1 = handshake.load [%11] %17#0, %trueResult_7 : index, i32 loc(#loc64)
    %12 = handshake.constant %8 {value = 0 : index} : index loc(#loc60)
    %13 = handshake.constant %8 {value = 1 : index} : index loc(#loc60)
    %14 = arith.select %9, %13, %12 : index loc(#loc60)
    %15 = handshake.mux %14 [%2, %dataResult_0] : index, i32 loc(#loc60)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %15, %24 : index, i32 loc(#loc61)
    %16:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc38)
    %17:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (i32, none) loc(#loc38)
    %18 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc38)
    %19 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc44)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %16#1 : none loc(#loc44)
    %20 = handshake.constant %1 {value = 0 : index} : index loc(#loc44)
    %21 = handshake.constant %1 {value = 1 : index} : index loc(#loc44)
    %22 = arith.select %4, %21, %20 : index loc(#loc44)
    %23 = handshake.mux %22 [%falseResult_4, %trueResult] : index, none loc(#loc44)
    %24 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc44)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %18 : none loc(#loc44)
    %25 = handshake.mux %22 [%falseResult_6, %trueResult] : index, none loc(#loc44)
    %26 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc44)
    %trueResult_7, %falseResult_8 = handshake.cond_br %9, %26 : none loc(#loc60)
    %27 = handshake.constant %26 {value = 0 : index} : index loc(#loc60)
    %28 = handshake.constant %26 {value = 1 : index} : index loc(#loc60)
    %29 = arith.select %9, %28, %27 : index loc(#loc60)
    %30 = handshake.mux %29 [%falseResult_8, %17#1] : index, none loc(#loc60)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %30 : none loc(#loc44)
    %31 = handshake.mux %22 [%falseResult_10, %trueResult] : index, none loc(#loc44)
    %32 = handshake.join %23, %25, %31 : none, none, none loc(#loc38)
    %33 = handshake.constant %32 {value = true} : i1 loc(#loc38)
    handshake.return %33 : i1 loc(#loc38)
  } loc(#loc38)
  handshake.func @_Z10gather_dsaPKjS0_Pjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg3: i32 loc(fused<#di_subprogram5>[#loc9]), %arg4: i32 loc(fused<#di_subprogram5>[#loc9]), %arg5: none loc(fused<#di_subprogram5>[#loc9]), ...) -> none attributes {argNames = ["src", "indices", "dst", "N", "src_size", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : none loc(#loc38)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi eq, %arg3, %1 : i32 loc(#loc51)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc44)
    %4 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc44)
    %5 = arith.index_cast %2 : i64 to index loc(#loc44)
    %6 = arith.index_cast %arg3 : i32 to index loc(#loc44)
    %index, %willContinue = dataflow.stream %5, %4, %6 {step_op = "+=", stop_cond = "!="} loc(#loc44)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc44)
    %7 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc44)
    %dataResult, %addressResults = handshake.load [%afterValue] %15#0, %18 : index, i32 loc(#loc55)
    %8 = arith.cmpi ult, %dataResult, %arg4 : i32 loc(#loc60)
    %9 = arith.extui %dataResult : i32 to i64 loc(#loc64)
    %10 = arith.index_cast %9 : i64 to index loc(#loc64)
    %dataResult_0, %addressResults_1 = handshake.load [%10] %16#0, %trueResult_7 : index, i32 loc(#loc64)
    %11 = handshake.constant %7 {value = 0 : index} : index loc(#loc60)
    %12 = handshake.constant %7 {value = 1 : index} : index loc(#loc60)
    %13 = arith.select %8, %12, %11 : index loc(#loc60)
    %14 = handshake.mux %13 [%1, %dataResult_0] : index, i32 loc(#loc60)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %14, %23 : index, i32 loc(#loc61)
    %15:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc38)
    %16:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (i32, none) loc(#loc38)
    %17 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc38)
    %18 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc44)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %15#1 : none loc(#loc44)
    %19 = handshake.constant %0 {value = 0 : index} : index loc(#loc44)
    %20 = handshake.constant %0 {value = 1 : index} : index loc(#loc44)
    %21 = arith.select %3, %20, %19 : index loc(#loc44)
    %22 = handshake.mux %21 [%falseResult_4, %trueResult] : index, none loc(#loc44)
    %23 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc44)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %17 : none loc(#loc44)
    %24 = handshake.mux %21 [%falseResult_6, %trueResult] : index, none loc(#loc44)
    %25 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc44)
    %trueResult_7, %falseResult_8 = handshake.cond_br %8, %25 : none loc(#loc60)
    %26 = handshake.constant %25 {value = 0 : index} : index loc(#loc60)
    %27 = handshake.constant %25 {value = 1 : index} : index loc(#loc60)
    %28 = arith.select %8, %27, %26 : index loc(#loc60)
    %29 = handshake.mux %28 [%falseResult_8, %16#1] : index, none loc(#loc60)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %29 : none loc(#loc44)
    %30 = handshake.mux %21 [%falseResult_10, %trueResult] : index, none loc(#loc44)
    %31 = handshake.join %22, %24, %30 : none, none, none loc(#loc38)
    handshake.return %31 : none loc(#loc39)
  } loc(#loc38)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc28)
    %false = arith.constant false loc(#loc28)
    %0 = seq.const_clock  low loc(#loc28)
    %c2_i32 = arith.constant 2 : i32 loc(#loc28)
    %1 = ub.poison : i64 loc(#loc28)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c256_i64 = arith.constant 256 : i64 loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc2)
    %c255_i32 = arith.constant 255 : i32 loc(#loc2)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %c256_i32 = arith.constant 256 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<15xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<15xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc45)
      %12 = arith.trunci %arg0 : i64 to i32 loc(#loc45)
      %13 = arith.shli %12, %c1_i32 : i32 loc(#loc45)
      memref.store %13, %alloca[%11] : memref<256xi32> loc(#loc45)
      %14 = arith.addi %arg0, %c1_i64 : i64 loc(#loc40)
      %15 = arith.cmpi ne, %14, %c256_i64 : i64 loc(#loc46)
      scf.condition(%15) %14 : i64 loc(#loc33)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc16])):
      scf.yield %arg0 : i64 loc(#loc33)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc33)
    %5 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc47)
      %12 = arith.muli %11, %c3_i32 : i32 loc(#loc47)
      %13 = arith.andi %12, %c255_i32 : i32 loc(#loc47)
      %14 = arith.index_cast %arg0 : i64 to index loc(#loc47)
      memref.store %13, %alloca_0[%14] : memref<1024xi32> loc(#loc47)
      %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc41)
      %16 = arith.cmpi ne, %15, %c1024_i64 : i64 loc(#loc48)
      scf.condition(%16) %15 : i64 loc(#loc34)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block10>[#loc18])):
      scf.yield %arg0 : i64 loc(#loc34)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc34)
    %cast = memref.cast %alloca : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %cast_3 = memref.cast %alloca_0 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %cast_4 = memref.cast %alloca_1 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    call @_Z10gather_cpuPKjS0_Pjjj(%cast, %cast_3, %cast_4, %c1024_i32, %c256_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc29)
    %cast_5 = memref.cast %alloca_2 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c1024_i32, %true : i32 loc(#loc30)
    %chanOutput_12, %ready_13 = esi.wrap.vr %c256_i32, %true : i32 loc(#loc30)
    %chanOutput_14, %ready_15 = esi.wrap.vr %true, %true : i1 loc(#loc30)
    %6 = handshake.esi_instance @_Z10gather_dsaPKjS0_Pjjj_esi "_Z10gather_dsaPKjS0_Pjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12, %chanOutput_14) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc30)
    %rawOutput, %valid = esi.unwrap.vr %6, %true : i1 loc(#loc30)
    %7:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc52)
      %12 = memref.load %alloca_1[%11] : memref<1024xi32> loc(#loc52)
      %13 = memref.load %alloca_2[%11] : memref<1024xi32> loc(#loc52)
      %14 = arith.cmpi eq, %12, %13 : i32 loc(#loc52)
      %15:3 = scf.if %14 -> (i64, i32, i32) {
        %17 = arith.addi %arg0, %c1_i64 : i64 loc(#loc42)
        %18 = arith.cmpi eq, %17, %c1024_i64 : i64 loc(#loc42)
        %19 = arith.extui %18 : i1 to i32 loc(#loc35)
        %20 = arith.cmpi ne, %17, %c1024_i64 : i64 loc(#loc49)
        %21 = arith.extui %20 : i1 to i32 loc(#loc35)
        scf.yield %17, %19, %21 : i64, i32, i32 loc(#loc52)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc52)
      } loc(#loc52)
      %16 = arith.trunci %15#2 : i32 to i1 loc(#loc35)
      scf.condition(%16) %15#0, %14, %15#1 : i64, i1, i32 loc(#loc35)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block11>[#loc22]), %arg1: i1 loc(fused<#di_lexical_block11>[#loc22]), %arg2: i32 loc(fused<#di_lexical_block11>[#loc22])):
      scf.yield %arg0 : i64 loc(#loc35)
    } loc(#loc35)
    %8 = arith.index_castui %7#2 : i32 to index loc(#loc35)
    %9 = scf.index_switch %8 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc35)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<15xi8> -> index loc(#loc56)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc56)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc56)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc56)
      scf.yield %c1_i32 : i32 loc(#loc57)
    } loc(#loc35)
    %10 = arith.select %7#1, %c0_i32, %9 : i32 loc(#loc2)
    scf.if %7#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<15xi8> -> index loc(#loc31)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc31)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc31)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc31)
    } loc(#loc2)
    return %10 : i32 loc(#loc32)
  } loc(#loc28)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/gather/gather.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/gather/gather.cpp":22:0)
#loc5 = loc("tests/app/gather/gather.cpp":23:0)
#loc6 = loc("tests/app/gather/gather.cpp":24:0)
#loc7 = loc("tests/app/gather/gather.cpp":25:0)
#loc8 = loc("tests/app/gather/gather.cpp":29:0)
#loc10 = loc("tests/app/gather/gather.cpp":39:0)
#loc11 = loc("tests/app/gather/gather.cpp":40:0)
#loc12 = loc("tests/app/gather/gather.cpp":41:0)
#loc13 = loc("tests/app/gather/gather.cpp":42:0)
#loc14 = loc("tests/app/gather/gather.cpp":47:0)
#loc15 = loc("tests/app/gather/main.cpp":5:0)
#loc17 = loc("tests/app/gather/main.cpp":21:0)
#loc19 = loc("tests/app/gather/main.cpp":26:0)
#loc20 = loc("tests/app/gather/main.cpp":30:0)
#loc21 = loc("tests/app/gather/main.cpp":33:0)
#loc23 = loc("tests/app/gather/main.cpp":37:0)
#loc24 = loc("tests/app/gather/main.cpp":38:0)
#loc25 = loc("tests/app/gather/main.cpp":39:0)
#loc26 = loc("tests/app/gather/main.cpp":43:0)
#loc27 = loc("tests/app/gather/main.cpp":45:0)
#loc28 = loc(fused<#di_subprogram3>[#loc15])
#loc29 = loc(fused<#di_subprogram3>[#loc20])
#loc30 = loc(fused<#di_subprogram3>[#loc21])
#loc31 = loc(fused<#di_subprogram3>[#loc26])
#loc32 = loc(fused<#di_subprogram3>[#loc27])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file1, line = 20>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 25>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 36>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 39>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file1, line = 20>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file1, line = 25>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 36>
#loc37 = loc(fused<#di_subprogram4>[#loc8])
#loc39 = loc(fused<#di_subprogram5>[#loc14])
#loc40 = loc(fused<#di_lexical_block12>[#loc16])
#loc41 = loc(fused<#di_lexical_block13>[#loc18])
#loc42 = loc(fused<#di_lexical_block14>[#loc22])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file, line = 21>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 39>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 37>
#loc44 = loc(fused<#di_lexical_block16>[#loc10])
#loc45 = loc(fused<#di_lexical_block17>[#loc17])
#loc46 = loc(fused[#loc33, #loc40])
#loc47 = loc(fused<#di_lexical_block18>[#loc19])
#loc48 = loc(fused[#loc34, #loc41])
#loc49 = loc(fused[#loc35, #loc42])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 21>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file, line = 39>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 37>
#loc50 = loc(fused<#di_lexical_block20>[#loc3])
#loc51 = loc(fused<#di_lexical_block21>[#loc10])
#loc52 = loc(fused<#di_lexical_block22>[#loc23])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file, line = 23>
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file, line = 41>
#loc53 = loc(fused<#di_lexical_block23>[#loc4])
#loc54 = loc(fused[#loc43, #loc50])
#loc55 = loc(fused<#di_lexical_block24>[#loc11])
#loc56 = loc(fused<#di_lexical_block25>[#loc24])
#loc57 = loc(fused<#di_lexical_block25>[#loc25])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file, line = 23>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file, line = 41>
#loc58 = loc(fused<#di_lexical_block26>[#loc5])
#loc59 = loc(fused<#di_lexical_block26>[#loc])
#loc60 = loc(fused<#di_lexical_block27>[#loc12])
#loc61 = loc(fused<#di_lexical_block27>[#loc])
#loc62 = loc(fused<#di_lexical_block28>[#loc6])
#loc63 = loc(fused<#di_lexical_block28>[#loc7])
#loc64 = loc(fused<#di_lexical_block29>[#loc13])
