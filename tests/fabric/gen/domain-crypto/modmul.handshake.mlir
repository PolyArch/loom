#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned long", sizeInBits = 64, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/modmul/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/modmul/modmul.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/modmul/main.cpp":18:0)
#loc8 = loc("tests/app/modmul/main.cpp":30:0)
#loc14 = loc("tests/app/modmul/modmul.cpp":27:0)
#loc21 = loc("tests/app/modmul/modmul.cpp":12:0)
#loc22 = loc("tests/app/modmul/modmul.cpp":17:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint64_t", baseType = #di_basic_type2>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 18>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 30>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 34>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 17>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint64_t", baseType = #di_derived_type1>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block2, file = #di_file1, line = 34>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file1, line = 17>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type2, sizeInBits = 32768, elements = #llvm.di_subrange<count = 1024 : i64>>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type2>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 34>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 17>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 18, type = #di_derived_type2>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 30, type = #di_derived_type2>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 34, type = #di_derived_type2>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 17, type = #di_derived_type2>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type4, sizeInBits = 64>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type5>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 6, type = #di_derived_type4>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "modulus", file = #di_file, line = 7, type = #di_derived_type4>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_A", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_B", file = #di_file, line = 11, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_output", file = #di_file, line = 14, type = #di_composite_type>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_output", file = #di_file, line = 15, type = #di_composite_type>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram1, name = "modulus", file = #di_file1, line = 30, arg = 4, type = #di_derived_type4>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 31, arg = 5, type = #di_derived_type4>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "a", file = #di_file1, line = 35, type = #di_derived_type3>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "b", file = #di_file1, line = 36, type = #di_derived_type3>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "result", file = #di_file1, line = 37, type = #di_derived_type3>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "modulus", file = #di_file1, line = 15, arg = 4, type = #di_derived_type4>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 16, arg = 5, type = #di_derived_type4>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "a", file = #di_file1, line = 18, type = #di_derived_type3>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "b", file = #di_file1, line = 19, type = #di_derived_type3>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "result", file = #di_file1, line = 20, type = #di_derived_type3>
#di_derived_type8 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type6>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_C", file = #di_file1, line = 29, arg = 3, type = #di_derived_type7>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_C", file = #di_file1, line = 14, arg = 3, type = #di_derived_type7>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable4, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable9, #di_local_variable, #di_local_variable1>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 18>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 30>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_A", file = #di_file1, line = 27, arg = 1, type = #di_derived_type8>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_B", file = #di_file1, line = 28, arg = 2, type = #di_derived_type8>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_A", file = #di_file1, line = 12, arg = 1, type = #di_derived_type8>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_B", file = #di_file1, line = 13, arg = 2, type = #di_derived_type8>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type8, #di_derived_type8, #di_derived_type7, #di_derived_type4, #di_derived_type4>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "modmul_dsa", linkageName = "_Z10modmul_dsaPKjS0_Pjjj", file = #di_file1, line = 27, scopeLine = 31, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable22, #di_local_variable23, #di_local_variable20, #di_local_variable10, #di_local_variable11, #di_local_variable2, #di_local_variable12, #di_local_variable13, #di_local_variable14>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "modmul_cpu", linkageName = "_Z10modmul_cpuPKjS0_Pjjj", file = #di_file1, line = 12, scopeLine = 16, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable24, #di_local_variable25, #di_local_variable21, #di_local_variable15, #di_local_variable16, #di_local_variable3, #di_local_variable17, #di_local_variable18, #di_local_variable19>
#loc33 = loc(fused<#di_lexical_block8>[#loc3])
#loc34 = loc(fused<#di_lexical_block9>[#loc8])
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 17>
#loc37 = loc(fused<#di_subprogram4>[#loc14])
#loc39 = loc(fused<#di_subprogram5>[#loc21])
#loc46 = loc(fused<#di_lexical_block15>[#loc22])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<15xi8> = dense<[109, 111, 100, 109, 117, 108, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<15xi8> = dense<[109, 111, 100, 109, 117, 108, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<28xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 109, 111, 100, 109, 117, 108, 47, 109, 111, 100, 109, 117, 108, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
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
    %c12345_i32 = arith.constant 12345 : i32 loc(#loc2)
    %c67890_i32 = arith.constant 67890 : i32 loc(#loc2)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc2)
    %c1000000007_i32 = arith.constant 1000000007 : i32 loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<15xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<15xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.addi %arg0, %c1_i64 : i64 loc(#loc41)
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc41)
      %12 = arith.trunci %10 : i64 to i32 loc(#loc41)
      %13 = arith.muli %12, %c12345_i32 : i32 loc(#loc41)
      memref.store %13, %alloca[%11] : memref<1024xi32> loc(#loc41)
      %14 = arith.muli %12, %c67890_i32 : i32 loc(#loc42)
      memref.store %14, %alloca_0[%11] : memref<1024xi32> loc(#loc42)
      %15 = arith.cmpi ne, %10, %c1024_i64 : i64 loc(#loc43)
      scf.condition(%15) %10 : i64 loc(#loc33)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block8>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc33)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc33)
    %cast = memref.cast %alloca : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %cast_3 = memref.cast %alloca_0 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %cast_4 = memref.cast %alloca_1 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    call @_Z10modmul_cpuPKjS0_Pjjj(%cast, %cast_3, %cast_4, %c1000000007_i32, %c1024_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc29)
    %cast_5 = memref.cast %alloca_2 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c1000000007_i32, %true : i32 loc(#loc30)
    %chanOutput_12, %ready_13 = esi.wrap.vr %c1024_i32, %true : i32 loc(#loc30)
    %chanOutput_14, %ready_15 = esi.wrap.vr %true, %true : i1 loc(#loc30)
    %5 = handshake.esi_instance @_Z10modmul_dsaPKjS0_Pjjj_esi "_Z10modmul_dsaPKjS0_Pjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12, %chanOutput_14) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc30)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc30)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc47)
      %11 = memref.load %alloca_1[%10] : memref<1024xi32> loc(#loc47)
      %12 = memref.load %alloca_2[%10] : memref<1024xi32> loc(#loc47)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc47)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc36)
        %17 = arith.cmpi eq, %16, %c1024_i64 : i64 loc(#loc36)
        %18 = arith.extui %17 : i1 to i32 loc(#loc34)
        %19 = arith.cmpi ne, %16, %c1024_i64 : i64 loc(#loc44)
        %20 = arith.extui %19 : i1 to i32 loc(#loc34)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc47)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc47)
      } loc(#loc47)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc34)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc34)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc8]), %arg1: i1 loc(fused<#di_lexical_block9>[#loc8]), %arg2: i32 loc(fused<#di_lexical_block9>[#loc8])):
      scf.yield %arg0 : i64 loc(#loc34)
    } loc(#loc34)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc34)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc34)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<15xi8> -> index loc(#loc50)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc50)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc50)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc50)
      scf.yield %c1_i32 : i32 loc(#loc51)
    } loc(#loc34)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<15xi8> -> index loc(#loc31)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc31)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc31)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc31)
    } loc(#loc2)
    return %9 : i32 loc(#loc32)
  } loc(#loc28)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z10modmul_dsaPKjS0_Pjjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg3: i32 loc(fused<#di_subprogram4>[#loc14]), %arg4: i32 loc(fused<#di_subprogram4>[#loc14]), %arg5: i1 loc(fused<#di_subprogram4>[#loc14]), ...) -> i1 attributes {argNames = ["input_A", "input_B", "output_C", "modulus", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : i1 loc(#loc37)
    %1 = handshake.join %0 : none loc(#loc37)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg4, %2 : i32 loc(#loc48)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc45)
    %5 = arith.extui %arg3 : i32 to i64 loc(#loc2)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc45)
    %7 = arith.index_cast %3 : i64 to index loc(#loc45)
    %8 = arith.index_cast %arg4 : i32 to index loc(#loc45)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.no_parallel", "loom.loop.no_unroll"], step_op = "+=", stop_cond = "!="} loc(#loc45)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc45)
    %dataResult, %addressResults = handshake.load [%afterValue] %15#0, %18 : index, i32 loc(#loc52)
    %9 = arith.extui %dataResult : i32 to i64 loc(#loc52)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %16#0, %25 : index, i32 loc(#loc53)
    %10 = arith.extui %dataResult_0 : i32 to i64 loc(#loc53)
    %11 = arith.muli %10, %9 : i64 loc(#loc54)
    %12 = dataflow.invariant %afterCond, %5 : i1, i64 -> i64 loc(#loc54)
    %13 = arith.remui %11, %12 : i64 loc(#loc54)
    %14 = arith.trunci %13 : i64 to i32 loc(#loc55)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %14, %23 : index, i32 loc(#loc55)
    %15:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc37)
    %16:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (i32, none) loc(#loc37)
    %17 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc37)
    %18 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc45)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %15#1 : none loc(#loc45)
    %19 = handshake.constant %1 {value = 0 : index} : index loc(#loc45)
    %20 = handshake.constant %1 {value = 1 : index} : index loc(#loc45)
    %21 = arith.select %4, %20, %19 : index loc(#loc45)
    %22 = handshake.mux %21 [%falseResult_4, %trueResult] : index, none loc(#loc45)
    %23 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc45)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %17 : none loc(#loc45)
    %24 = handshake.mux %21 [%falseResult_6, %trueResult] : index, none loc(#loc45)
    %25 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc45)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %16#1 : none loc(#loc45)
    %26 = handshake.mux %21 [%falseResult_8, %trueResult] : index, none loc(#loc45)
    %27 = handshake.join %22, %24, %26 : none, none, none loc(#loc37)
    %28 = handshake.constant %27 {value = true} : i1 loc(#loc37)
    handshake.return %28 : i1 loc(#loc37)
  } loc(#loc37)
  handshake.func @_Z10modmul_dsaPKjS0_Pjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg3: i32 loc(fused<#di_subprogram4>[#loc14]), %arg4: i32 loc(fused<#di_subprogram4>[#loc14]), %arg5: none loc(fused<#di_subprogram4>[#loc14]), ...) -> none attributes {argNames = ["input_A", "input_B", "output_C", "modulus", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : none loc(#loc37)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi eq, %arg4, %1 : i32 loc(#loc48)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc45)
    %4 = arith.extui %arg3 : i32 to i64 loc(#loc2)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc45)
    %6 = arith.index_cast %2 : i64 to index loc(#loc45)
    %7 = arith.index_cast %arg4 : i32 to index loc(#loc45)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.no_parallel", "loom.loop.no_unroll"], step_op = "+=", stop_cond = "!="} loc(#loc45)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc45)
    %dataResult, %addressResults = handshake.load [%afterValue] %14#0, %17 : index, i32 loc(#loc52)
    %8 = arith.extui %dataResult : i32 to i64 loc(#loc52)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %15#0, %24 : index, i32 loc(#loc53)
    %9 = arith.extui %dataResult_0 : i32 to i64 loc(#loc53)
    %10 = arith.muli %9, %8 : i64 loc(#loc54)
    %11 = dataflow.invariant %afterCond, %4 : i1, i64 -> i64 loc(#loc54)
    %12 = arith.remui %10, %11 : i64 loc(#loc54)
    %13 = arith.trunci %12 : i64 to i32 loc(#loc55)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %13, %22 : index, i32 loc(#loc55)
    %14:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc37)
    %15:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (i32, none) loc(#loc37)
    %16 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc37)
    %17 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc45)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %14#1 : none loc(#loc45)
    %18 = handshake.constant %0 {value = 0 : index} : index loc(#loc45)
    %19 = handshake.constant %0 {value = 1 : index} : index loc(#loc45)
    %20 = arith.select %3, %19, %18 : index loc(#loc45)
    %21 = handshake.mux %20 [%falseResult_4, %trueResult] : index, none loc(#loc45)
    %22 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc45)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %16 : none loc(#loc45)
    %23 = handshake.mux %20 [%falseResult_6, %trueResult] : index, none loc(#loc45)
    %24 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc45)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %15#1 : none loc(#loc45)
    %25 = handshake.mux %20 [%falseResult_8, %trueResult] : index, none loc(#loc45)
    %26 = handshake.join %21, %23, %25 : none, none, none loc(#loc37)
    handshake.return %26 : none loc(#loc38)
  } loc(#loc37)
  func.func @_Z10modmul_cpuPKjS0_Pjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc21]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc21]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc21]), %arg3: i32 loc(fused<#di_subprogram5>[#loc21]), %arg4: i32 loc(fused<#di_subprogram5>[#loc21])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg4, %c0_i32 : i32 loc(#loc49)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg3 : i32 to i64 loc(#loc2)
      %2 = arith.extui %arg4 : i32 to i64 loc(#loc49)
      %3 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        %4 = arith.index_cast %arg5 : i64 to index loc(#loc56)
        %5 = memref.load %arg0[%4] : memref<?xi32, strided<[1], offset: ?>> loc(#loc56)
        %6 = arith.extui %5 : i32 to i64 loc(#loc56)
        %7 = memref.load %arg1[%4] : memref<?xi32, strided<[1], offset: ?>> loc(#loc57)
        %8 = arith.extui %7 : i32 to i64 loc(#loc57)
        %9 = arith.muli %8, %6 : i64 loc(#loc58)
        %10 = arith.remui %9, %1 : i64 loc(#loc58)
        %11 = arith.trunci %10 : i64 to i32 loc(#loc59)
        memref.store %11, %arg2[%4] : memref<?xi32, strided<[1], offset: ?>> loc(#loc59)
        %12 = arith.addi %arg5, %c1_i64 : i64 loc(#loc49)
        %13 = arith.cmpi ne, %12, %2 : i64 loc(#loc60)
        scf.condition(%13) %12 : i64 loc(#loc46)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block15>[#loc22])):
        scf.yield %arg5 : i64 loc(#loc46)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc46)
    } loc(#loc46)
    return loc(#loc40)
  } loc(#loc39)
} loc(#loc)
#loc = loc("tests/app/modmul/main.cpp":0:0)
#loc1 = loc("tests/app/modmul/main.cpp":5:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/modmul/main.cpp":19:0)
#loc5 = loc("tests/app/modmul/main.cpp":20:0)
#loc6 = loc("tests/app/modmul/main.cpp":24:0)
#loc7 = loc("tests/app/modmul/main.cpp":27:0)
#loc9 = loc("tests/app/modmul/main.cpp":31:0)
#loc10 = loc("tests/app/modmul/main.cpp":32:0)
#loc11 = loc("tests/app/modmul/main.cpp":33:0)
#loc12 = loc("tests/app/modmul/main.cpp":37:0)
#loc13 = loc("tests/app/modmul/main.cpp":39:0)
#loc15 = loc("tests/app/modmul/modmul.cpp":34:0)
#loc16 = loc("tests/app/modmul/modmul.cpp":35:0)
#loc17 = loc("tests/app/modmul/modmul.cpp":36:0)
#loc18 = loc("tests/app/modmul/modmul.cpp":37:0)
#loc19 = loc("tests/app/modmul/modmul.cpp":38:0)
#loc20 = loc("tests/app/modmul/modmul.cpp":40:0)
#loc23 = loc("tests/app/modmul/modmul.cpp":18:0)
#loc24 = loc("tests/app/modmul/modmul.cpp":19:0)
#loc25 = loc("tests/app/modmul/modmul.cpp":20:0)
#loc26 = loc("tests/app/modmul/modmul.cpp":21:0)
#loc27 = loc("tests/app/modmul/modmul.cpp":23:0)
#loc28 = loc(fused<#di_subprogram3>[#loc1])
#loc29 = loc(fused<#di_subprogram3>[#loc6])
#loc30 = loc(fused<#di_subprogram3>[#loc7])
#loc31 = loc(fused<#di_subprogram3>[#loc12])
#loc32 = loc(fused<#di_subprogram3>[#loc13])
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 18>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 30>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 18>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 30>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 34>
#loc35 = loc(fused<#di_lexical_block10>[#loc3])
#loc36 = loc(fused<#di_lexical_block11>[#loc8])
#loc38 = loc(fused<#di_subprogram4>[#loc20])
#loc40 = loc(fused<#di_subprogram5>[#loc27])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 31>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 34>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 17>
#loc41 = loc(fused<#di_lexical_block12>[#loc4])
#loc42 = loc(fused<#di_lexical_block12>[#loc5])
#loc43 = loc(fused[#loc33, #loc35])
#loc44 = loc(fused[#loc34, #loc36])
#loc45 = loc(fused<#di_lexical_block14>[#loc15])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 31>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 34>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 17>
#loc47 = loc(fused<#di_lexical_block16>[#loc9])
#loc48 = loc(fused<#di_lexical_block17>[#loc15])
#loc49 = loc(fused<#di_lexical_block18>[#loc22])
#loc50 = loc(fused<#di_lexical_block19>[#loc10])
#loc51 = loc(fused<#di_lexical_block19>[#loc11])
#loc52 = loc(fused<#di_lexical_block20>[#loc16])
#loc53 = loc(fused<#di_lexical_block20>[#loc17])
#loc54 = loc(fused<#di_lexical_block20>[#loc18])
#loc55 = loc(fused<#di_lexical_block20>[#loc19])
#loc56 = loc(fused<#di_lexical_block21>[#loc23])
#loc57 = loc(fused<#di_lexical_block21>[#loc24])
#loc58 = loc(fused<#di_lexical_block21>[#loc25])
#loc59 = loc(fused<#di_lexical_block21>[#loc26])
#loc60 = loc(fused[#loc46, #loc49])
