#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned long", sizeInBits = 64, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/modexp/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/modexp/modexp.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/modexp/main.cpp":18:0)
#loc8 = loc("tests/app/modexp/main.cpp":30:0)
#loc14 = loc("tests/app/modexp/modexp.cpp":36:0)
#loc24 = loc("tests/app/modexp/modexp.cpp":12:0)
#loc25 = loc("tests/app/modexp/modexp.cpp":17:0)
#loc27 = loc("tests/app/modexp/modexp.cpp":22:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint64_t", baseType = #di_basic_type2>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 18>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 30>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 43>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 17>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint64_t", baseType = #di_derived_type1>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block2, file = #di_file1, line = 43>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file1, line = 17>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type2, sizeInBits = 8192, elements = #llvm.di_subrange<count = 256 : i64>>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type2>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 43>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 17>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 18, type = #di_derived_type2>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 30, type = #di_derived_type2>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 43, type = #di_derived_type2>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 17, type = #di_derived_type2>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type4, sizeInBits = 64>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type5>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 6, type = #di_derived_type4>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "modulus", file = #di_file, line = 7, type = #di_derived_type4>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_base", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_exp", file = #di_file, line = 11, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_output", file = #di_file, line = 14, type = #di_composite_type>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_output", file = #di_file, line = 15, type = #di_composite_type>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram1, name = "modulus", file = #di_file1, line = 39, arg = 4, type = #di_derived_type4>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 40, arg = 5, type = #di_derived_type4>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "result", file = #di_file1, line = 44, type = #di_derived_type3>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "base", file = #di_file1, line = 45, type = #di_derived_type3>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "exp", file = #di_file1, line = 46, type = #di_derived_type2>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "modulus", file = #di_file1, line = 15, arg = 4, type = #di_derived_type4>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 16, arg = 5, type = #di_derived_type4>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "result", file = #di_file1, line = 18, type = #di_derived_type3>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "base", file = #di_file1, line = 19, type = #di_derived_type3>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "exp", file = #di_file1, line = 20, type = #di_derived_type2>
#di_derived_type8 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type6>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_result", file = #di_file1, line = 38, arg = 3, type = #di_derived_type7>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_result", file = #di_file1, line = 14, arg = 3, type = #di_derived_type7>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable4, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable9, #di_local_variable, #di_local_variable1>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 18>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 30>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_base", file = #di_file1, line = 36, arg = 1, type = #di_derived_type8>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_exp", file = #di_file1, line = 37, arg = 2, type = #di_derived_type8>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_base", file = #di_file1, line = 12, arg = 1, type = #di_derived_type8>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_exp", file = #di_file1, line = 13, arg = 2, type = #di_derived_type8>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type8, #di_derived_type8, #di_derived_type7, #di_derived_type4, #di_derived_type4>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "modexp_dsa", linkageName = "_Z10modexp_dsaPKjS0_Pjjj", file = #di_file1, line = 36, scopeLine = 40, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable22, #di_local_variable23, #di_local_variable20, #di_local_variable10, #di_local_variable11, #di_local_variable2, #di_local_variable12, #di_local_variable13, #di_local_variable14>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "modexp_cpu", linkageName = "_Z10modexp_cpuPKjS0_Pjjj", file = #di_file1, line = 12, scopeLine = 16, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable24, #di_local_variable25, #di_local_variable21, #di_local_variable15, #di_local_variable16, #di_local_variable3, #di_local_variable17, #di_local_variable18, #di_local_variable19>
#loc41 = loc(fused<#di_lexical_block8>[#loc3])
#loc42 = loc(fused<#di_lexical_block9>[#loc8])
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 17>
#loc45 = loc(fused<#di_subprogram4>[#loc14])
#loc47 = loc(fused<#di_subprogram5>[#loc24])
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 17>
#loc54 = loc(fused<#di_lexical_block15>[#loc25])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 17>
#loc65 = loc(fused<#di_lexical_block21>[#loc27])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<15xi8> = dense<[109, 111, 100, 101, 120, 112, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<15xi8> = dense<[109, 111, 100, 101, 120, 112, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<28xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 109, 111, 100, 101, 120, 112, 47, 109, 111, 100, 101, 120, 112, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc36)
    %false = arith.constant false loc(#loc36)
    %0 = seq.const_clock  low loc(#loc36)
    %c2_i32 = arith.constant 2 : i32 loc(#loc36)
    %1 = ub.poison : i64 loc(#loc36)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c123_i32 = arith.constant 123 : i32 loc(#loc2)
    %c7_i32 = arith.constant 7 : i32 loc(#loc2)
    %c256_i64 = arith.constant 256 : i64 loc(#loc2)
    %c1000000007_i32 = arith.constant 1000000007 : i32 loc(#loc2)
    %c256_i32 = arith.constant 256 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<15xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<15xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<256xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.addi %arg0, %c1_i64 : i64 loc(#loc49)
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc49)
      %12 = arith.trunci %10 : i64 to i32 loc(#loc49)
      %13 = arith.muli %12, %c123_i32 : i32 loc(#loc49)
      memref.store %13, %alloca[%11] : memref<256xi32> loc(#loc49)
      %14 = arith.muli %12, %c7_i32 : i32 loc(#loc50)
      memref.store %14, %alloca_0[%11] : memref<256xi32> loc(#loc50)
      %15 = arith.cmpi ne, %10, %c256_i64 : i64 loc(#loc51)
      scf.condition(%15) %10 : i64 loc(#loc41)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block8>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc41)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc41)
    %cast = memref.cast %alloca : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc37)
    %cast_3 = memref.cast %alloca_0 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc37)
    %cast_4 = memref.cast %alloca_1 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc37)
    call @_Z10modexp_cpuPKjS0_Pjjj(%cast, %cast_3, %cast_4, %c1000000007_i32, %c256_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc37)
    %cast_5 = memref.cast %alloca_2 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c1000000007_i32, %true : i32 loc(#loc38)
    %chanOutput_12, %ready_13 = esi.wrap.vr %c256_i32, %true : i32 loc(#loc38)
    %chanOutput_14, %ready_15 = esi.wrap.vr %true, %true : i1 loc(#loc38)
    %5 = handshake.esi_instance @_Z10modexp_dsaPKjS0_Pjjj_esi "_Z10modexp_dsaPKjS0_Pjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12, %chanOutput_14) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc38)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc38)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc55)
      %11 = memref.load %alloca_1[%10] : memref<256xi32> loc(#loc55)
      %12 = memref.load %alloca_2[%10] : memref<256xi32> loc(#loc55)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc55)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc44)
        %17 = arith.cmpi eq, %16, %c256_i64 : i64 loc(#loc44)
        %18 = arith.extui %17 : i1 to i32 loc(#loc42)
        %19 = arith.cmpi ne, %16, %c256_i64 : i64 loc(#loc52)
        %20 = arith.extui %19 : i1 to i32 loc(#loc42)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc55)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc55)
      } loc(#loc55)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc42)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc42)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc8]), %arg1: i1 loc(fused<#di_lexical_block9>[#loc8]), %arg2: i32 loc(fused<#di_lexical_block9>[#loc8])):
      scf.yield %arg0 : i64 loc(#loc42)
    } loc(#loc42)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc42)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc42)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<15xi8> -> index loc(#loc58)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc58)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc58)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc58)
      scf.yield %c1_i32 : i32 loc(#loc59)
    } loc(#loc42)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<15xi8> -> index loc(#loc39)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc39)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc39)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc39)
    } loc(#loc2)
    return %9 : i32 loc(#loc40)
  } loc(#loc36)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z10modexp_dsaPKjS0_Pjjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg3: i32 loc(fused<#di_subprogram4>[#loc14]), %arg4: i32 loc(fused<#di_subprogram4>[#loc14]), %arg5: i1 loc(fused<#di_subprogram4>[#loc14]), ...) -> i1 attributes {argNames = ["input_base", "input_exp", "output_result", "modulus", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : i1 loc(#loc45)
    %1 = handshake.join %0 : none loc(#loc45)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %1 {value = 1 : i64} : i64 loc(#loc2)
    %6 = arith.cmpi eq, %arg4, %3 : i32 loc(#loc56)
    %trueResult, %falseResult = handshake.cond_br %6, %1 : none loc(#loc53)
    %7 = arith.extui %arg3 : i32 to i64 loc(#loc2)
    %8 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc53)
    %9 = arith.index_cast %4 : i64 to index loc(#loc53)
    %10 = arith.index_cast %arg4 : i32 to index loc(#loc53)
    %index, %willContinue = dataflow.stream %9, %8, %10 {loom.annotations = ["loom.loop.no_parallel", "loom.loop.no_unroll"], step_op = "+=", stop_cond = "!="} loc(#loc53)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc53)
    %11 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc53)
    %dataResult, %addressResults = handshake.load [%afterValue] %39#0, %41 : index, i32 loc(#loc60)
    %12 = arith.cmpi eq, %dataResult, %3 : i32 loc(#loc61)
    %trueResult_0, %falseResult_1 = handshake.cond_br %12, %11 : none loc(#loc61)
    %dataResult_2, %addressResults_3 = handshake.load [%afterValue] %38#0, %falseResult_20 : index, i32 loc(#loc62)
    %13 = arith.remui %dataResult_2, %arg3 : i32 loc(#loc62)
    %14 = arith.extui %13 : i32 to i64 loc(#loc62)
    %15 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc61)
    %16 = arith.index_cast %dataResult : i32 to index loc(#loc61)
    %17 = arith.index_cast %3 : i32 to index loc(#loc61)
    %index_4, %willContinue_5 = dataflow.stream %16, %15, %17 {step_op = ">>=", stop_cond = "!="} loc(#loc61)
    %afterValue_6, %afterCond_7 = dataflow.gate %index_4, %willContinue_5 : index, i1 -> index, i1 loc(#loc61)
    %18 = dataflow.carry %willContinue_5, %14, %32 : i1, i64, i64 -> i64 loc(#loc61)
    %afterValue_8, %afterCond_9 = dataflow.gate %18, %willContinue_5 : i64, i1 -> i64, i1 loc(#loc61)
    handshake.sink %afterCond_9 : i1 loc(#loc61)
    %19 = dataflow.carry %willContinue_5, %5, %30 : i1, i64, i64 -> i64 loc(#loc61)
    %afterValue_10, %afterCond_11 = dataflow.gate %19, %willContinue_5 : i64, i1 -> i64, i1 loc(#loc61)
    handshake.sink %afterCond_11 : i1 loc(#loc61)
    %trueResult_12, %falseResult_13 = handshake.cond_br %willContinue_5, %19 : i64 loc(#loc61)
    %20 = dataflow.invariant %afterCond_7, %falseResult_1 : i1, none -> none loc(#loc61)
    %21 = arith.index_cast %afterValue_6 : index to i32 loc(#loc61)
    %22 = arith.andi %21, %2 : i32 loc(#loc72)
    %23 = arith.cmpi eq, %22, %3 : i32 loc(#loc72)
    %24 = arith.muli %afterValue_8, %afterValue_10 : i64 loc(#loc74)
    %25 = dataflow.invariant %afterCond, %7 : i1, i64 -> i64 loc(#loc74)
    %26 = arith.remui %24, %25 : i64 loc(#loc74)
    %27 = handshake.constant %20 {value = 0 : index} : index loc(#loc72)
    %28 = handshake.constant %20 {value = 1 : index} : index loc(#loc72)
    %29 = arith.select %23, %28, %27 : index loc(#loc72)
    %30 = handshake.mux %29 [%26, %afterValue_10] : index, i64 loc(#loc72)
    %31 = arith.muli %afterValue_8, %afterValue_8 : i64 loc(#loc69)
    %32 = arith.remui %31, %25 : i64 loc(#loc69)
    %33 = arith.trunci %falseResult_13 : i64 to i32 loc(#loc63)
    %34 = handshake.constant %11 {value = 0 : index} : index loc(#loc61)
    %35 = handshake.constant %11 {value = 1 : index} : index loc(#loc61)
    %36 = arith.select %12, %35, %34 : index loc(#loc61)
    %37 = handshake.mux %36 [%33, %2] : index, i32 loc(#loc61)
    %dataResult_14, %addressResult = handshake.store [%afterValue] %37, %46 : index, i32 loc(#loc63)
    %38:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_3) {id = 0 : i32} : (index) -> (i32, none) loc(#loc45)
    %39:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc45)
    %40 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_14, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc45)
    %41 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc53)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %39#1 : none loc(#loc53)
    %42 = handshake.constant %1 {value = 0 : index} : index loc(#loc53)
    %43 = handshake.constant %1 {value = 1 : index} : index loc(#loc53)
    %44 = arith.select %6, %43, %42 : index loc(#loc53)
    %45 = handshake.mux %44 [%falseResult_16, %trueResult] : index, none loc(#loc53)
    %46 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc53)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %40 : none loc(#loc53)
    %47 = handshake.mux %44 [%falseResult_18, %trueResult] : index, none loc(#loc53)
    %48 = dataflow.carry %willContinue, %falseResult, %trueResult_21 : i1, none, none -> none loc(#loc53)
    %trueResult_19, %falseResult_20 = handshake.cond_br %12, %48 : none loc(#loc61)
    %49 = handshake.constant %48 {value = 0 : index} : index loc(#loc61)
    %50 = handshake.constant %48 {value = 1 : index} : index loc(#loc61)
    %51 = arith.select %12, %50, %49 : index loc(#loc61)
    %52 = handshake.mux %51 [%38#1, %trueResult_19] : index, none loc(#loc61)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue, %52 : none loc(#loc53)
    %53 = handshake.mux %44 [%falseResult_22, %trueResult] : index, none loc(#loc53)
    %54 = handshake.join %45, %47, %53 : none, none, none loc(#loc45)
    %55 = handshake.constant %54 {value = true} : i1 loc(#loc45)
    handshake.return %55 : i1 loc(#loc45)
  } loc(#loc45)
  handshake.func @_Z10modexp_dsaPKjS0_Pjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg3: i32 loc(fused<#di_subprogram4>[#loc14]), %arg4: i32 loc(fused<#di_subprogram4>[#loc14]), %arg5: none loc(fused<#di_subprogram4>[#loc14]), ...) -> none attributes {argNames = ["input_base", "input_exp", "output_result", "modulus", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : none loc(#loc45)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %0 {value = 1 : i64} : i64 loc(#loc2)
    %5 = arith.cmpi eq, %arg4, %2 : i32 loc(#loc56)
    %trueResult, %falseResult = handshake.cond_br %5, %0 : none loc(#loc53)
    %6 = arith.extui %arg3 : i32 to i64 loc(#loc2)
    %7 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc53)
    %8 = arith.index_cast %3 : i64 to index loc(#loc53)
    %9 = arith.index_cast %arg4 : i32 to index loc(#loc53)
    %index, %willContinue = dataflow.stream %8, %7, %9 {loom.annotations = ["loom.loop.no_parallel", "loom.loop.no_unroll"], step_op = "+=", stop_cond = "!="} loc(#loc53)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc53)
    %10 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc53)
    %dataResult, %addressResults = handshake.load [%afterValue] %38#0, %40 : index, i32 loc(#loc60)
    %11 = arith.cmpi eq, %dataResult, %2 : i32 loc(#loc61)
    %trueResult_0, %falseResult_1 = handshake.cond_br %11, %10 : none loc(#loc61)
    %dataResult_2, %addressResults_3 = handshake.load [%afterValue] %37#0, %falseResult_20 : index, i32 loc(#loc62)
    %12 = arith.remui %dataResult_2, %arg3 : i32 loc(#loc62)
    %13 = arith.extui %12 : i32 to i64 loc(#loc62)
    %14 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc61)
    %15 = arith.index_cast %dataResult : i32 to index loc(#loc61)
    %16 = arith.index_cast %2 : i32 to index loc(#loc61)
    %index_4, %willContinue_5 = dataflow.stream %15, %14, %16 {step_op = ">>=", stop_cond = "!="} loc(#loc61)
    %afterValue_6, %afterCond_7 = dataflow.gate %index_4, %willContinue_5 : index, i1 -> index, i1 loc(#loc61)
    %17 = dataflow.carry %willContinue_5, %13, %31 : i1, i64, i64 -> i64 loc(#loc61)
    %afterValue_8, %afterCond_9 = dataflow.gate %17, %willContinue_5 : i64, i1 -> i64, i1 loc(#loc61)
    handshake.sink %afterCond_9 : i1 loc(#loc61)
    %18 = dataflow.carry %willContinue_5, %4, %29 : i1, i64, i64 -> i64 loc(#loc61)
    %afterValue_10, %afterCond_11 = dataflow.gate %18, %willContinue_5 : i64, i1 -> i64, i1 loc(#loc61)
    handshake.sink %afterCond_11 : i1 loc(#loc61)
    %trueResult_12, %falseResult_13 = handshake.cond_br %willContinue_5, %18 : i64 loc(#loc61)
    %19 = dataflow.invariant %afterCond_7, %falseResult_1 : i1, none -> none loc(#loc61)
    %20 = arith.index_cast %afterValue_6 : index to i32 loc(#loc61)
    %21 = arith.andi %20, %1 : i32 loc(#loc72)
    %22 = arith.cmpi eq, %21, %2 : i32 loc(#loc72)
    %23 = arith.muli %afterValue_8, %afterValue_10 : i64 loc(#loc74)
    %24 = dataflow.invariant %afterCond, %6 : i1, i64 -> i64 loc(#loc74)
    %25 = arith.remui %23, %24 : i64 loc(#loc74)
    %26 = handshake.constant %19 {value = 0 : index} : index loc(#loc72)
    %27 = handshake.constant %19 {value = 1 : index} : index loc(#loc72)
    %28 = arith.select %22, %27, %26 : index loc(#loc72)
    %29 = handshake.mux %28 [%25, %afterValue_10] : index, i64 loc(#loc72)
    %30 = arith.muli %afterValue_8, %afterValue_8 : i64 loc(#loc69)
    %31 = arith.remui %30, %24 : i64 loc(#loc69)
    %32 = arith.trunci %falseResult_13 : i64 to i32 loc(#loc63)
    %33 = handshake.constant %10 {value = 0 : index} : index loc(#loc61)
    %34 = handshake.constant %10 {value = 1 : index} : index loc(#loc61)
    %35 = arith.select %11, %34, %33 : index loc(#loc61)
    %36 = handshake.mux %35 [%32, %1] : index, i32 loc(#loc61)
    %dataResult_14, %addressResult = handshake.store [%afterValue] %36, %45 : index, i32 loc(#loc63)
    %37:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_3) {id = 0 : i32} : (index) -> (i32, none) loc(#loc45)
    %38:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc45)
    %39 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_14, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc45)
    %40 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc53)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %38#1 : none loc(#loc53)
    %41 = handshake.constant %0 {value = 0 : index} : index loc(#loc53)
    %42 = handshake.constant %0 {value = 1 : index} : index loc(#loc53)
    %43 = arith.select %5, %42, %41 : index loc(#loc53)
    %44 = handshake.mux %43 [%falseResult_16, %trueResult] : index, none loc(#loc53)
    %45 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc53)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %39 : none loc(#loc53)
    %46 = handshake.mux %43 [%falseResult_18, %trueResult] : index, none loc(#loc53)
    %47 = dataflow.carry %willContinue, %falseResult, %trueResult_21 : i1, none, none -> none loc(#loc53)
    %trueResult_19, %falseResult_20 = handshake.cond_br %11, %47 : none loc(#loc61)
    %48 = handshake.constant %47 {value = 0 : index} : index loc(#loc61)
    %49 = handshake.constant %47 {value = 1 : index} : index loc(#loc61)
    %50 = arith.select %11, %49, %48 : index loc(#loc61)
    %51 = handshake.mux %50 [%37#1, %trueResult_19] : index, none loc(#loc61)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue, %51 : none loc(#loc53)
    %52 = handshake.mux %43 [%falseResult_22, %trueResult] : index, none loc(#loc53)
    %53 = handshake.join %44, %46, %52 : none, none, none loc(#loc45)
    handshake.return %53 : none loc(#loc46)
  } loc(#loc45)
  func.func @_Z10modexp_cpuPKjS0_Pjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc24]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc24]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc24]), %arg3: i32 loc(fused<#di_subprogram5>[#loc24]), %arg4: i32 loc(fused<#di_subprogram5>[#loc24])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg4, %c0_i32 : i32 loc(#loc57)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg3 : i32 to i64 loc(#loc2)
      %2 = arith.extui %arg4 : i32 to i64 loc(#loc57)
      %3 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        %4 = arith.index_cast %arg5 : i64 to index loc(#loc64)
        %5 = memref.load %arg1[%4] : memref<?xi32, strided<[1], offset: ?>> loc(#loc64)
        %6 = arith.cmpi eq, %5, %c0_i32 : i32 loc(#loc65)
        %7 = scf.if %6 -> (i32) {
          scf.yield %c1_i32 : i32 loc(#loc65)
        } else {
          %10 = memref.load %arg0[%4] : memref<?xi32, strided<[1], offset: ?>> loc(#loc66)
          %11 = arith.remui %10, %arg3 : i32 loc(#loc66)
          %12 = arith.extui %11 : i32 to i64 loc(#loc66)
          %13:3 = scf.while (%arg6 = %5, %arg7 = %12, %arg8 = %c1_i64) : (i32, i64, i64) -> (i32, i64, i64) {
            %15 = arith.andi %arg6, %c1_i32 : i32 loc(#loc73)
            %16 = arith.cmpi eq, %15, %c0_i32 : i32 loc(#loc73)
            %17 = scf.if %16 -> (i64) {
              scf.yield %arg8 : i64 loc(#loc73)
            } else {
              %22 = arith.muli %arg7, %arg8 : i64 loc(#loc75)
              %23 = arith.remui %22, %1 : i64 loc(#loc75)
              scf.yield %23 : i64 loc(#loc76)
            } loc(#loc73)
            %18 = arith.muli %arg7, %arg7 : i64 loc(#loc70)
            %19 = arith.remui %18, %1 : i64 loc(#loc70)
            %20 = arith.shrui %arg6, %c1_i32 : i32 loc(#loc71)
            %21 = arith.cmpi ne, %20, %c0_i32 : i32 loc(#loc65)
            scf.condition(%21) %20, %19, %17 : i32, i64, i64 loc(#loc65)
          } do {
          ^bb0(%arg6: i32 loc(fused<#di_lexical_block21>[#loc27]), %arg7: i64 loc(fused<#di_lexical_block21>[#loc27]), %arg8: i64 loc(fused<#di_lexical_block21>[#loc27])):
            scf.yield %arg6, %arg7, %arg8 : i32, i64, i64 loc(#loc65)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = ">>=", stop_cond = "!="}} loc(#loc65)
          %14 = arith.trunci %13#2 : i64 to i32 loc(#loc67)
          scf.yield %14 : i32 loc(#loc67)
        } loc(#loc65)
        memref.store %7, %arg2[%4] : memref<?xi32, strided<[1], offset: ?>> loc(#loc67)
        %8 = arith.addi %arg5, %c1_i64 : i64 loc(#loc57)
        %9 = arith.cmpi ne, %8, %2 : i64 loc(#loc68)
        scf.condition(%9) %8 : i64 loc(#loc54)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block15>[#loc25])):
        scf.yield %arg5 : i64 loc(#loc54)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc54)
    } loc(#loc54)
    return loc(#loc48)
  } loc(#loc47)
} loc(#loc)
#loc = loc("tests/app/modexp/main.cpp":0:0)
#loc1 = loc("tests/app/modexp/main.cpp":5:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/modexp/main.cpp":19:0)
#loc5 = loc("tests/app/modexp/main.cpp":20:0)
#loc6 = loc("tests/app/modexp/main.cpp":24:0)
#loc7 = loc("tests/app/modexp/main.cpp":27:0)
#loc9 = loc("tests/app/modexp/main.cpp":31:0)
#loc10 = loc("tests/app/modexp/main.cpp":32:0)
#loc11 = loc("tests/app/modexp/main.cpp":33:0)
#loc12 = loc("tests/app/modexp/main.cpp":37:0)
#loc13 = loc("tests/app/modexp/main.cpp":39:0)
#loc15 = loc("tests/app/modexp/modexp.cpp":43:0)
#loc16 = loc("tests/app/modexp/modexp.cpp":46:0)
#loc17 = loc("tests/app/modexp/modexp.cpp":48:0)
#loc18 = loc("tests/app/modexp/modexp.cpp":45:0)
#loc19 = loc("tests/app/modexp/modexp.cpp":49:0)
#loc20 = loc("tests/app/modexp/modexp.cpp":50:0)
#loc21 = loc("tests/app/modexp/modexp.cpp":52:0)
#loc22 = loc("tests/app/modexp/modexp.cpp":56:0)
#loc23 = loc("tests/app/modexp/modexp.cpp":58:0)
#loc26 = loc("tests/app/modexp/modexp.cpp":20:0)
#loc28 = loc("tests/app/modexp/modexp.cpp":19:0)
#loc29 = loc("tests/app/modexp/modexp.cpp":23:0)
#loc30 = loc("tests/app/modexp/modexp.cpp":24:0)
#loc31 = loc("tests/app/modexp/modexp.cpp":25:0)
#loc32 = loc("tests/app/modexp/modexp.cpp":26:0)
#loc33 = loc("tests/app/modexp/modexp.cpp":27:0)
#loc34 = loc("tests/app/modexp/modexp.cpp":30:0)
#loc35 = loc("tests/app/modexp/modexp.cpp":32:0)
#loc36 = loc(fused<#di_subprogram3>[#loc1])
#loc37 = loc(fused<#di_subprogram3>[#loc6])
#loc38 = loc(fused<#di_subprogram3>[#loc7])
#loc39 = loc(fused<#di_subprogram3>[#loc12])
#loc40 = loc(fused<#di_subprogram3>[#loc13])
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 18>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 30>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 18>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 30>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 43>
#loc43 = loc(fused<#di_lexical_block10>[#loc3])
#loc44 = loc(fused<#di_lexical_block11>[#loc8])
#loc46 = loc(fused<#di_subprogram4>[#loc23])
#loc48 = loc(fused<#di_subprogram5>[#loc35])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 31>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 43>
#loc49 = loc(fused<#di_lexical_block12>[#loc4])
#loc50 = loc(fused<#di_lexical_block12>[#loc5])
#loc51 = loc(fused[#loc41, #loc43])
#loc52 = loc(fused[#loc42, #loc44])
#loc53 = loc(fused<#di_lexical_block14>[#loc15])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 31>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 43>
#loc55 = loc(fused<#di_lexical_block16>[#loc9])
#loc56 = loc(fused<#di_lexical_block17>[#loc15])
#loc57 = loc(fused<#di_lexical_block18>[#loc25])
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file1, line = 48>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file1, line = 22>
#loc58 = loc(fused<#di_lexical_block19>[#loc10])
#loc59 = loc(fused<#di_lexical_block19>[#loc11])
#loc60 = loc(fused<#di_lexical_block20>[#loc16])
#loc61 = loc(fused<#di_lexical_block20>[#loc17])
#loc62 = loc(fused<#di_lexical_block20>[#loc18])
#loc63 = loc(fused<#di_lexical_block20>[#loc22])
#loc64 = loc(fused<#di_lexical_block21>[#loc26])
#loc66 = loc(fused<#di_lexical_block21>[#loc28])
#loc67 = loc(fused<#di_lexical_block21>[#loc34])
#loc68 = loc(fused[#loc54, #loc57])
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 49>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file1, line = 23>
#loc69 = loc(fused<#di_lexical_block22>[#loc21])
#loc70 = loc(fused<#di_lexical_block23>[#loc32])
#loc71 = loc(fused<#di_lexical_block23>[#loc33])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file1, line = 49>
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file1, line = 23>
#loc72 = loc(fused<#di_lexical_block24>[#loc19])
#loc73 = loc(fused<#di_lexical_block25>[#loc29])
#loc74 = loc(fused<#di_lexical_block26>[#loc20])
#loc75 = loc(fused<#di_lexical_block27>[#loc30])
#loc76 = loc(fused<#di_lexical_block27>[#loc31])
