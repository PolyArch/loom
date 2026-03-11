#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_file = #llvm.di_file<"tests/app/newton_iter/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/newton_iter/newton_iter.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/newton_iter/main.cpp":19:0)
#loc10 = loc("tests/app/newton_iter/main.cpp":33:0)
#loc16 = loc("tests/app/newton_iter/newton_iter.cpp":29:0)
#loc20 = loc("tests/app/newton_iter/newton_iter.cpp":17:0)
#loc21 = loc("tests/app/newton_iter/newton_iter.cpp":22:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type2, sizeInBits = 2048, elements = #llvm.di_subrange<count = 64 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type2>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type2, sizeInBits = 64>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 19>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 33>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 36>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 22>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type2>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 19>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "input_x", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_f", file = #di_file, line = 11, type = #di_composite_type>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_df", file = #di_file, line = 12, type = #di_composite_type>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_x", file = #di_file, line = 15, type = #di_composite_type>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_x", file = #di_file, line = 16, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type3>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 19>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 19, type = #di_derived_type3>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 33, type = #di_derived_type3>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_x", file = #di_file1, line = 32, arg = 4, type = #di_derived_type5>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 36, type = #di_derived_type3>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_x", file = #di_file1, line = 20, arg = 4, type = #di_derived_type5>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 22, type = #di_derived_type3>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 7, type = #di_derived_type6>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block5, name = "c", file = #di_file, line = 20, type = #di_basic_type2>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_x", file = #di_file1, line = 29, arg = 1, type = #di_derived_type7>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_f", file = #di_file1, line = 30, arg = 2, type = #di_derived_type7>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_df", file = #di_file1, line = 31, arg = 3, type = #di_derived_type7>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 33, arg = 5, type = #di_derived_type6>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_x", file = #di_file1, line = 17, arg = 1, type = #di_derived_type7>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_f", file = #di_file1, line = 18, arg = 2, type = #di_derived_type7>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_df", file = #di_file1, line = 19, arg = 3, type = #di_derived_type7>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 21, arg = 5, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type7, #di_derived_type7, #di_derived_type7, #di_derived_type5, #di_derived_type6>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable11, #di_local_variable, #di_local_variable1, #di_local_variable2, #di_local_variable3, #di_local_variable4, #di_local_variable5, #di_local_variable12, #di_local_variable6>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "newton_iter_dsa", linkageName = "_Z15newton_iter_dsaPKfS0_S0_Pfj", file = #di_file1, line = 29, scopeLine = 33, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable13, #di_local_variable14, #di_local_variable15, #di_local_variable7, #di_local_variable16, #di_local_variable8>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "newton_iter_cpu", linkageName = "_Z15newton_iter_cpuPKfS0_S0_Pfj", file = #di_file1, line = 17, scopeLine = 21, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable17, #di_local_variable18, #di_local_variable19, #di_local_variable9, #di_local_variable20, #di_local_variable10>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 19>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 33>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 22>
#loc29 = loc(fused<#di_subprogram4>[#loc16])
#loc31 = loc(fused<#di_subprogram5>[#loc20])
#loc33 = loc(fused<#di_lexical_block6>[#loc3])
#loc34 = loc(fused<#di_lexical_block7>[#loc10])
#loc36 = loc(fused<#di_lexical_block9>[#loc21])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<20xi8> = dense<[110, 101, 119, 116, 111, 110, 95, 105, 116, 101, 114, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<20xi8> = dense<[110, 101, 119, 116, 111, 110, 95, 105, 116, 101, 114, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<38xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 110, 101, 119, 116, 111, 110, 95, 105, 116, 101, 114, 47, 110, 101, 119, 116, 111, 110, 95, 105, 116, 101, 114, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc24)
    %false = arith.constant false loc(#loc24)
    %0 = seq.const_clock  low loc(#loc24)
    %c2_i32 = arith.constant 2 : i32 loc(#loc24)
    %1 = ub.poison : i64 loc(#loc24)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %cst = arith.constant 2.000000e+00 : f32 loc(#loc2)
    %c64_i64 = arith.constant 64 : i64 loc(#loc2)
    %c64_i32 = arith.constant 64 : i32 loc(#loc2)
    %cst_0 = arith.constant 9.99999974E-6 : f32 loc(#loc2)
    %2 = memref.get_global @str : memref<20xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<20xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<64xf32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<64xf32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<64xf32> loc(#loc2)
    %alloca_3 = memref.alloca() : memref<64xf32> loc(#loc2)
    %alloca_4 = memref.alloca() : memref<64xf32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.addi %arg0, %c1_i64 : i64 loc(#loc41)
      %11 = arith.trunci %10 : i64 to i32 loc(#loc41)
      %12 = arith.uitofp %11 : i32 to f32 loc(#loc41)
      %13 = arith.index_cast %arg0 : i64 to index loc(#loc42)
      memref.store %12, %alloca[%13] : memref<64xf32> loc(#loc42)
      %14 = arith.negf %12 : f32 loc(#loc43)
      %15 = math.fma %12, %12, %14 : f32 loc(#loc43)
      memref.store %15, %alloca_1[%13] : memref<64xf32> loc(#loc43)
      %16 = arith.mulf %12, %cst : f32 loc(#loc44)
      memref.store %16, %alloca_2[%13] : memref<64xf32> loc(#loc44)
      %17 = arith.cmpi ne, %10, %c64_i64 : i64 loc(#loc45)
      scf.condition(%17) %10 : i64 loc(#loc33)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block6>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc33)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc33)
    %cast = memref.cast %alloca : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc25)
    %cast_5 = memref.cast %alloca_1 : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc25)
    %cast_6 = memref.cast %alloca_2 : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc25)
    %cast_7 = memref.cast %alloca_3 : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc25)
    call @_Z15newton_iter_cpuPKfS0_S0_Pfj(%cast, %cast_5, %cast_6, %cast_7, %c64_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, i32) -> () loc(#loc25)
    %cast_8 = memref.cast %alloca_4 : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc26)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc26)
    %chanOutput_9, %ready_10 = esi.wrap.vr %cast_5, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc26)
    %chanOutput_11, %ready_12 = esi.wrap.vr %cast_6, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc26)
    %chanOutput_13, %ready_14 = esi.wrap.vr %cast_8, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc26)
    %chanOutput_15, %ready_16 = esi.wrap.vr %c64_i32, %true : i32 loc(#loc26)
    %chanOutput_17, %ready_18 = esi.wrap.vr %true, %true : i1 loc(#loc26)
    %5 = handshake.esi_instance @_Z15newton_iter_dsaPKfS0_S0_Pfj_esi "_Z15newton_iter_dsaPKfS0_S0_Pfj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_9, %chanOutput_11, %chanOutput_13, %chanOutput_15, %chanOutput_17) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc26)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc26)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc50)
      %11 = memref.load %alloca_3[%10] : memref<64xf32> loc(#loc50)
      %12 = memref.load %alloca_4[%10] : memref<64xf32> loc(#loc50)
      %13 = arith.subf %11, %12 : f32 loc(#loc50)
      %14 = math.absf %13 : f32 loc(#loc50)
      %15 = arith.cmpf ule, %14, %cst_0 : f32 loc(#loc50)
      %16:3 = scf.if %15 -> (i64, i32, i32) {
        %18 = arith.addi %arg0, %c1_i64 : i64 loc(#loc38)
        %19 = arith.cmpi eq, %18, %c64_i64 : i64 loc(#loc38)
        %20 = arith.extui %19 : i1 to i32 loc(#loc34)
        %21 = arith.cmpi ne, %18, %c64_i64 : i64 loc(#loc46)
        %22 = arith.extui %21 : i1 to i32 loc(#loc34)
        scf.yield %18, %20, %22 : i64, i32, i32 loc(#loc50)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc50)
      } loc(#loc50)
      %17 = arith.trunci %16#2 : i32 to i1 loc(#loc34)
      scf.condition(%17) %16#0, %15, %16#1 : i64, i1, i32 loc(#loc34)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block7>[#loc10]), %arg1: i1 loc(fused<#di_lexical_block7>[#loc10]), %arg2: i32 loc(fused<#di_lexical_block7>[#loc10])):
      scf.yield %arg0 : i64 loc(#loc34)
    } loc(#loc34)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc34)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc34)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<20xi8> -> index loc(#loc51)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc51)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc51)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc51)
      scf.yield %c1_i32 : i32 loc(#loc52)
    } loc(#loc34)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<20xi8> -> index loc(#loc27)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc27)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc27)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc27)
    } loc(#loc2)
    return %9 : i32 loc(#loc28)
  } loc(#loc24)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.fmuladd.f32(f32, f32, f32) -> f32 loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z15newton_iter_dsaPKfS0_S0_Pfj_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc16]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc16]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc16]), %arg3: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc16]), %arg4: i32 loc(fused<#di_subprogram4>[#loc16]), %arg5: i1 loc(fused<#di_subprogram4>[#loc16]), ...) -> i1 attributes {argNames = ["input_x", "input_f", "input_df", "output_x", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : i1 loc(#loc29)
    %1 = handshake.join %0 : none loc(#loc29)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg4, %2 : i32 loc(#loc39)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc35)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc35)
    %6 = arith.index_cast %3 : i64 to index loc(#loc35)
    %7 = arith.index_cast %arg4 : i32 to index loc(#loc35)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.no_parallel", "loom.loop.no_unroll"], step_op = "+=", stop_cond = "!="} loc(#loc35)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc35)
    %dataResult, %addressResults = handshake.load [%afterValue] %11#0, %14 : index, f32 loc(#loc47)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %12#0, %21 : index, f32 loc(#loc47)
    %dataResult_2, %addressResults_3 = handshake.load [%afterValue] %10#0, %19 : index, f32 loc(#loc47)
    %8 = arith.divf %dataResult_0, %dataResult_2 : f32 loc(#loc47)
    %9 = arith.subf %dataResult, %8 : f32 loc(#loc47)
    %dataResult_4, %addressResult = handshake.store [%afterValue] %9, %23 : index, f32 loc(#loc47)
    %10:2 = handshake.extmemory[ld = 1, st = 0] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_3) {id = 0 : i32} : (index) -> (f32, none) loc(#loc29)
    %11:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (f32, none) loc(#loc29)
    %12:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_1) {id = 2 : i32} : (index) -> (f32, none) loc(#loc29)
    %13 = handshake.extmemory[ld = 0, st = 1] (%arg3 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_4, %addressResult) {id = 3 : i32} : (f32, index) -> none loc(#loc29)
    %14 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc35)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %11#1 : none loc(#loc35)
    %15 = handshake.constant %1 {value = 0 : index} : index loc(#loc35)
    %16 = handshake.constant %1 {value = 1 : index} : index loc(#loc35)
    %17 = arith.select %4, %16, %15 : index loc(#loc35)
    %18 = handshake.mux %17 [%falseResult_6, %trueResult] : index, none loc(#loc35)
    %19 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc35)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %10#1 : none loc(#loc35)
    %20 = handshake.mux %17 [%falseResult_8, %trueResult] : index, none loc(#loc35)
    %21 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc35)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %12#1 : none loc(#loc35)
    %22 = handshake.mux %17 [%falseResult_10, %trueResult] : index, none loc(#loc35)
    %23 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc35)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %13 : none loc(#loc35)
    %24 = handshake.mux %17 [%falseResult_12, %trueResult] : index, none loc(#loc35)
    %25 = handshake.join %18, %20, %22, %24 : none, none, none, none loc(#loc29)
    %26 = handshake.constant %25 {value = true} : i1 loc(#loc29)
    handshake.return %26 : i1 loc(#loc29)
  } loc(#loc29)
  handshake.func @_Z15newton_iter_dsaPKfS0_S0_Pfj(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc16]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc16]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc16]), %arg3: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc16]), %arg4: i32 loc(fused<#di_subprogram4>[#loc16]), %arg5: none loc(fused<#di_subprogram4>[#loc16]), ...) -> none attributes {argNames = ["input_x", "input_f", "input_df", "output_x", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : none loc(#loc29)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi eq, %arg4, %1 : i32 loc(#loc39)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc35)
    %4 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc35)
    %5 = arith.index_cast %2 : i64 to index loc(#loc35)
    %6 = arith.index_cast %arg4 : i32 to index loc(#loc35)
    %index, %willContinue = dataflow.stream %5, %4, %6 {loom.annotations = ["loom.loop.no_parallel", "loom.loop.no_unroll"], step_op = "+=", stop_cond = "!="} loc(#loc35)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc35)
    %dataResult, %addressResults = handshake.load [%afterValue] %10#0, %13 : index, f32 loc(#loc47)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %11#0, %20 : index, f32 loc(#loc47)
    %dataResult_2, %addressResults_3 = handshake.load [%afterValue] %9#0, %18 : index, f32 loc(#loc47)
    %7 = arith.divf %dataResult_0, %dataResult_2 : f32 loc(#loc47)
    %8 = arith.subf %dataResult, %7 : f32 loc(#loc47)
    %dataResult_4, %addressResult = handshake.store [%afterValue] %8, %22 : index, f32 loc(#loc47)
    %9:2 = handshake.extmemory[ld = 1, st = 0] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_3) {id = 0 : i32} : (index) -> (f32, none) loc(#loc29)
    %10:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (f32, none) loc(#loc29)
    %11:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_1) {id = 2 : i32} : (index) -> (f32, none) loc(#loc29)
    %12 = handshake.extmemory[ld = 0, st = 1] (%arg3 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_4, %addressResult) {id = 3 : i32} : (f32, index) -> none loc(#loc29)
    %13 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc35)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %10#1 : none loc(#loc35)
    %14 = handshake.constant %0 {value = 0 : index} : index loc(#loc35)
    %15 = handshake.constant %0 {value = 1 : index} : index loc(#loc35)
    %16 = arith.select %3, %15, %14 : index loc(#loc35)
    %17 = handshake.mux %16 [%falseResult_6, %trueResult] : index, none loc(#loc35)
    %18 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc35)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %9#1 : none loc(#loc35)
    %19 = handshake.mux %16 [%falseResult_8, %trueResult] : index, none loc(#loc35)
    %20 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc35)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %11#1 : none loc(#loc35)
    %21 = handshake.mux %16 [%falseResult_10, %trueResult] : index, none loc(#loc35)
    %22 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc35)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %12 : none loc(#loc35)
    %23 = handshake.mux %16 [%falseResult_12, %trueResult] : index, none loc(#loc35)
    %24 = handshake.join %17, %19, %21, %23 : none, none, none, none loc(#loc29)
    handshake.return %24 : none loc(#loc30)
  } loc(#loc29)
  func.func @_Z15newton_iter_cpuPKfS0_S0_Pfj(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg2: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg3: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg4: i32 loc(fused<#di_subprogram5>[#loc20])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg4, %c0_i32 : i32 loc(#loc40)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg4 : i32 to i64 loc(#loc40)
      %2 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg5 : i64 to index loc(#loc48)
        %4 = memref.load %arg0[%3] : memref<?xf32, strided<[1], offset: ?>> loc(#loc48)
        %5 = memref.load %arg1[%3] : memref<?xf32, strided<[1], offset: ?>> loc(#loc48)
        %6 = memref.load %arg2[%3] : memref<?xf32, strided<[1], offset: ?>> loc(#loc48)
        %7 = arith.divf %5, %6 : f32 loc(#loc48)
        %8 = arith.subf %4, %7 : f32 loc(#loc48)
        memref.store %8, %arg3[%3] : memref<?xf32, strided<[1], offset: ?>> loc(#loc48)
        %9 = arith.addi %arg5, %c1_i64 : i64 loc(#loc40)
        %10 = arith.cmpi ne, %9, %1 : i64 loc(#loc49)
        scf.condition(%10) %9 : i64 loc(#loc36)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block9>[#loc21])):
        scf.yield %arg5 : i64 loc(#loc36)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc36)
    } loc(#loc36)
    return loc(#loc32)
  } loc(#loc31)
} loc(#loc)
#loc = loc("tests/app/newton_iter/main.cpp":0:0)
#loc1 = loc("tests/app/newton_iter/main.cpp":6:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/newton_iter/main.cpp":20:0)
#loc5 = loc("tests/app/newton_iter/main.cpp":21:0)
#loc6 = loc("tests/app/newton_iter/main.cpp":22:0)
#loc7 = loc("tests/app/newton_iter/main.cpp":23:0)
#loc8 = loc("tests/app/newton_iter/main.cpp":27:0)
#loc9 = loc("tests/app/newton_iter/main.cpp":30:0)
#loc11 = loc("tests/app/newton_iter/main.cpp":34:0)
#loc12 = loc("tests/app/newton_iter/main.cpp":35:0)
#loc13 = loc("tests/app/newton_iter/main.cpp":36:0)
#loc14 = loc("tests/app/newton_iter/main.cpp":40:0)
#loc15 = loc("tests/app/newton_iter/main.cpp":42:0)
#loc17 = loc("tests/app/newton_iter/newton_iter.cpp":36:0)
#loc18 = loc("tests/app/newton_iter/newton_iter.cpp":37:0)
#loc19 = loc("tests/app/newton_iter/newton_iter.cpp":39:0)
#loc22 = loc("tests/app/newton_iter/newton_iter.cpp":23:0)
#loc23 = loc("tests/app/newton_iter/newton_iter.cpp":25:0)
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 36>
#loc24 = loc(fused<#di_subprogram3>[#loc1])
#loc25 = loc(fused<#di_subprogram3>[#loc8])
#loc26 = loc(fused<#di_subprogram3>[#loc9])
#loc27 = loc(fused<#di_subprogram3>[#loc14])
#loc28 = loc(fused<#di_subprogram3>[#loc15])
#loc30 = loc(fused<#di_subprogram4>[#loc19])
#loc32 = loc(fused<#di_subprogram5>[#loc23])
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 19>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 33>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 36>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file1, line = 22>
#loc35 = loc(fused<#di_lexical_block8>[#loc17])
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 19>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 33>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file1, line = 36>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file1, line = 22>
#loc37 = loc(fused<#di_lexical_block10>[#loc3])
#loc38 = loc(fused<#di_lexical_block11>[#loc10])
#loc39 = loc(fused<#di_lexical_block12>[#loc17])
#loc40 = loc(fused<#di_lexical_block13>[#loc21])
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file, line = 34>
#loc41 = loc(fused<#di_lexical_block14>[#loc4])
#loc42 = loc(fused<#di_lexical_block14>[#loc5])
#loc43 = loc(fused<#di_lexical_block14>[#loc6])
#loc44 = loc(fused<#di_lexical_block14>[#loc7])
#loc45 = loc(fused[#loc33, #loc37])
#loc46 = loc(fused[#loc34, #loc38])
#loc47 = loc(fused<#di_lexical_block16>[#loc18])
#loc48 = loc(fused<#di_lexical_block17>[#loc22])
#loc49 = loc(fused[#loc36, #loc40])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file, line = 34>
#loc50 = loc(fused<#di_lexical_block18>[#loc11])
#loc51 = loc(fused<#di_lexical_block19>[#loc12])
#loc52 = loc(fused<#di_lexical_block19>[#loc13])
