#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_file = #llvm.di_file<"tests/app/runge_kutta_step/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/runge_kutta_step/runge_kutta_step.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/runge_kutta_step/main.cpp":22:0)
#loc11 = loc("tests/app/runge_kutta_step/main.cpp":37:0)
#loc17 = loc("tests/app/runge_kutta_step/runge_kutta_step.cpp":35:0)
#loc21 = loc("tests/app/runge_kutta_step/runge_kutta_step.cpp":20:0)
#loc22 = loc("tests/app/runge_kutta_step/runge_kutta_step.cpp":28:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type2, sizeInBits = 2048, elements = #llvm.di_subrange<count = 64 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type2>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type2, sizeInBits = 64>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 22>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 37>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 44>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 28>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type2>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "h", file = #di_file, line = 8, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_y", file = #di_file, line = 11, type = #di_composite_type>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_k1", file = #di_file, line = 12, type = #di_composite_type>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_k2", file = #di_file, line = 13, type = #di_composite_type>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_k3", file = #di_file, line = 14, type = #di_composite_type>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_k4", file = #di_file, line = 15, type = #di_composite_type>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_y", file = #di_file, line = 18, type = #di_composite_type>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_y", file = #di_file, line = 19, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram1, name = "h", file = #di_file1, line = 41, arg = 7, type = #di_derived_type1>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram2, name = "h", file = #di_file1, line = 26, arg = 7, type = #di_derived_type1>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type3>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 22, type = #di_derived_type3>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 37, type = #di_derived_type3>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_y", file = #di_file1, line = 40, arg = 6, type = #di_derived_type5>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 44, type = #di_derived_type3>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_y", file = #di_file1, line = 25, arg = 6, type = #di_derived_type5>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 28, type = #di_derived_type3>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 7, type = #di_derived_type6>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_y", file = #di_file1, line = 35, arg = 1, type = #di_derived_type7>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_k1", file = #di_file1, line = 36, arg = 2, type = #di_derived_type7>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_k2", file = #di_file1, line = 37, arg = 3, type = #di_derived_type7>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_k3", file = #di_file1, line = 38, arg = 4, type = #di_derived_type7>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_k4", file = #di_file1, line = 39, arg = 5, type = #di_derived_type7>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 42, arg = 8, type = #di_derived_type6>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_y", file = #di_file1, line = 20, arg = 1, type = #di_derived_type7>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_k1", file = #di_file1, line = 21, arg = 2, type = #di_derived_type7>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_k2", file = #di_file1, line = 22, arg = 3, type = #di_derived_type7>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_k3", file = #di_file1, line = 23, arg = 4, type = #di_derived_type7>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_k4", file = #di_file1, line = 24, arg = 5, type = #di_derived_type7>
#di_local_variable28 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 27, arg = 8, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type7, #di_derived_type7, #di_derived_type7, #di_derived_type7, #di_derived_type7, #di_derived_type5, #di_derived_type1, #di_derived_type6>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable16, #di_local_variable, #di_local_variable1, #di_local_variable2, #di_local_variable3, #di_local_variable4, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable10, #di_local_variable11>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "runge_kutta_step_dsa", linkageName = "_Z20runge_kutta_step_dsaPKfS0_S0_S0_S0_Pffj", file = #di_file1, line = 35, scopeLine = 42, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable17, #di_local_variable18, #di_local_variable19, #di_local_variable20, #di_local_variable21, #di_local_variable12, #di_local_variable8, #di_local_variable22, #di_local_variable13>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "runge_kutta_step_cpu", linkageName = "_Z20runge_kutta_step_cpuPKfS0_S0_S0_S0_Pffj", file = #di_file1, line = 20, scopeLine = 27, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable23, #di_local_variable24, #di_local_variable25, #di_local_variable26, #di_local_variable27, #di_local_variable14, #di_local_variable9, #di_local_variable28, #di_local_variable15>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 22>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 37>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 28>
#loc30 = loc(fused<#di_subprogram4>[#loc17])
#loc32 = loc(fused<#di_subprogram5>[#loc21])
#loc34 = loc(fused<#di_lexical_block4>[#loc3])
#loc35 = loc(fused<#di_lexical_block5>[#loc11])
#loc37 = loc(fused<#di_lexical_block7>[#loc22])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<25xi8> = dense<[114, 117, 110, 103, 101, 95, 107, 117, 116, 116, 97, 95, 115, 116, 101, 112, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<25xi8> = dense<[114, 117, 110, 103, 101, 95, 107, 117, 116, 116, 97, 95, 115, 116, 101, 112, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<48xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 114, 117, 110, 103, 101, 95, 107, 117, 116, 116, 97, 95, 115, 116, 101, 112, 47, 114, 117, 110, 103, 101, 95, 107, 117, 116, 116, 97, 95, 115, 116, 101, 112, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc25)
    %false = arith.constant false loc(#loc25)
    %0 = seq.const_clock  low loc(#loc25)
    %c2_i32 = arith.constant 2 : i32 loc(#loc25)
    %1 = ub.poison : i64 loc(#loc25)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %cst = arith.constant 1.000000e-01 : f32 loc(#loc2)
    %cst_0 = arith.constant 1.000000e+00 : f32 loc(#loc2)
    %cst_1 = arith.constant 1.100000e+00 : f32 loc(#loc2)
    %cst_2 = arith.constant 1.200000e+00 : f32 loc(#loc2)
    %cst_3 = arith.constant 1.300000e+00 : f32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c64_i64 = arith.constant 64 : i64 loc(#loc2)
    %c64_i32 = arith.constant 64 : i32 loc(#loc2)
    %cst_4 = arith.constant 9.99999974E-6 : f32 loc(#loc2)
    %2 = memref.get_global @str : memref<25xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<25xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<64xf32> loc(#loc2)
    %alloca_5 = memref.alloca() : memref<64xf32> loc(#loc2)
    %alloca_6 = memref.alloca() : memref<64xf32> loc(#loc2)
    %alloca_7 = memref.alloca() : memref<64xf32> loc(#loc2)
    %alloca_8 = memref.alloca() : memref<64xf32> loc(#loc2)
    %alloca_9 = memref.alloca() : memref<64xf32> loc(#loc2)
    %alloca_10 = memref.alloca() : memref<64xf32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc42)
      %11 = arith.uitofp %10 : i32 to f32 loc(#loc42)
      %12 = arith.index_cast %arg0 : i64 to index loc(#loc42)
      memref.store %11, %alloca[%12] : memref<64xf32> loc(#loc42)
      %13 = math.fma %11, %cst, %cst_0 : f32 loc(#loc43)
      memref.store %13, %alloca_5[%12] : memref<64xf32> loc(#loc43)
      %14 = math.fma %11, %cst, %cst_1 : f32 loc(#loc44)
      memref.store %14, %alloca_6[%12] : memref<64xf32> loc(#loc44)
      %15 = math.fma %11, %cst, %cst_2 : f32 loc(#loc45)
      memref.store %15, %alloca_7[%12] : memref<64xf32> loc(#loc45)
      %16 = math.fma %11, %cst, %cst_3 : f32 loc(#loc46)
      memref.store %16, %alloca_8[%12] : memref<64xf32> loc(#loc46)
      %17 = arith.addi %arg0, %c1_i64 : i64 loc(#loc38)
      %18 = arith.cmpi ne, %17, %c64_i64 : i64 loc(#loc47)
      scf.condition(%18) %17 : i64 loc(#loc34)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block4>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc34)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc34)
    %cast = memref.cast %alloca : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc26)
    %cast_11 = memref.cast %alloca_5 : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc26)
    %cast_12 = memref.cast %alloca_6 : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc26)
    %cast_13 = memref.cast %alloca_7 : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc26)
    %cast_14 = memref.cast %alloca_8 : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc26)
    %cast_15 = memref.cast %alloca_9 : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc26)
    call @_Z20runge_kutta_step_cpuPKfS0_S0_S0_S0_Pffj(%cast, %cast_11, %cast_12, %cast_13, %cast_14, %cast_15, %cst, %c64_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, f32, i32) -> () loc(#loc26)
    %cast_16 = memref.cast %alloca_10 : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput_17, %ready_18 = esi.wrap.vr %cast_11, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput_19, %ready_20 = esi.wrap.vr %cast_12, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput_21, %ready_22 = esi.wrap.vr %cast_13, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput_23, %ready_24 = esi.wrap.vr %cast_14, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput_25, %ready_26 = esi.wrap.vr %cast_16, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput_27, %ready_28 = esi.wrap.vr %cst, %true : f32 loc(#loc27)
    %chanOutput_29, %ready_30 = esi.wrap.vr %c64_i32, %true : i32 loc(#loc27)
    %chanOutput_31, %ready_32 = esi.wrap.vr %true, %true : i1 loc(#loc27)
    %5 = handshake.esi_instance @_Z20runge_kutta_step_dsaPKfS0_S0_S0_S0_Pffj_esi "_Z20runge_kutta_step_dsaPKfS0_S0_S0_S0_Pffj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_17, %chanOutput_19, %chanOutput_21, %chanOutput_23, %chanOutput_25, %chanOutput_27, %chanOutput_29, %chanOutput_31) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<f32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc27)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc27)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc52)
      %11 = memref.load %alloca_9[%10] : memref<64xf32> loc(#loc52)
      %12 = memref.load %alloca_10[%10] : memref<64xf32> loc(#loc52)
      %13 = arith.subf %11, %12 : f32 loc(#loc52)
      %14 = math.absf %13 : f32 loc(#loc52)
      %15 = arith.cmpf ule, %14, %cst_4 : f32 loc(#loc52)
      %16:3 = scf.if %15 -> (i64, i32, i32) {
        %18 = arith.addi %arg0, %c1_i64 : i64 loc(#loc39)
        %19 = arith.cmpi eq, %18, %c64_i64 : i64 loc(#loc39)
        %20 = arith.extui %19 : i1 to i32 loc(#loc35)
        %21 = arith.cmpi ne, %18, %c64_i64 : i64 loc(#loc48)
        %22 = arith.extui %21 : i1 to i32 loc(#loc35)
        scf.yield %18, %20, %22 : i64, i32, i32 loc(#loc52)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc52)
      } loc(#loc52)
      %17 = arith.trunci %16#2 : i32 to i1 loc(#loc35)
      scf.condition(%17) %16#0, %15, %16#1 : i64, i1, i32 loc(#loc35)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block5>[#loc11]), %arg1: i1 loc(fused<#di_lexical_block5>[#loc11]), %arg2: i32 loc(fused<#di_lexical_block5>[#loc11])):
      scf.yield %arg0 : i64 loc(#loc35)
    } loc(#loc35)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc35)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc35)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<25xi8> -> index loc(#loc53)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc53)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc53)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc53)
      scf.yield %c1_i32 : i32 loc(#loc54)
    } loc(#loc35)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<25xi8> -> index loc(#loc28)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc28)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc28)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc28)
    } loc(#loc2)
    return %9 : i32 loc(#loc29)
  } loc(#loc25)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.fmuladd.f32(f32, f32, f32) -> f32 loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z20runge_kutta_step_dsaPKfS0_S0_S0_S0_Pffj_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg3: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg4: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg5: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg6: f32 loc(fused<#di_subprogram4>[#loc17]), %arg7: i32 loc(fused<#di_subprogram4>[#loc17]), %arg8: i1 loc(fused<#di_subprogram4>[#loc17]), ...) -> i1 attributes {argNames = ["input_y", "input_k1", "input_k2", "input_k3", "input_k4", "output_y", "h", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg8 : i1 loc(#loc30)
    %1 = handshake.join %0 : none loc(#loc30)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 6.000000e+00 : f32} : f32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %1 {value = 2.000000e+00 : f32} : f32 loc(#loc2)
    %6 = arith.cmpi eq, %arg7, %2 : i32 loc(#loc40)
    %trueResult, %falseResult = handshake.cond_br %6, %1 : none loc(#loc36)
    %7 = arith.divf %arg6, %3 : f32 loc(#loc2)
    %8 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc36)
    %9 = arith.index_cast %4 : i64 to index loc(#loc36)
    %10 = arith.index_cast %arg7 : i32 to index loc(#loc36)
    %index, %willContinue = dataflow.stream %9, %8, %10 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc36)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc36)
    %dataResult, %addressResults = handshake.load [%afterValue] %17#0, %22 : index, f32 loc(#loc49)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %19#0, %31 : index, f32 loc(#loc49)
    %dataResult_2, %addressResults_3 = handshake.load [%afterValue] %20#0, %27 : index, f32 loc(#loc49)
    %11 = math.fma %dataResult_2, %5, %dataResult_0 : f32 loc(#loc49)
    %dataResult_4, %addressResults_5 = handshake.load [%afterValue] %16#0, %33 : index, f32 loc(#loc49)
    %12 = math.fma %dataResult_4, %5, %11 : f32 loc(#loc49)
    %dataResult_6, %addressResults_7 = handshake.load [%afterValue] %18#0, %29 : index, f32 loc(#loc49)
    %13 = arith.addf %12, %dataResult_6 : f32 loc(#loc49)
    %14 = dataflow.invariant %afterCond, %7 : i1, f32 -> f32 loc(#loc49)
    %15 = math.fma %14, %13, %dataResult : f32 loc(#loc49)
    %dataResult_8, %addressResult = handshake.store [%afterValue] %15, %35 : index, f32 loc(#loc49)
    %16:2 = handshake.extmemory[ld = 1, st = 0] (%arg3 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_5) {id = 0 : i32} : (index) -> (f32, none) loc(#loc30)
    %17:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (f32, none) loc(#loc30)
    %18:2 = handshake.extmemory[ld = 1, st = 0] (%arg4 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_7) {id = 2 : i32} : (index) -> (f32, none) loc(#loc30)
    %19:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_1) {id = 3 : i32} : (index) -> (f32, none) loc(#loc30)
    %20:2 = handshake.extmemory[ld = 1, st = 0] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_3) {id = 4 : i32} : (index) -> (f32, none) loc(#loc30)
    %21 = handshake.extmemory[ld = 0, st = 1] (%arg5 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_8, %addressResult) {id = 5 : i32} : (f32, index) -> none loc(#loc30)
    %22 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc36)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %17#1 : none loc(#loc36)
    %23 = handshake.constant %1 {value = 0 : index} : index loc(#loc36)
    %24 = handshake.constant %1 {value = 1 : index} : index loc(#loc36)
    %25 = arith.select %6, %24, %23 : index loc(#loc36)
    %26 = handshake.mux %25 [%falseResult_10, %trueResult] : index, none loc(#loc36)
    %27 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc36)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %20#1 : none loc(#loc36)
    %28 = handshake.mux %25 [%falseResult_12, %trueResult] : index, none loc(#loc36)
    %29 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc36)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %18#1 : none loc(#loc36)
    %30 = handshake.mux %25 [%falseResult_14, %trueResult] : index, none loc(#loc36)
    %31 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc36)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %19#1 : none loc(#loc36)
    %32 = handshake.mux %25 [%falseResult_16, %trueResult] : index, none loc(#loc36)
    %33 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc36)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %16#1 : none loc(#loc36)
    %34 = handshake.mux %25 [%falseResult_18, %trueResult] : index, none loc(#loc36)
    %35 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc36)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %21 : none loc(#loc36)
    %36 = handshake.mux %25 [%falseResult_20, %trueResult] : index, none loc(#loc36)
    %37 = handshake.join %26, %28, %30, %32, %34, %36 : none, none, none, none, none, none loc(#loc30)
    %38 = handshake.constant %37 {value = true} : i1 loc(#loc30)
    handshake.return %38 : i1 loc(#loc30)
  } loc(#loc30)
  handshake.func @_Z20runge_kutta_step_dsaPKfS0_S0_S0_S0_Pffj(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg3: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg4: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg5: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg6: f32 loc(fused<#di_subprogram4>[#loc17]), %arg7: i32 loc(fused<#di_subprogram4>[#loc17]), %arg8: none loc(fused<#di_subprogram4>[#loc17]), ...) -> none attributes {argNames = ["input_y", "input_k1", "input_k2", "input_k3", "input_k4", "output_y", "h", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg8 : none loc(#loc30)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 6.000000e+00 : f32} : f32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %0 {value = 2.000000e+00 : f32} : f32 loc(#loc2)
    %5 = arith.cmpi eq, %arg7, %1 : i32 loc(#loc40)
    %trueResult, %falseResult = handshake.cond_br %5, %0 : none loc(#loc36)
    %6 = arith.divf %arg6, %2 : f32 loc(#loc2)
    %7 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc36)
    %8 = arith.index_cast %3 : i64 to index loc(#loc36)
    %9 = arith.index_cast %arg7 : i32 to index loc(#loc36)
    %index, %willContinue = dataflow.stream %8, %7, %9 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc36)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc36)
    %dataResult, %addressResults = handshake.load [%afterValue] %16#0, %21 : index, f32 loc(#loc49)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %18#0, %30 : index, f32 loc(#loc49)
    %dataResult_2, %addressResults_3 = handshake.load [%afterValue] %19#0, %26 : index, f32 loc(#loc49)
    %10 = math.fma %dataResult_2, %4, %dataResult_0 : f32 loc(#loc49)
    %dataResult_4, %addressResults_5 = handshake.load [%afterValue] %15#0, %32 : index, f32 loc(#loc49)
    %11 = math.fma %dataResult_4, %4, %10 : f32 loc(#loc49)
    %dataResult_6, %addressResults_7 = handshake.load [%afterValue] %17#0, %28 : index, f32 loc(#loc49)
    %12 = arith.addf %11, %dataResult_6 : f32 loc(#loc49)
    %13 = dataflow.invariant %afterCond, %6 : i1, f32 -> f32 loc(#loc49)
    %14 = math.fma %13, %12, %dataResult : f32 loc(#loc49)
    %dataResult_8, %addressResult = handshake.store [%afterValue] %14, %34 : index, f32 loc(#loc49)
    %15:2 = handshake.extmemory[ld = 1, st = 0] (%arg3 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_5) {id = 0 : i32} : (index) -> (f32, none) loc(#loc30)
    %16:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (f32, none) loc(#loc30)
    %17:2 = handshake.extmemory[ld = 1, st = 0] (%arg4 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_7) {id = 2 : i32} : (index) -> (f32, none) loc(#loc30)
    %18:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_1) {id = 3 : i32} : (index) -> (f32, none) loc(#loc30)
    %19:2 = handshake.extmemory[ld = 1, st = 0] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_3) {id = 4 : i32} : (index) -> (f32, none) loc(#loc30)
    %20 = handshake.extmemory[ld = 0, st = 1] (%arg5 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_8, %addressResult) {id = 5 : i32} : (f32, index) -> none loc(#loc30)
    %21 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc36)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %16#1 : none loc(#loc36)
    %22 = handshake.constant %0 {value = 0 : index} : index loc(#loc36)
    %23 = handshake.constant %0 {value = 1 : index} : index loc(#loc36)
    %24 = arith.select %5, %23, %22 : index loc(#loc36)
    %25 = handshake.mux %24 [%falseResult_10, %trueResult] : index, none loc(#loc36)
    %26 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc36)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %19#1 : none loc(#loc36)
    %27 = handshake.mux %24 [%falseResult_12, %trueResult] : index, none loc(#loc36)
    %28 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc36)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %17#1 : none loc(#loc36)
    %29 = handshake.mux %24 [%falseResult_14, %trueResult] : index, none loc(#loc36)
    %30 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc36)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %18#1 : none loc(#loc36)
    %31 = handshake.mux %24 [%falseResult_16, %trueResult] : index, none loc(#loc36)
    %32 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc36)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %15#1 : none loc(#loc36)
    %33 = handshake.mux %24 [%falseResult_18, %trueResult] : index, none loc(#loc36)
    %34 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc36)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %20 : none loc(#loc36)
    %35 = handshake.mux %24 [%falseResult_20, %trueResult] : index, none loc(#loc36)
    %36 = handshake.join %25, %27, %29, %31, %33, %35 : none, none, none, none, none, none loc(#loc30)
    handshake.return %36 : none loc(#loc31)
  } loc(#loc30)
  func.func @_Z20runge_kutta_step_cpuPKfS0_S0_S0_S0_Pffj(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc21]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc21]), %arg2: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc21]), %arg3: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc21]), %arg4: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc21]), %arg5: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc21]), %arg6: f32 loc(fused<#di_subprogram5>[#loc21]), %arg7: i32 loc(fused<#di_subprogram5>[#loc21])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %cst = arith.constant 6.000000e+00 : f32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %cst_0 = arith.constant 2.000000e+00 : f32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg7, %c0_i32 : i32 loc(#loc41)
    scf.if %0 {
    } else {
      %1 = arith.divf %arg6, %cst : f32 loc(#loc2)
      %2 = arith.extui %arg7 : i32 to i64 loc(#loc41)
      %3 = scf.while (%arg8 = %c0_i64) : (i64) -> i64 {
        %4 = arith.index_cast %arg8 : i64 to index loc(#loc50)
        %5 = memref.load %arg0[%4] : memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
        %6 = memref.load %arg1[%4] : memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
        %7 = memref.load %arg2[%4] : memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
        %8 = math.fma %7, %cst_0, %6 : f32 loc(#loc50)
        %9 = memref.load %arg3[%4] : memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
        %10 = math.fma %9, %cst_0, %8 : f32 loc(#loc50)
        %11 = memref.load %arg4[%4] : memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
        %12 = arith.addf %10, %11 : f32 loc(#loc50)
        %13 = math.fma %1, %12, %5 : f32 loc(#loc50)
        memref.store %13, %arg5[%4] : memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
        %14 = arith.addi %arg8, %c1_i64 : i64 loc(#loc41)
        %15 = arith.cmpi ne, %14, %2 : i64 loc(#loc51)
        scf.condition(%15) %14 : i64 loc(#loc37)
      } do {
      ^bb0(%arg8: i64 loc(fused<#di_lexical_block7>[#loc22])):
        scf.yield %arg8 : i64 loc(#loc37)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc37)
    } loc(#loc37)
    return loc(#loc33)
  } loc(#loc32)
} loc(#loc)
#loc = loc("tests/app/runge_kutta_step/main.cpp":0:0)
#loc1 = loc("tests/app/runge_kutta_step/main.cpp":6:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/runge_kutta_step/main.cpp":23:0)
#loc5 = loc("tests/app/runge_kutta_step/main.cpp":24:0)
#loc6 = loc("tests/app/runge_kutta_step/main.cpp":25:0)
#loc7 = loc("tests/app/runge_kutta_step/main.cpp":26:0)
#loc8 = loc("tests/app/runge_kutta_step/main.cpp":27:0)
#loc9 = loc("tests/app/runge_kutta_step/main.cpp":31:0)
#loc10 = loc("tests/app/runge_kutta_step/main.cpp":34:0)
#loc12 = loc("tests/app/runge_kutta_step/main.cpp":38:0)
#loc13 = loc("tests/app/runge_kutta_step/main.cpp":39:0)
#loc14 = loc("tests/app/runge_kutta_step/main.cpp":40:0)
#loc15 = loc("tests/app/runge_kutta_step/main.cpp":44:0)
#loc16 = loc("tests/app/runge_kutta_step/main.cpp":46:0)
#loc18 = loc("tests/app/runge_kutta_step/runge_kutta_step.cpp":44:0)
#loc19 = loc("tests/app/runge_kutta_step/runge_kutta_step.cpp":45:0)
#loc20 = loc("tests/app/runge_kutta_step/runge_kutta_step.cpp":47:0)
#loc23 = loc("tests/app/runge_kutta_step/runge_kutta_step.cpp":29:0)
#loc24 = loc("tests/app/runge_kutta_step/runge_kutta_step.cpp":31:0)
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 44>
#loc25 = loc(fused<#di_subprogram3>[#loc1])
#loc26 = loc(fused<#di_subprogram3>[#loc9])
#loc27 = loc(fused<#di_subprogram3>[#loc10])
#loc28 = loc(fused<#di_subprogram3>[#loc15])
#loc29 = loc(fused<#di_subprogram3>[#loc16])
#loc31 = loc(fused<#di_subprogram4>[#loc20])
#loc33 = loc(fused<#di_subprogram5>[#loc24])
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 22>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 37>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 44>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 28>
#loc36 = loc(fused<#di_lexical_block6>[#loc18])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 22>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 37>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 44>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 28>
#loc38 = loc(fused<#di_lexical_block8>[#loc3])
#loc39 = loc(fused<#di_lexical_block9>[#loc11])
#loc40 = loc(fused<#di_lexical_block10>[#loc18])
#loc41 = loc(fused<#di_lexical_block11>[#loc22])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 38>
#loc42 = loc(fused<#di_lexical_block12>[#loc4])
#loc43 = loc(fused<#di_lexical_block12>[#loc5])
#loc44 = loc(fused<#di_lexical_block12>[#loc6])
#loc45 = loc(fused<#di_lexical_block12>[#loc7])
#loc46 = loc(fused<#di_lexical_block12>[#loc8])
#loc47 = loc(fused[#loc34, #loc38])
#loc48 = loc(fused[#loc35, #loc39])
#loc49 = loc(fused<#di_lexical_block14>[#loc19])
#loc50 = loc(fused<#di_lexical_block15>[#loc23])
#loc51 = loc(fused[#loc37, #loc41])
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 38>
#loc52 = loc(fused<#di_lexical_block16>[#loc12])
#loc53 = loc(fused<#di_lexical_block17>[#loc13])
#loc54 = loc(fused<#di_lexical_block17>[#loc14])
