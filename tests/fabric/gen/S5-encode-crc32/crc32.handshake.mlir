#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/crc32/crc32.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/crc32/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/crc32/crc32.cpp":15:0)
#loc3 = loc("tests/app/crc32/crc32.cpp":21:0)
#loc5 = loc("tests/app/crc32/crc32.cpp":25:0)
#loc11 = loc("tests/app/crc32/crc32.cpp":44:0)
#loc21 = loc("tests/app/crc32/main.cpp":10:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 21>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 52>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 10>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 21>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 52>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 8192, elements = #llvm.di_subrange<count = 256 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file, line = 21>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 52>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "crc", file = #di_file, line = 19, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 21, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram1, name = "crc", file = #di_file, line = 48, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 52, type = #di_derived_type1>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 10, type = #di_derived_type1>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_checksum", file = #di_file1, line = 15, type = #di_derived_type1>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_checksum", file = #di_file1, line = 16, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 25>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 56>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 17, arg = 3, type = #di_derived_type2>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "polynomial", file = #di_file, line = 18, type = #di_derived_type2>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block5, name = "data", file = #di_file, line = 22, type = #di_derived_type1>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 46, arg = 3, type = #di_derived_type2>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "polynomial", file = #di_file, line = 47, type = #di_derived_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "data", file = #di_file, line = 53, type = #di_derived_type1>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 6, type = #di_derived_type2>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file1, line = 9, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 25>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 56>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_checksum", file = #di_file, line = 16, arg = 2, type = #di_derived_type5>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "byte_idx", file = #di_file, line = 25, type = #di_derived_type1>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_checksum", file = #di_file, line = 45, arg = 2, type = #di_derived_type5>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "byte_idx", file = #di_file, line = 56, type = #di_derived_type1>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable13, #di_local_variable14, #di_local_variable4, #di_local_variable5, #di_local_variable6>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 25>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 56>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 10>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_data", file = #di_file, line = 15, arg = 1, type = #di_derived_type6>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_data", file = #di_file, line = 44, arg = 1, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 29>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 60>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_lexical_block11, name = "byte", file = #di_file, line = 26, type = #di_derived_type1>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_lexical_block12, name = "byte", file = #di_file, line = 57, type = #di_derived_type1>
#loc33 = loc(fused<#di_lexical_block13>[#loc21])
#di_local_variable23 = #llvm.di_local_variable<scope = #di_lexical_block15, name = "bit", file = #di_file, line = 29, type = #di_derived_type1>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_lexical_block16, name = "bit", file = #di_file, line = 60, type = #di_derived_type1>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "crc32_cpu", linkageName = "_Z9crc32_cpuPKjPjj", file = #di_file, line = 15, scopeLine = 17, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable19, #di_local_variable15, #di_local_variable7, #di_local_variable8, #di_local_variable, #di_local_variable1, #di_local_variable9, #di_local_variable16, #di_local_variable21, #di_local_variable23>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "crc32_dsa", linkageName = "_Z9crc32_dsaPKjPjj", file = #di_file, line = 44, scopeLine = 46, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable20, #di_local_variable17, #di_local_variable10, #di_local_variable11, #di_local_variable2, #di_local_variable3, #di_local_variable12, #di_local_variable18, #di_local_variable22, #di_local_variable24>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 21>
#loc38 = loc(fused<#di_subprogram4>[#loc1])
#loc41 = loc(fused<#di_subprogram5>[#loc11])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 21>
#loc44 = loc(fused<#di_lexical_block19>[#loc3])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file, line = 21>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file, line = 25>
#loc51 = loc(fused<#di_lexical_block25>[#loc5])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu", loom.global_constant..crctable.2 = {data = dense<"0x00000000963007772C610EEEBA51099919C46D078FF46A7035A563E9A395649E3288DB0EA4B8DC791EE9D5E088D9D2972B4CB609BD7CB17E072DB8E7911DBF906410B71DF220B06A4871B9F3DE41BE847DD4DA1AEBE4DD6D51B5D4F4C785D38356986C13C0A86B647AF962FDECC9658A4F5C0114D96C0663633D0FFAF50D088DC8206E3B5E10694CE44160D5727167A2D1E4033C47D4044BFD850DD26BB50AA5FAA8B5356C98B242D6C9BBDB40F9BCACE36CD832755CDF45CF0DD6DC593DD1ABAC30D9263A00DE518051D7C81661D0BFB5F4B42123C4B3569995BACF0FA5BDB89EB802280888055FB2D90CC624E90BB1877C6F2F114C6858AB1D61C13D2D66B69041DC760671DB01BC20D2982A10D5EF8985B1711FB5B606A5E4BF9F33D4B8E8A2C9077834F9000F8EA8099618980EE1BB0D6A7F2D3D6D08976C6491015C63E6F4516B6B62616C1CD83065854E0062F2ED95066C7BA5011BC1F4088257C40FF5C6D9B06550E9B712EAB8BE8B7C88B9FCDF1DDD62492DDA15F37CD38C654CD4FB5861B24DCE51B53A7400BCA3E230BBD441A5DF4AD795D83D6DC4D1A4FBF4D6D36AE96943FCD96E34468867ADD0B860DA732D0444E51D03335F4C0AAAC97C0DDD3C710550AA41022710100BBE86200CC925B56857B3856F2009D466B99FE461CE0EF9DE5E98C9D9292298D0B0B4A8D7C7173DB359810DB42E3B5CBDB7AD6CBAC02083B8EDB6B3BF9A0CE2B6039AD2B1743947D5EAAF77D29D1526DB048316DC73120B63E3843B64943E6A6D0DA85A6A7A0BCF0EE49DFF099327AE000AB19E077D44930FF0D2A3088768F2011EFEC206695D5762F7CB67658071366C19E7066B6E761BD4FEE02BD3895A7ADA10CC4ADD676FDFB9F9F9EFBE8E43BEB717D58EB060E8A3D6D67E93D1A1C4C2D83852F2DF4FF167BBD16757BCA6DD06B53F4B36B248DA2B0DD84C1B0AAFF64A0336607A0441C3EF60DF55DF67A8EF8E6E3179BE69468CB361CB1A8366BCA0D26F2536E2685295770CCC03470BBBB91602222F260555BE3BBAC5280BBDB2925AB42B046AB35CA7FFD7C231CFD0B58B9ED92C1DAEDE5BB0C2649B26F263EC9CA36A750A936D02A906099C3F360EEB8567077213570005824ABF95147AB8E2AE2BB17B381BB60C9B8ED2920DBED5E5B7EFDC7C21DFDB0BD4D2D38642E2D4F1F8B3DD686E83DA1FCD16BE815B26B9F6E177B06F7747B718E65A0888706A0FFFCA3B06665C0B0111FF9E658F69AE62F8D3FF6B6145CF6C1678E20AA0EED20DD75483044EC2B30339612667A7F71660D04D476949DB776E3E4A6AD1AEDC5AD6D9660BDF40F03BD83753AEBCA9C59EBBDE7FCFB247E9FFB5301CF2BDBD8AC2BACA3093B353A6A3B4240536D0BA9306D7CD2957DE54BF67D9232E7A66B3B84A61C4021B685D942B6F2A37BE0BB4A18E0CC31BDF055A8DEF022D"> : tensor<256xi32>, type = memref<256xi32>}} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<26xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 99, 114, 99, 51, 50, 47, 99, 114, 99, 51, 50, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".crctable.2" : memref<256xi32> = dense<"0x00000000963007772C610EEEBA51099919C46D078FF46A7035A563E9A395649E3288DB0EA4B8DC791EE9D5E088D9D2972B4CB609BD7CB17E072DB8E7911DBF906410B71DF220B06A4871B9F3DE41BE847DD4DA1AEBE4DD6D51B5D4F4C785D38356986C13C0A86B647AF962FDECC9658A4F5C0114D96C0663633D0FFAF50D088DC8206E3B5E10694CE44160D5727167A2D1E4033C47D4044BFD850DD26BB50AA5FAA8B5356C98B242D6C9BBDB40F9BCACE36CD832755CDF45CF0DD6DC593DD1ABAC30D9263A00DE518051D7C81661D0BFB5F4B42123C4B3569995BACF0FA5BDB89EB802280888055FB2D90CC624E90BB1877C6F2F114C6858AB1D61C13D2D66B69041DC760671DB01BC20D2982A10D5EF8985B1711FB5B606A5E4BF9F33D4B8E8A2C9077834F9000F8EA8099618980EE1BB0D6A7F2D3D6D08976C6491015C63E6F4516B6B62616C1CD83065854E0062F2ED95066C7BA5011BC1F4088257C40FF5C6D9B06550E9B712EAB8BE8B7C88B9FCDF1DDD62492DDA15F37CD38C654CD4FB5861B24DCE51B53A7400BCA3E230BBD441A5DF4AD795D83D6DC4D1A4FBF4D6D36AE96943FCD96E34468867ADD0B860DA732D0444E51D03335F4C0AAAC97C0DDD3C710550AA41022710100BBE86200CC925B56857B3856F2009D466B99FE461CE0EF9DE5E98C9D9292298D0B0B4A8D7C7173DB359810DB42E3B5CBDB7AD6CBAC02083B8EDB6B3BF9A0CE2B6039AD2B1743947D5EAAF77D29D1526DB048316DC73120B63E3843B64943E6A6D0DA85A6A7A0BCF0EE49DFF099327AE000AB19E077D44930FF0D2A3088768F2011EFEC206695D5762F7CB67658071366C19E7066B6E761BD4FEE02BD3895A7ADA10CC4ADD676FDFB9F9F9EFBE8E43BEB717D58EB060E8A3D6D67E93D1A1C4C2D83852F2DF4FF167BBD16757BCA6DD06B53F4B36B248DA2B0DD84C1B0AAFF64A0336607A0441C3EF60DF55DF67A8EF8E6E3179BE69468CB361CB1A8366BCA0D26F2536E2685295770CCC03470BBBB91602222F260555BE3BBAC5280BBDB2925AB42B046AB35CA7FFD7C231CFD0B58B9ED92C1DAEDE5BB0C2649B26F263EC9CA36A750A936D02A906099C3F360EEB8567077213570005824ABF95147AB8E2AE2BB17B381BB60C9B8ED2920DBED5E5B7EFDC7C21DFDB0BD4D2D38642E2D4F1F8B3DD686E83DA1FCD16BE815B26B9F6E177B06F7747B718E65A0888706A0FFFCA3B06665C0B0111FF9E658F69AE62F8D3FF6B6145CF6C1678E20AA0EED20DD75483044EC2B30339612667A7F71660D04D476949DB776E3E4A6AD1AEDC5AD6D9660BDF40F03BD83753AEBCA9C59EBBDE7FCFB247E9FFB5301CF2BDBD8AC2BACA3093B353A6A3B4240536D0BA9306D7CD2957DE54BF67D9232E7A66B3B84A61C4021B685D942B6F2A37BE0BB4A18E0CC31BDF055A8DEF022D"> loc(#loc)
  memref.global constant @str : memref<14xi8> = dense<[99, 114, 99, 51, 50, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<14xi8> = dense<[99, 114, 99, 51, 50, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z9crc32_cpuPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: i32 loc(fused<#di_subprogram4>[#loc1])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c4_i32 = arith.constant 4 : i32 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %c8_i32 = arith.constant 8 : i32 loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc2)
    %c255_i32 = arith.constant 255 : i32 loc(#loc2)
    %0 = memref.get_global @".crctable.2" : memref<256xi32> loc(#loc2)
    %1 = arith.cmpi eq, %arg2, %c0_i32 : i32 loc(#loc46)
    %2 = scf.if %1 -> (i32) {
      scf.yield %c0_i32 : i32 loc(#loc44)
    } else {
      %3 = arith.extui %arg2 : i32 to i64 loc(#loc46)
      %4:2 = scf.while (%arg3 = %c0_i64, %arg4 = %c-1_i32) : (i64, i32) -> (i64, i32) {
        %6 = arith.index_cast %arg3 : i64 to index loc(#loc48)
        %7 = memref.load %arg0[%6] : memref<?xi32, strided<[1], offset: ?>> loc(#loc48)
        %8:2 = scf.while (%arg5 = %arg4, %arg6 = %c0_i32) : (i32, i32) -> (i32, i32) {
          %11 = arith.shrui %arg5, %c8_i32 : i32 loc(#loc59)
          %12 = arith.shli %arg6, %c3_i32 : i32 loc(#loc54)
          %13 = arith.shrui %7, %12 : i32 loc(#loc54)
          %14 = arith.xori %13, %arg5 : i32 loc(#loc55)
          %15 = arith.andi %14, %c255_i32 : i32 loc(#loc59)
          %16 = arith.extui %15 : i32 to i64 loc(#loc59)
          %17 = arith.index_cast %16 : i64 to index loc(#loc59)
          %18 = memref.load %0[%17] : memref<256xi32> loc(#loc59)
          %19 = arith.xori %11, %18 : i32 loc(#loc59)
          %20 = arith.addi %arg6, %c1_i32 : i32 loc(#loc53)
          %21 = arith.cmpi ne, %20, %c4_i32 : i32 loc(#loc56)
          scf.condition(%21) %19, %20 : i32, i32 loc(#loc51)
        } do {
        ^bb0(%arg5: i32 loc(fused<#di_lexical_block25>[#loc5]), %arg6: i32 loc(fused<#di_lexical_block25>[#loc5])):
          scf.yield %arg5, %arg6 : i32, i32 loc(#loc51)
        } attributes {loom.stream = {cmp_on_update = true, iv = 1 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc51)
        %9 = arith.addi %arg3, %c1_i64 : i64 loc(#loc46)
        %10 = arith.cmpi ne, %9, %3 : i64 loc(#loc49)
        scf.condition(%10) %9, %8#0 : i64, i32 loc(#loc44)
      } do {
      ^bb0(%arg3: i64 loc(fused<#di_lexical_block19>[#loc3]), %arg4: i32 loc(fused<#di_lexical_block19>[#loc3])):
        scf.yield %arg3, %arg4 : i64, i32 loc(#loc44)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc44)
      %5 = arith.xori %4#1, %c-1_i32 : i32 loc(#loc39)
      scf.yield %5 : i32 loc(#loc39)
    } loc(#loc44)
    memref.store %2, %arg1[%c0] : memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    return loc(#loc40)
  } loc(#loc38)
  handshake.func @_Z9crc32_dsaPKjPjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc11]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc11]), %arg2: i32 loc(fused<#di_subprogram5>[#loc11]), %arg3: i1 loc(fused<#di_subprogram5>[#loc11]), ...) -> i1 attributes {argNames = ["input_data", "output_checksum", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc41)
    %1 = handshake.join %0 : none loc(#loc41)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 4 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : index} : index loc(#loc45)
    %5 = handshake.constant %1 {value = 8 : i32} : i32 loc(#loc2)
    %6 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %7 = handshake.constant %1 {value = -1 : i32} : i32 loc(#loc2)
    %8 = handshake.constant %1 {value = 3 : i32} : i32 loc(#loc2)
    %9 = handshake.constant %1 {value = 255 : i32} : i32 loc(#loc2)
    %10 = arith.cmpi eq, %arg2, %2 : i32 loc(#loc47)
    %trueResult, %falseResult = handshake.cond_br %10, %1 : none loc(#loc45)
    %11 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc45)
    %12 = arith.index_cast %6 : i64 to index loc(#loc45)
    %13 = arith.index_cast %arg2 : i32 to index loc(#loc45)
    %index, %willContinue = dataflow.stream %12, %11, %13 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc45)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc45)
    %14 = dataflow.carry %willContinue, %7, %falseResult_11 : i1, i32, i32 -> i32 loc(#loc45)
    %afterValue_0, %afterCond_1 = dataflow.gate %14, %willContinue : i32, i1 -> i32, i1 loc(#loc45)
    handshake.sink %afterCond_1 : i1 loc(#loc45)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %14 : i32 loc(#loc45)
    %15 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc45)
    %dataResult, %addressResults = handshake.load [%afterValue] %34#0, %37 : index, i32 loc(#loc50)
    %16 = handshake.constant %15 {value = 1 : index} : index loc(#loc52)
    %17 = arith.index_cast %2 : i32 to index loc(#loc52)
    %18 = arith.index_cast %3 : i32 to index loc(#loc52)
    %index_4, %willContinue_5 = dataflow.stream %17, %16, %18 {step_op = "+=", stop_cond = "!="} loc(#loc52)
    %afterValue_6, %afterCond_7 = dataflow.gate %index_4, %willContinue_5 : index, i1 -> index, i1 loc(#loc52)
    %19 = dataflow.carry %willContinue_5, %afterValue_0, %29 : i1, i32, i32 -> i32 loc(#loc52)
    %afterValue_8, %afterCond_9 = dataflow.gate %19, %willContinue_5 : i32, i1 -> i32, i1 loc(#loc52)
    handshake.sink %afterCond_9 : i1 loc(#loc52)
    %trueResult_10, %falseResult_11 = handshake.cond_br %willContinue_5, %19 : i32 loc(#loc52)
    %20 = arith.index_cast %afterValue_6 : index to i32 loc(#loc52)
    %21 = arith.shrui %afterValue_8, %5 : i32 loc(#loc60)
    %22 = arith.shli %20, %8 : i32 loc(#loc57)
    %23 = dataflow.invariant %afterCond_7, %dataResult : i1, i32 -> i32 loc(#loc57)
    %24 = arith.shrui %23, %22 : i32 loc(#loc57)
    %25 = arith.xori %24, %afterValue_8 : i32 loc(#loc58)
    %26 = arith.andi %25, %9 : i32 loc(#loc60)
    %27 = arith.extui %26 : i32 to i64 loc(#loc60)
    %28 = arith.index_cast %27 : i64 to index loc(#loc60)
    %dataResult_12, %addressResults_13 = handshake.load [%28] %35#0, %40 : index, i32 loc(#loc60)
    %29 = arith.xori %21, %dataResult_12 : i32 loc(#loc60)
    %30 = arith.xori %falseResult_3, %7 : i32 loc(#loc42)
    %31 = handshake.constant %1 {value = 1 : index} : index loc(#loc45)
    %32 = arith.select %10, %31, %4 : index loc(#loc45)
    %33 = handshake.mux %32 [%30, %2] : index, i32 loc(#loc45)
    %dataResult_14, %addressResult = handshake.store [%4] %33, %1 : index, i32 loc(#loc42)
    %34:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc41)
    %35:2 = handshake.memory[ld = 1, st = 0] (%addressResults_13) {id = 1 : i32, loom.global_constant = ".crctable.2", loom.global_memref = @".crctable.2", lsq = false} : memref<256xi32>, (index) -> (i32, none) loc(#loc2)
    %36 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_14, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc41)
    %37 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc45)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %34#1 : none loc(#loc45)
    %38 = handshake.mux %32 [%falseResult_16, %trueResult] : index, none loc(#loc45)
    %39 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc45)
    %40 = dataflow.carry %willContinue_5, %39, %trueResult_17 : i1, none, none -> none loc(#loc52)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue_5, %35#1 : none loc(#loc52)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %falseResult_18 : none loc(#loc45)
    %41 = handshake.mux %32 [%falseResult_20, %trueResult] : index, none loc(#loc45)
    %42 = handshake.join %38, %36, %41 : none, none, none loc(#loc41)
    %43 = handshake.constant %42 {value = true} : i1 loc(#loc41)
    handshake.return %43 : i1 loc(#loc41)
  } loc(#loc41)
  handshake.func @_Z9crc32_dsaPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc11]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc11]), %arg2: i32 loc(fused<#di_subprogram5>[#loc11]), %arg3: none loc(fused<#di_subprogram5>[#loc11]), ...) -> none attributes {argNames = ["input_data", "output_checksum", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc41)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 4 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : index} : index loc(#loc45)
    %4 = handshake.constant %0 {value = 8 : i32} : i32 loc(#loc2)
    %5 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %6 = handshake.constant %0 {value = -1 : i32} : i32 loc(#loc2)
    %7 = handshake.constant %0 {value = 3 : i32} : i32 loc(#loc2)
    %8 = handshake.constant %0 {value = 255 : i32} : i32 loc(#loc2)
    %9 = arith.cmpi eq, %arg2, %1 : i32 loc(#loc47)
    %trueResult, %falseResult = handshake.cond_br %9, %0 : none loc(#loc45)
    %10 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc45)
    %11 = arith.index_cast %5 : i64 to index loc(#loc45)
    %12 = arith.index_cast %arg2 : i32 to index loc(#loc45)
    %index, %willContinue = dataflow.stream %11, %10, %12 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc45)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc45)
    %13 = dataflow.carry %willContinue, %6, %falseResult_11 : i1, i32, i32 -> i32 loc(#loc45)
    %afterValue_0, %afterCond_1 = dataflow.gate %13, %willContinue : i32, i1 -> i32, i1 loc(#loc45)
    handshake.sink %afterCond_1 : i1 loc(#loc45)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %13 : i32 loc(#loc45)
    %14 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc45)
    %dataResult, %addressResults = handshake.load [%afterValue] %33#0, %36 : index, i32 loc(#loc50)
    %15 = handshake.constant %14 {value = 1 : index} : index loc(#loc52)
    %16 = arith.index_cast %1 : i32 to index loc(#loc52)
    %17 = arith.index_cast %2 : i32 to index loc(#loc52)
    %index_4, %willContinue_5 = dataflow.stream %16, %15, %17 {step_op = "+=", stop_cond = "!="} loc(#loc52)
    %afterValue_6, %afterCond_7 = dataflow.gate %index_4, %willContinue_5 : index, i1 -> index, i1 loc(#loc52)
    %18 = dataflow.carry %willContinue_5, %afterValue_0, %28 : i1, i32, i32 -> i32 loc(#loc52)
    %afterValue_8, %afterCond_9 = dataflow.gate %18, %willContinue_5 : i32, i1 -> i32, i1 loc(#loc52)
    handshake.sink %afterCond_9 : i1 loc(#loc52)
    %trueResult_10, %falseResult_11 = handshake.cond_br %willContinue_5, %18 : i32 loc(#loc52)
    %19 = arith.index_cast %afterValue_6 : index to i32 loc(#loc52)
    %20 = arith.shrui %afterValue_8, %4 : i32 loc(#loc60)
    %21 = arith.shli %19, %7 : i32 loc(#loc57)
    %22 = dataflow.invariant %afterCond_7, %dataResult : i1, i32 -> i32 loc(#loc57)
    %23 = arith.shrui %22, %21 : i32 loc(#loc57)
    %24 = arith.xori %23, %afterValue_8 : i32 loc(#loc58)
    %25 = arith.andi %24, %8 : i32 loc(#loc60)
    %26 = arith.extui %25 : i32 to i64 loc(#loc60)
    %27 = arith.index_cast %26 : i64 to index loc(#loc60)
    %dataResult_12, %addressResults_13 = handshake.load [%27] %34#0, %39 : index, i32 loc(#loc60)
    %28 = arith.xori %20, %dataResult_12 : i32 loc(#loc60)
    %29 = arith.xori %falseResult_3, %6 : i32 loc(#loc42)
    %30 = handshake.constant %0 {value = 1 : index} : index loc(#loc45)
    %31 = arith.select %9, %30, %3 : index loc(#loc45)
    %32 = handshake.mux %31 [%29, %1] : index, i32 loc(#loc45)
    %dataResult_14, %addressResult = handshake.store [%3] %32, %0 : index, i32 loc(#loc42)
    %33:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc41)
    %34:2 = handshake.memory[ld = 1, st = 0] (%addressResults_13) {id = 1 : i32, loom.global_constant = ".crctable.2", loom.global_memref = @".crctable.2", lsq = false} : memref<256xi32>, (index) -> (i32, none) loc(#loc2)
    %35 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_14, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc41)
    %36 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc45)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %33#1 : none loc(#loc45)
    %37 = handshake.mux %31 [%falseResult_16, %trueResult] : index, none loc(#loc45)
    %38 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc45)
    %39 = dataflow.carry %willContinue_5, %38, %trueResult_17 : i1, none, none -> none loc(#loc52)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue_5, %34#1 : none loc(#loc52)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %falseResult_18 : none loc(#loc45)
    %40 = handshake.mux %31 [%falseResult_20, %trueResult] : index, none loc(#loc45)
    %41 = handshake.join %37, %35, %40 : none, none, none loc(#loc41)
    handshake.return %41 : none loc(#loc43)
  } loc(#loc41)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc28)
    %false = arith.constant false loc(#loc28)
    %0 = seq.const_clock  low loc(#loc28)
    %c0 = arith.constant 0 : index loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c305419896_i32 = arith.constant 305419896 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c256_i64 = arith.constant 256 : i64 loc(#loc2)
    %c256_i32 = arith.constant 256 : i32 loc(#loc2)
    %1 = memref.get_global @str.2 : memref<14xi8> loc(#loc2)
    %2 = memref.get_global @str : memref<14xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1xi32> loc(#loc2)
    %3 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %13 = arith.trunci %arg0 : i64 to i32 loc(#loc36)
      %14 = arith.muli %13, %c305419896_i32 : i32 loc(#loc36)
      %15 = arith.index_cast %arg0 : i64 to index loc(#loc36)
      memref.store %14, %alloca[%15] : memref<256xi32> loc(#loc36)
      %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc35)
      %17 = arith.cmpi ne, %16, %c256_i64 : i64 loc(#loc37)
      scf.condition(%17) %16 : i64 loc(#loc33)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block13>[#loc21])):
      scf.yield %arg0 : i64 loc(#loc33)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc33)
    %cast = memref.cast %alloca : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %cast_2 = memref.cast %alloca_0 : memref<1xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    call @_Z9crc32_cpuPKjPjj(%cast, %cast_2, %c256_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc29)
    %cast_3 = memref.cast %alloca_1 : memref<1xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c256_i32, %true : i32 loc(#loc30)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc30)
    %4 = handshake.esi_instance @_Z9crc32_dsaPKjPjj_esi "_Z9crc32_dsaPKjPjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc30)
    %rawOutput, %valid = esi.unwrap.vr %4, %true : i1 loc(#loc30)
    %5 = memref.load %alloca_0[%c0] : memref<1xi32> loc(#loc34)
    %6 = memref.load %alloca_1[%c0] : memref<1xi32> loc(#loc34)
    %7 = arith.cmpi ne, %5, %6 : i32 loc(#loc34)
    %cast_10 = memref.cast %1 : memref<14xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc31)
    %cast_11 = memref.cast %2 : memref<14xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc31)
    %8 = arith.select %7, %cast_10, %cast_11 : memref<?xi8, strided<[1], offset: ?>> loc(#loc31)
    %9 = arith.extui %7 : i1 to i32 loc(#loc31)
    %intptr = memref.extract_aligned_pointer_as_index %8 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc31)
    %10 = arith.index_cast %intptr : index to i64 loc(#loc31)
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc31)
    %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc31)
    return %9 : i32 loc(#loc32)
  } loc(#loc28)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/crc32/crc32.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/crc32/crc32.cpp":22:0)
#loc6 = loc("tests/app/crc32/crc32.cpp":30:0)
#loc7 = loc("tests/app/crc32/crc32.cpp":26:0)
#loc8 = loc("tests/app/crc32/crc32.cpp":27:0)
#loc9 = loc("tests/app/crc32/crc32.cpp":39:0)
#loc10 = loc("tests/app/crc32/crc32.cpp":40:0)
#loc12 = loc("tests/app/crc32/crc32.cpp":52:0)
#loc13 = loc("tests/app/crc32/crc32.cpp":53:0)
#loc14 = loc("tests/app/crc32/crc32.cpp":56:0)
#loc15 = loc("tests/app/crc32/crc32.cpp":61:0)
#loc16 = loc("tests/app/crc32/crc32.cpp":57:0)
#loc17 = loc("tests/app/crc32/crc32.cpp":58:0)
#loc18 = loc("tests/app/crc32/crc32.cpp":70:0)
#loc19 = loc("tests/app/crc32/crc32.cpp":71:0)
#loc20 = loc("tests/app/crc32/main.cpp":5:0)
#loc22 = loc("tests/app/crc32/main.cpp":11:0)
#loc23 = loc("tests/app/crc32/main.cpp":19:0)
#loc24 = loc("tests/app/crc32/main.cpp":22:0)
#loc25 = loc("tests/app/crc32/main.cpp":25:0)
#loc26 = loc("tests/app/crc32/main.cpp":0:0)
#loc27 = loc("tests/app/crc32/main.cpp":32:0)
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 25>
#loc28 = loc(fused<#di_subprogram3>[#loc20])
#loc29 = loc(fused<#di_subprogram3>[#loc23])
#loc30 = loc(fused<#di_subprogram3>[#loc24])
#loc31 = loc(fused<#di_subprogram3>[#loc26])
#loc32 = loc(fused<#di_subprogram3>[#loc27])
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file1, line = 10>
#loc34 = loc(fused<#di_lexical_block14>[#loc25])
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 10>
#loc35 = loc(fused<#di_lexical_block17>[#loc21])
#loc36 = loc(fused<#di_lexical_block18>[#loc22])
#loc37 = loc(fused[#loc33, #loc35])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 52>
#loc39 = loc(fused<#di_subprogram4>[#loc9])
#loc40 = loc(fused<#di_subprogram4>[#loc10])
#loc42 = loc(fused<#di_subprogram5>[#loc18])
#loc43 = loc(fused<#di_subprogram5>[#loc19])
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 52>
#loc45 = loc(fused<#di_lexical_block20>[#loc12])
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file, line = 52>
#loc46 = loc(fused<#di_lexical_block21>[#loc3])
#loc47 = loc(fused<#di_lexical_block22>[#loc12])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file, line = 56>
#loc48 = loc(fused<#di_lexical_block23>[#loc4])
#loc49 = loc(fused[#loc44, #loc46])
#loc50 = loc(fused<#di_lexical_block24>[#loc13])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file, line = 25>
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file, line = 56>
#loc52 = loc(fused<#di_lexical_block26>[#loc14])
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file, line = 25>
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file, line = 56>
#loc53 = loc(fused<#di_lexical_block27>[#loc5])
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block29, file = #di_file, line = 29>
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file, line = 60>
#loc54 = loc(fused<#di_lexical_block29>[#loc7])
#loc55 = loc(fused<#di_lexical_block29>[#loc8])
#loc56 = loc(fused[#loc51, #loc53])
#loc57 = loc(fused<#di_lexical_block30>[#loc16])
#loc58 = loc(fused<#di_lexical_block30>[#loc17])
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_lexical_block31, file = #di_file, line = 29>
#di_lexical_block34 = #llvm.di_lexical_block<scope = #di_lexical_block32, file = #di_file, line = 60>
#di_lexical_block35 = #llvm.di_lexical_block<scope = #di_lexical_block33, file = #di_file, line = 29>
#di_lexical_block36 = #llvm.di_lexical_block<scope = #di_lexical_block34, file = #di_file, line = 60>
#di_lexical_block37 = #llvm.di_lexical_block<scope = #di_lexical_block35, file = #di_file, line = 30>
#di_lexical_block38 = #llvm.di_lexical_block<scope = #di_lexical_block36, file = #di_file, line = 61>
#loc59 = loc(fused<#di_lexical_block37>[#loc6])
#loc60 = loc(fused<#di_lexical_block38>[#loc15])
