// Dual 3x3 lattice mesh: native PEs (Mesh A) + temporal PEs (Mesh B)
// with bidirectional cross-mesh connections via add_tag / del_tag.
//
// Mesh A (native, !dataflow.bits<32>):
//       col0        col1        col2
// row0: SA00 ------ SA01 ------ SA02 ----add_tag--->  SB00
//        |  CellA00  |  CellA01  |                      |
// row1: SA10 ------ SA11 ------ SA12 ----add_tag--->  SB10
//        |  CellA10  |  CellA11  |                      |
// row2: SA20 ------ SA21 ------ SA22 ----add_tag--->  SB20
//
// Mesh B (tagged, !dataflow.tagged<!dataflow.bits<32>, i4>):
//       col0        col1        col2
// row0: SB00 ------ SB01 ------ SB02 --del_tag+fifo->  SA02
//        |  CellB00  |  CellB01  |
// row1: SB10 ------ SB11 ------ SB12 --del_tag+fifo->  SA12
//        |  CellB10  |  CellB11  |
// row2: SB20 ------ SB21 ------ SB22 --del_tag+fifo->  SA22
//
// Native PEs:
//   CellA00: pe_addi (i32 add)    CellA01: pe_addi (i32 add)
//   CellA10: pe_muli (i32 mul)    CellA11: pe_muli (i32 mul)
//
// Temporal PEs (add + mul FUs each, tag-dispatched):
//   CellB00: tpe_alu    CellB01: tpe_alu
//   CellB10: tpe_alu    CellB11: tpe_alu
//
// PE connections per cell (2-in 1-out):
//   input 0 <- top-left switch, input 1 <- bottom-left switch
//   output 0 -> bottom-right switch
//
// East/South edges: direct.  West/North edges: via FIFO (breaks loops).
// A->B cross: direct (add_tag).  B->A cross: del_tag then FIFO.
//
// Module I/O:
//   inputs:  in0 -> SA00, in1 -> SA02, in2 -> SA20
//   outputs: SA22 -> out0, SA12 -> out1

module {
  // ---- PE Definitions ----

  fabric.pe @pe_addi(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg0, %arg1 : i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_muli(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.muli %arg0, %arg1 : i32
    fabric.yield %0 : i32
  }

  // Temporal PE: 2 FUs (add, mul), tag-dispatched, 4 instruction slots
  fabric.temporal_pe @tpe_alu(
      %arg0: !dataflow.tagged<!dataflow.bits<32>, i4>,
      %arg1: !dataflow.tagged<!dataflow.bits<32>, i4>
  ) [num_register = 0, num_instruction = 4, reg_fifo_depth = 0]
    -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
    fabric.pe @fu_add(%a: i32, %b: i32) -> (i32) {
      %r = arith.addi %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.pe @fu_mul(%a: i32, %b: i32) -> (i32) {
      %r = arith.muli %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.yield
  }

  fabric.module @mixed_temporal(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>,
      %in2: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>, !dataflow.bits<32>) {

    // ======== Mesh A Switches (native, bits<32>) ========

    // SA00 [3in, 3out]
    //   in:  module %in0, fifo<-SA01(W), fifo<-SA10(N)
    //   out: #0 E->SA01, #1 S->SA10, #2 PE CellA00 in0
    %sa00:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in0, %fifo_sa01_sa00, %fifo_sa10_sa00
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SA01 [3in, 4out]
    //   in:  SA00#0(E), fifo<-SA02(W), fifo<-SA11(N)
    //   out: #0 W->fifo->SA00, #1 E->SA02, #2 S->SA11, #3 PE CellA01 in0
    %sa01:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sa00#0, %fifo_sa02_sa01, %fifo_sa11_sa01
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SA02 [4in, 3out]
    //   in:  module %in1, SA01#1(E), fifo<-SA12(N), cross<-SB02(del_tag+fifo)
    //   out: #0 W->fifo->SA01, #1 S->SA12, #2 cross->SB00(add_tag)
    %sa02:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in1, %sa01#1, %fifo_sa12_sa02, %cross_b02_a02
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SA10 [3in, 5out]
    //   in:  SA00#1(S), fifo<-SA11(W), fifo<-SA20(N)
    //   out: #0 N->fifo->SA00, #1 E->SA11, #2 S->SA20,
    //        #3 PE CellA00 in1, #4 PE CellA10 in0
    %sa10:5 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sa00#1, %fifo_sa11_sa10, %fifo_sa20_sa10
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>,
          !dataflow.bits<32>, !dataflow.bits<32>

    // SA11 [5in, 6out]
    //   in:  SA01#2(S), SA10#1(E), fifo<-SA12(W), fifo<-SA21(N), PE CellA00 out
    //   out: #0 N->fifo->SA01, #1 W->fifo->SA10, #2 E->SA12, #3 S->SA21,
    //        #4 PE CellA01 in1, #5 PE CellA11 in0
    %sa11:6 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sa01#2, %sa10#1, %fifo_sa12_sa11, %fifo_sa21_sa11, %pe_a00
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>,
          !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SA12 [5in, 5out]
    //   in:  SA02#1(S), SA11#2(E), fifo<-SA22(N), PE CellA01 out,
    //        cross<-SB12(del_tag+fifo)
    //   out: #0 N->fifo->SA02, #1 W->fifo->SA11, #2 S->SA22,
    //        #3 module out1, #4 cross->SB10(add_tag)
    %sa12:5 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sa02#1, %sa11#2, %fifo_sa22_sa12, %pe_a01, %cross_b12_a12
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>,
          !dataflow.bits<32>, !dataflow.bits<32>

    // SA20 [3in, 3out]
    //   in:  module %in2, SA10#2(S), fifo<-SA21(W)
    //   out: #0 N->fifo->SA10, #1 E->SA21, #2 PE CellA10 in1
    %sa20:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in2, %sa10#2, %fifo_sa21_sa20
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SA21 [4in, 4out]
    //   in:  SA11#3(S), SA20#1(E), fifo<-SA22(W), PE CellA10 out
    //   out: #0 N->fifo->SA11, #1 W->fifo->SA20, #2 E->SA22, #3 PE CellA11 in1
    %sa21:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sa11#3, %sa20#1, %fifo_sa22_sa21, %pe_a10
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SA22 [4in, 4out]
    //   in:  SA12#2(S), SA21#2(E), PE CellA11 out, cross<-SB22(del_tag+fifo)
    //   out: #0 N->fifo->SA12, #1 W->fifo->SA21, #2 module out0,
    //        #3 cross->SB20(add_tag)
    %sa22:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sa12#2, %sa21#2, %pe_a11, %cross_b22_a22
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // ======== Mesh A FIFOs (backward: West and North) ========

    // West FIFOs
    %fifo_sa01_sa00 = fabric.fifo [depth = 2] %sa01#0 : !dataflow.bits<32>
    %fifo_sa02_sa01 = fabric.fifo [depth = 2] %sa02#0 : !dataflow.bits<32>
    %fifo_sa11_sa10 = fabric.fifo [depth = 2] %sa11#1 : !dataflow.bits<32>
    %fifo_sa12_sa11 = fabric.fifo [depth = 2] %sa12#1 : !dataflow.bits<32>
    %fifo_sa21_sa20 = fabric.fifo [depth = 2] %sa21#1 : !dataflow.bits<32>
    %fifo_sa22_sa21 = fabric.fifo [depth = 2] %sa22#1 : !dataflow.bits<32>

    // North FIFOs
    %fifo_sa10_sa00 = fabric.fifo [depth = 2] %sa10#0 : !dataflow.bits<32>
    %fifo_sa11_sa01 = fabric.fifo [depth = 2] %sa11#0 : !dataflow.bits<32>
    %fifo_sa12_sa02 = fabric.fifo [depth = 2] %sa12#0 : !dataflow.bits<32>
    %fifo_sa20_sa10 = fabric.fifo [depth = 2] %sa20#0 : !dataflow.bits<32>
    %fifo_sa21_sa11 = fabric.fifo [depth = 2] %sa21#0 : !dataflow.bits<32>
    %fifo_sa22_sa12 = fabric.fifo [depth = 2] %sa22#0 : !dataflow.bits<32>

    // ======== Mesh A PE Instances ========

    // CellA[0,0]: addi, TL=SA00 BL=SA10, output->SA11(BR)
    %pe_a00 = fabric.instance @pe_addi(%sa00#2, %sa10#3)
        {sym_name = "pe_a_0_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // CellA[0,1]: addi, TL=SA01 BL=SA11, output->SA12(BR)
    %pe_a01 = fabric.instance @pe_addi(%sa01#3, %sa11#4)
        {sym_name = "pe_a_0_1"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // CellA[1,0]: muli, TL=SA10 BL=SA20, output->SA21(BR)
    %pe_a10 = fabric.instance @pe_muli(%sa10#4, %sa20#2)
        {sym_name = "pe_a_1_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // CellA[1,1]: muli, TL=SA11 BL=SA21, output->SA22(BR)
    %pe_a11 = fabric.instance @pe_muli(%sa11#5, %sa21#3)
        {sym_name = "pe_a_1_1"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // ======== Cross Connections: A -> B (add_tag, direct) ========

    %cross_a02_b00 = fabric.add_tag %sa02#2 {tag = 0 : i4}
        : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i4>
    %cross_a12_b10 = fabric.add_tag %sa12#4 {tag = 0 : i4}
        : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i4>
    %cross_a22_b20 = fabric.add_tag %sa22#3 {tag = 0 : i4}
        : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Mesh B Switches (tagged, tagged<bits<32>, i4>) ========

    // SB00 [3in, 3out]
    //   in:  cross<-SA02(add_tag), fifo<-SB01(W), fifo<-SB10(N)
    //   out: #0 E->SB01, #1 S->SB10, #2 TPE CellB00 in0
    %sb00:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %cross_a02_b00, %fifo_sb01_sb00, %fifo_sb10_sb00
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // SB01 [3in, 4out]
    //   in:  SB00#0(E), fifo<-SB02(W), fifo<-SB11(N)
    //   out: #0 W->fifo->SB00, #1 E->SB02, #2 S->SB11, #3 TPE CellB01 in0
    %sb01:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sb00#0, %fifo_sb02_sb01, %fifo_sb11_sb01
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // SB02 [2in, 3out]
    //   in:  SB01#1(E), fifo<-SB12(N)
    //   out: #0 W->fifo->SB01, #1 S->SB12, #2 cross->SA02(del_tag+fifo)
    %sb02:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1]]
        %sb01#1, %fifo_sb12_sb02
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // SB10 [4in, 5out]
    //   in:  cross<-SA12(add_tag), SB00#1(S), fifo<-SB11(W), fifo<-SB20(N)
    //   out: #0 N->fifo->SB00, #1 E->SB11, #2 S->SB20,
    //        #3 TPE CellB00 in1, #4 TPE CellB10 in0
    %sb10:5 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %cross_a12_b10, %sb00#1, %fifo_sb11_sb10, %fifo_sb20_sb10
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // SB11 [5in, 6out]
    //   in:  SB01#2(S), SB10#1(E), fifo<-SB12(W), fifo<-SB21(N), TPE CellB00 out
    //   out: #0 N->fifo->SB01, #1 W->fifo->SB10, #2 E->SB12, #3 S->SB21,
    //        #4 TPE CellB01 in1, #5 TPE CellB11 in0
    %sb11:6 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sb01#2, %sb10#1, %fifo_sb12_sb11, %fifo_sb21_sb11, %tpe_b00
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // SB12 [4in, 4out]
    //   in:  SB02#1(S), SB11#2(E), fifo<-SB22(N), TPE CellB01 out
    //   out: #0 N->fifo->SB02, #1 W->fifo->SB11, #2 S->SB22,
    //        #3 cross->SA12(del_tag+fifo)
    %sb12:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sb02#1, %sb11#2, %fifo_sb22_sb12, %tpe_b01
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // SB20 [3in, 3out]
    //   in:  cross<-SA22(add_tag), SB10#2(S), fifo<-SB21(W)
    //   out: #0 N->fifo->SB10, #1 E->SB21, #2 TPE CellB10 in1
    %sb20:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %cross_a22_b20, %sb10#2, %fifo_sb21_sb20
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // SB21 [4in, 4out]
    //   in:  SB11#3(S), SB20#1(E), fifo<-SB22(W), TPE CellB10 out
    //   out: #0 N->fifo->SB11, #1 W->fifo->SB20, #2 E->SB22,
    //        #3 TPE CellB11 in1
    %sb21:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sb11#3, %sb20#1, %fifo_sb22_sb21, %tpe_b10
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // SB22 [3in, 3out]
    //   in:  SB12#2(S), SB21#2(E), TPE CellB11 out
    //   out: #0 N->fifo->SB12, #1 W->fifo->SB21,
    //        #2 cross->SA22(del_tag+fifo)
    %sb22:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sb12#2, %sb21#2, %tpe_b11
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Mesh B FIFOs (backward: West and North) ========

    // West FIFOs
    %fifo_sb01_sb00 = fabric.fifo [depth = 2] %sb01#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_sb02_sb01 = fabric.fifo [depth = 2] %sb02#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_sb11_sb10 = fabric.fifo [depth = 2] %sb11#1
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_sb12_sb11 = fabric.fifo [depth = 2] %sb12#1
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_sb21_sb20 = fabric.fifo [depth = 2] %sb21#1
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_sb22_sb21 = fabric.fifo [depth = 2] %sb22#1
        : !dataflow.tagged<!dataflow.bits<32>, i4>

    // North FIFOs
    %fifo_sb10_sb00 = fabric.fifo [depth = 2] %sb10#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_sb11_sb01 = fabric.fifo [depth = 2] %sb11#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_sb12_sb02 = fabric.fifo [depth = 2] %sb12#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_sb20_sb10 = fabric.fifo [depth = 2] %sb20#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_sb21_sb11 = fabric.fifo [depth = 2] %sb21#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_sb22_sb12 = fabric.fifo [depth = 2] %sb22#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Mesh B TPE Instances ========

    // CellB[0,0]: tpe_alu, TL=SB00 BL=SB10, output->SB11(BR)
    %tpe_b00 = fabric.instance @tpe_alu(%sb00#2, %sb10#3)
        {sym_name = "tpe_b_0_0"}
        : (!dataflow.tagged<!dataflow.bits<32>, i4>,
           !dataflow.tagged<!dataflow.bits<32>, i4>)
          -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // CellB[0,1]: tpe_alu, TL=SB01 BL=SB11, output->SB12(BR)
    %tpe_b01 = fabric.instance @tpe_alu(%sb01#3, %sb11#4)
        {sym_name = "tpe_b_0_1"}
        : (!dataflow.tagged<!dataflow.bits<32>, i4>,
           !dataflow.tagged<!dataflow.bits<32>, i4>)
          -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // CellB[1,0]: tpe_alu, TL=SB10 BL=SB20, output->SB21(BR)
    %tpe_b10 = fabric.instance @tpe_alu(%sb10#4, %sb20#2)
        {sym_name = "tpe_b_1_0"}
        : (!dataflow.tagged<!dataflow.bits<32>, i4>,
           !dataflow.tagged<!dataflow.bits<32>, i4>)
          -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // CellB[1,1]: tpe_alu, TL=SB11 BL=SB21, output->SB22(BR)
    %tpe_b11 = fabric.instance @tpe_alu(%sb11#5, %sb21#3)
        {sym_name = "tpe_b_1_1"}
        : (!dataflow.tagged<!dataflow.bits<32>, i4>,
           !dataflow.tagged<!dataflow.bits<32>, i4>)
          -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Cross Connections: B -> A (del_tag + fifo) ========

    %del_b02 = fabric.del_tag %sb02#2
        : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.bits<32>
    %cross_b02_a02 = fabric.fifo [depth = 2] %del_b02 : !dataflow.bits<32>

    %del_b12 = fabric.del_tag %sb12#3
        : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.bits<32>
    %cross_b12_a12 = fabric.fifo [depth = 2] %del_b12 : !dataflow.bits<32>

    %del_b22 = fabric.del_tag %sb22#2
        : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.bits<32>
    %cross_b22_a22 = fabric.fifo [depth = 2] %del_b22 : !dataflow.bits<32>

    // ======== Module Outputs ========
    // out0 from SA22 (bottom-right of Mesh A)
    // out1 from SA12 (right-middle of Mesh A)
    fabric.yield %sa22#2, %sa12#3 : !dataflow.bits<32>, !dataflow.bits<32>
  }
}
