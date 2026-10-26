module barrel_rot_w4(input [3:0] in, input [1:0] shamt, input dir, output [3:0] out);
  assign out = dir ? ((in >> shamt) | (in << (4 - shamt))) : ((in << shamt) | (in >> (4 - shamt)));
endmodule
