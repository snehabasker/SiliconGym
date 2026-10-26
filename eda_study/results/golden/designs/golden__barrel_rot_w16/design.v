module barrel_rot_w16(input [15:0] in, input [3:0] shamt, input dir, output [15:0] out);
  assign out = dir ? ((in >> shamt) | (in << (16 - shamt))) : ((in << shamt) | (in >> (16 - shamt)));
endmodule
