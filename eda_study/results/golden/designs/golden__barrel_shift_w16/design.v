module barrel_shift_w16(input [15:0] in, input [3:0] shamt, input dir, output [15:0] out);
  assign out = dir ? (in >> shamt) : (in << shamt);
endmodule
