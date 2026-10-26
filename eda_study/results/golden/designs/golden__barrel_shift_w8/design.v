module barrel_shift_w8(input [7:0] in, input [2:0] shamt, input dir, output [7:0] out);
  assign out = dir ? (in >> shamt) : (in << shamt);
endmodule
