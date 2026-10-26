module barrel_shift_w4(input [3:0] in, input [1:0] shamt, input dir, output [3:0] out);
  assign out = dir ? (in >> shamt) : (in << shamt);
endmodule
