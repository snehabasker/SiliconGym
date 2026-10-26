module popcount_w8(input [7:0] in, output reg [3:0] count);
  integer j;
  always @(*) begin
    count = 0;
    for (j = 0; j < 8; j = j + 1) count = count + in[j];
  end
endmodule
