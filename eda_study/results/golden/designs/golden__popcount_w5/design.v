module popcount_w5(input [4:0] in, output reg [2:0] count);
  integer j;
  always @(*) begin
    count = 0;
    for (j = 0; j < 5; j = j + 1) count = count + in[j];
  end
endmodule
