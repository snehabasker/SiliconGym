module popcount_w2(input [1:0] in, output reg [1:0] count);
  integer j;
  always @(*) begin
    count = 0;
    for (j = 0; j < 2; j = j + 1) count = count + in[j];
  end
endmodule
