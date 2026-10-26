module gray_dec_w4(input [3:0] g, output reg [3:0] b);
  integer j;
  always @(*) begin
    b[3] = g[3];
    for (j = 2; j >= 0; j = j - 1) b[j] = b[j + 1] ^ g[j];
  end
endmodule
