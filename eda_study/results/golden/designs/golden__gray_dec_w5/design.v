module gray_dec_w5(input [4:0] g, output reg [4:0] b);
  integer j;
  always @(*) begin
    b[4] = g[4];
    for (j = 3; j >= 0; j = j - 1) b[j] = b[j + 1] ^ g[j];
  end
endmodule
