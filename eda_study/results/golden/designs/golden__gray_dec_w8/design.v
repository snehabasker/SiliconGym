module gray_dec_w8(input [7:0] g, output reg [7:0] b);
  integer j;
  always @(*) begin
    b[7] = g[7];
    for (j = 6; j >= 0; j = j - 1) b[j] = b[j + 1] ^ g[j];
  end
endmodule
