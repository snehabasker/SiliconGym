module gray_dec_w3(input [2:0] g, output reg [2:0] b);
  integer j;
  always @(*) begin
    b[2] = g[2];
    for (j = 1; j >= 0; j = j - 1) b[j] = b[j + 1] ^ g[j];
  end
endmodule
