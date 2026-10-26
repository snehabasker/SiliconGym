module prienc_n8(input [7:0] in, output reg [2:0] out, output reg valid);
  integer j;
  always @(*) begin
    valid = |in; out = 0;
    for (j = 0; j < 8; j = j + 1) if (in[j]) out = j;
  end
endmodule
