module prienc_n16(input [15:0] in, output reg [3:0] out, output reg valid);
  integer j;
  always @(*) begin
    valid = |in; out = 0;
    for (j = 0; j < 16; j = j + 1) if (in[j]) out = j;
  end
endmodule
