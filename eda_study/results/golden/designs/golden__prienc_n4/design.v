module prienc_n4(input [3:0] in, output reg [1:0] out, output reg valid);
  integer j;
  always @(*) begin
    valid = |in; out = 0;
    for (j = 0; j < 4; j = j + 1) if (in[j]) out = j;
  end
endmodule
