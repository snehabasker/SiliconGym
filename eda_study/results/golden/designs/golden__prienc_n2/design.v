module prienc_n2(input [1:0] in, output reg out, output reg valid);
  integer j;
  always @(*) begin
    valid = |in; out = 0;
    for (j = 0; j < 2; j = j + 1) if (in[j]) out = j;
  end
endmodule
