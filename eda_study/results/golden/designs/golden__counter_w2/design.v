module counter_w2(input clk, input rst, input en, output reg [1:0] count);
  always @(posedge clk)
    if (rst) count <= 0;
    else if (en) count <= count + 1'b1;
endmodule
