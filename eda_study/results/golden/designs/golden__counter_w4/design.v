module counter_w4(input clk, input rst, input en, output reg [3:0] count);
  always @(posedge clk)
    if (rst) count <= 0;
    else if (en) count <= count + 1'b1;
endmodule
