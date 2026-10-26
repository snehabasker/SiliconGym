module dff_async_hi_en(input clk, input rst, input en, input [3:0] d, output reg [3:0] q);
  always @(posedge clk or posedge rst)
    if (rst == 1'b1) q <= 4'd0;
    else if (en) q <= d;
endmodule
