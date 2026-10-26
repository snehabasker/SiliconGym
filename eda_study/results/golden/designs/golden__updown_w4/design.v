module updown_w4(input clk, input rst, input en, input up, output reg [3:0] count);
  always @(posedge clk)
    if (rst) count <= 0;
    else if (en) count <= (up ? count + 1'b1 : count - 1'b1);
endmodule
