module updown_w8(input clk, input rst, input en, input up, output reg [7:0] count);
  always @(posedge clk)
    if (rst) count <= 0;
    else if (en) count <= (up ? count + 1'b1 : count - 1'b1);
endmodule
