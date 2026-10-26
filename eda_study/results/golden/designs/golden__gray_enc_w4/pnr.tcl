# Placement-level physical flow in OpenROAD for one synthesised sky130_fd_sc_hd netlist.
# Not OpenLane: no PDN, no tap/endcap cells, no detailed routing, no signoff. See eda_study/README.md.
#
# Inputs (environment): TOP NETLIST OUTDIR PERIOD_NS IS_SEQ LIB TLEF LEF CLK_PORT
# Output: OUTDIR/pnr_report.txt (every number below is parsed from this file)

set top      $::env(TOP)
set netlist  $::env(NETLIST)
set outdir   $::env(OUTDIR)
set period   $::env(PERIOD_NS)
set is_seq   $::env(IS_SEQ)

read_liberty $::env(LIB)
read_lef $::env(TLEF)
read_lef $::env(LEF)
read_verilog $netlist
link_design $top

# timing constraints: one 'clk' (virtual for combinational designs), zero I/O delays
if {$is_seq} {
  create_clock -name clk -period $period [get_ports $::env(CLK_PORT)]
  set_input_delay 0 -clock clk [lsearch -all -inline -not -exact [all_inputs] [get_ports $::env(CLK_PORT)]]
} else {
  create_clock -name vclk -period $period
  set_input_delay 0 -clock vclk [all_inputs]
}
set clk_name [expr {$is_seq ? "clk" : "vclk"}]
set_output_delay 0 -clock $clk_name [all_outputs]

# floorplan: square core of side CORE_SIDE um (set by eda_run.py: sqrt(synth area / 0.30), at least 20 um so that the
# smallest designs still have room for pins and clock buffers), sky130hd site
set side $::env(CORE_SIDE)
set die [expr {$side + 4.0}]
initialize_floorplan -die_area "0 0 $die $die" -core_area "2 2 [expr {$side + 2.0}] [expr {$side + 2.0}]" -site unithd
make_tracks li1  -x_offset 0.23 -x_pitch 0.46 -y_offset 0.17 -y_pitch 0.34
make_tracks met1 -x_offset 0.17 -x_pitch 0.34 -y_offset 0.17 -y_pitch 0.34
make_tracks met2 -x_offset 0.23 -x_pitch 0.46 -y_offset 0.23 -y_pitch 0.46
make_tracks met3 -x_offset 0.34 -x_pitch 0.68 -y_offset 0.34 -y_pitch 0.68
make_tracks met4 -x_offset 0.46 -x_pitch 0.92 -y_offset 0.46 -y_pitch 0.92
make_tracks met5 -x_offset 1.70 -x_pitch 3.40 -y_offset 1.70 -y_pitch 3.40

set_wire_rc -signal -layer met2
set_wire_rc -clock  -layer met5

place_pins -hor_layers met3 -ver_layers met2 -random
global_placement -density 0.5
estimate_parasitics -placement
repair_design
detailed_placement

if {$is_seq} {
  clock_tree_synthesis -root_buf sky130_fd_sc_hd__clkbuf_16 \
      -buf_list {sky130_fd_sc_hd__clkbuf_8 sky130_fd_sc_hd__clkbuf_4 sky130_fd_sc_hd__clkbuf_2}
  set_propagated_clock [all_clocks]
  detailed_placement
}
estimate_parasitics -placement

puts "=== REPORT_START"
puts "--- design_area"
report_design_area
puts "--- cell_type_report"
puts "--- worst_slack_max"
report_worst_slack -max
puts "--- worst_slack_min"
report_worst_slack -min
puts "--- tns"
report_tns
puts "--- critical_path"
report_checks -path_delay max -fields {slew cap} -digits 3
puts "--- power"
report_power
puts "=== REPORT_END"
write_def $outdir/placed.def
exit
