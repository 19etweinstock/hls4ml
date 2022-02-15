############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project vivado_hls_prj
add_files sherylll/keras-to-hls/lenet5-proj/firmware/lenet5.cpp
add_files -tb sherylll/keras-to-hls/lenet5-proj/lenet5_test.cpp
open_solution "solution1"
set_part {xc7vx485tffg1761-2}
create_clock -period 10 -name default
#source "./vivado_hls_prj/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
