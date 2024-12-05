import os

from phidl import Device, Layer, make_device, Path, CrossSection
from phidl import quickplot as qp
import phidl.geometry as pg
import phidl.routing as pr
import phidl.path as pp
from phidl.utilities import write_svg
import phidl
import gdspy
import scipy.io
import scipy.special as sc
from scipy.constants import pi
from matplotlib import pyplot as plt
import numpy as np
from datetime import *
import operator
import libsw as sw
import matplotlib.pyplot as plt
# import di
# import Lib_ZhudiGroup as zdg

# %%
# layer definition
layer_outline = 0
layer_ebeam_mark = 10
layer_ebeam_mark_local = 11
layer_photo_mark = 12
layer_metal = 1
layer_wg = 2
layer_grating = 4
layer_litho_test = 6
layer_etch_test = 7
layer_ring = 3
layer_text = 13
layer_ebeam_metal = 5
layer_EL6_fine = 21
layer_EL6_coarse = 22
layer_oxide_open = 5


def ensure_center_in_list(pitch, tolerance, num_points):
    # 生成包含 (num_points - 1) 的数列
    pitch_list = np.linspace(pitch * (1 - tolerance), pitch * (1 + tolerance), num_points - 1).tolist()

    # 插入 pitch
    pitch_list.append(pitch)

    # 确保数列有序，并保持 num_points 个点
    pitch_list = sorted(pitch_list)

    return pitch_list

# %% what is this marker

# %%
DIE = Device()
DIE = pg.basic_die(size=(40000, 40000), street_width=100, street_length=300,
                   die_name='ZY_Couple_TestChip1', text_size=100, text_location='NW',  layer=99,
                   draw_bbox=False,  bbox_layer=99)


# global ebeam mark
mark_offset = 5000
DIE.add_ref(sw.alignment_mark_array(width=2, length=300, center_square=False,
                                    offset=[(-200, 200), (-200, -200),
                                            (200, -200), (200, 200)],
                                    coordinates=[(-mark_offset, 26000), (-mark_offset, -20000),
                                                 (mark_offset+4000, -20000), (mark_offset+4000, 26000)],
                                    layer=layer_ebeam_mark, double_marks=False))
qp(DIE)




# parameters
name = 'Test_Connection'
D = Device(name)

R_ring = 206
Theta_c = 10
Gap_start = 0.4
Gap_end = 1.2
W_ring = 2.5
W_bus_1550 = 0.9
W_bus_780 = 0.8
Poling_p = 3.849
mode_num = round(2*pi*R_ring/Poling_p)
w_1550_out = 3
w_780_out = 2
gap =0.4
R_out =300
R_couple =200

dx = 1100-30
dy = 800
detaX = 500     # 不同环之间的x错位 in same group
detaY = 2*127      # 不同环之间的y错位 in the same group
detaY2 = -4500      # 不同环之间的y错位 in different group
D_fiber_array=127  #光纤之间的间隔127um
L_ytop = 3000
L_ybottom = 3000

gap_ir = gap
gap_vis =gap
pulley_angle = 29

A = Device("single")
# Ring_point_couple = A << sw.ring_resonator(width=W_ring,R_ring=R_ring,gap_s=gap_ir,gap_e=6,layer=layer_ring)
#                                                 #=gap_ir,gap_e=6, layer=layer_ring)
#Ring_point_couple = A << sw.ring_coupling_symmetric(w_ring=W_ring,w_bus=W_bus_1550,R= R_ring,g=gap_ir,layer=layer_ring)
Ring_point_couple = A << sw.ring_coupling_pulley_taper(w_ring=W_ring,w_bus=W_bus_1550,w_thin=0.3,R=R_ring,g=gap_vis,layer=layer_ring)
qp(A)

A.write_gds(filename=name+'.gds',precision=1e-10)