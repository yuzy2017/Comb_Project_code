### code has the shape of Junyong  ###

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
name = 'ZY_Couple_TestChip1'
D = Device(name)

R_ring = 206
Theta_c = 10
Gap_start = 0.4
Gap_end = 1.2
W_ring = 2.5
W_bus_1550 = 0.9
W_bus_780 = 0.7
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

gap_1550 = gap
gap_780 =gap

def ring_coupler_1550(gap_ir=0.4,colume_idx = 0, total_colum=9):
    A = Device("single")
    Ring_point_couple = A << sw.ring_coupling_point(width=W_ring, w_bus=W_bus_1550, R_ring=R_ring, R_couple=R_couple,
                                                    gap_s=gap_ir, gap_e=6, layer=layer_ring).rotate(90)

    Taper_in = A << sw.sbend_taper(w1=w_1550_out, w2=W_bus_1550, l=500, layer=layer_ring)
    Taper_in.move((-600, -600))
    Route_in = A << pr.route_smooth(Ring_point_couple.ports[1], Taper_in.ports[2], width=W_bus_1550, radius=R_out,
                                    layer=layer_ring)
    Taper_out = A << sw.sbend_taper(w1=W_bus_1550, w2=w_1550_out, l=500, layer=layer_ring)
    # Taper_out.connect(1,destination=Ring_point_couple.ports[2])
    Taper_out.move((600, 600))

    Route_out = A << pr.route_smooth(Ring_point_couple.ports[2], Taper_out.ports[1], width=W_bus_1550, radius=R_out,
                                     layer=layer_ring)
    ydiff = abs(Ring_point_couple.ports[2].midpoint[1] - Route_out.ymin)
    xdiff = abs(Ring_point_couple.ports[2].midpoint[0] - W_bus_1550 / 2 - Route_out.xmin)
    Route_out.move((-xdiff, -ydiff))
    # Ring_point_couple = A<< sw.ring_resonator(width=1, R_ring=200, gap_s=0.3, gap_e=6, layer=6)

    WG_1550_in = A << sw.wg(width1=w_1550_out, width2=w_1550_out, length=2500 + colume_idx * detaX, layer=layer_ring)
    WG_1550_in.connect(2, destination=Taper_in.ports[1])
    WG_1550_out = A << sw.wg(width1=w_1550_out, width2=w_1550_out, length=2500 + (total_colum - colume_idx) * detaX,
                             layer=layer_ring)
    WG_1550_out.connect(1, destination=Taper_out.ports[2])
    # %%
    # label #as we are going to sweep the width we try also sweep the gap here
    txt = ('R_ring {:.2f}\n'.format(R_ring) +
           'R_couple_ir {:.3f}\n'.format(R_couple) +
           'W_ring {:.3f}\n'.format(W_ring) +
           'W_bus_ir {:.3f}\n'.format(W_bus_1550) +
           'g_ir {:.2f}\n'.format(gap_ir) )

    LABEL1 = pg.text(text=txt, size=20, justify='center',
                     layer=layer_text, font='DEPLOF')
    label1 = A << LABEL1

    return A

def ring_coupler_780(gap_vis=0.4,pulley_angle=29,colume_idx = 0, total_colum=9):
    A = Device("single")

    Pulley_coupler = A << sw.ring_coupling_pulley_zy(w_ring=W_ring, w_bus=W_bus_780, R=R_ring, g=gap_vis, angle=pulley_angle/2,
                                                     layer=layer_ring)
    Pulley_coupler.rotate(90)

    Taper_in = A << sw.sbend_taper(w1=w_780_out, w2=W_bus_780, l=500, layer=layer_ring)
    Taper_in.move((-630, -600))
    Route_in = A << pr.route_smooth(Pulley_coupler.ports[2], Taper_in.ports[2], width=W_bus_780, radius=R_out,
                                    layer=layer_ring)
    Taper_out = A << sw.sbend_taper(w1=W_bus_780, w2=w_780_out, l=500, layer=layer_ring)
    # Taper_out.connect(1,destination=Ring_point_couple.ports[2])
    Taper_out.move((600, 600))
    Route_out = A << pr.route_smooth(Pulley_coupler.ports[1], Taper_out.ports[1], width=W_bus_780, radius=R_out,
                                     layer=layer_ring)
    ydiff = abs(Pulley_coupler.ports[1].midpoint[1] - Route_out.ymin)
    xdiff = abs(Pulley_coupler.ports[1].midpoint[0] - W_bus_780 / 2 - Route_out.xmin)
    Route_out.move((-xdiff, -ydiff))

    A.write_gds(filename=name + '.gds', precision=1e-10)
    # Ring_point_couple = A<< sw.ring_resonator(width=1, R_ring=200, gap_s=0.3, gap_e=6, layer=6)

    WG_780_in = A << sw.wg(width1=w_780_out, width2=w_780_out, length=2500 + colume_idx * detaX, layer=layer_ring)
    WG_780_in.connect(2, destination=Taper_in.ports[1])
    WG_780_out = A << sw.wg(width1=w_780_out, width2=w_780_out, length=2500 + (total_colum - colume_idx) * detaX,
                            layer=layer_ring)
    WG_780_out.connect(1, destination=Taper_out.ports[2])
    # %%
    # label #as we are going to sweep the width we try also sweep the gap here
    txt = ('R_ring {:.2f}\n'.format(R_ring) +
           'W_ring {:.3f}\n'.format(W_ring) +
           'W_bus_vis {:.3f}\n'.format(W_bus_780) +
           'g_vis {:.2f}\n'.format(gap_vis))

    LABEL1 = pg.text(text=txt, size=20, justify='center',
                     layer=layer_text, font='DEPLOF')
    label1 = A << LABEL1

    return  A


def single_device(gap_ir=0.4,gap_vis=0.4,pulley_angle=29,colume_idx = 0, total_colum=9,extra_move = 50):
    A = Device("single")
    Ring_point_couple = A << sw.ring_coupling_point(width=W_ring, w_bus=W_bus_1550, R_ring=R_ring, R_couple=R_couple,
                                                    gap_s=gap_ir, gap_e=6, layer=layer_ring)

    # here try to generate the poling fingers
    # Poling_finger = A<<sw.circular_poling_finger(R_ring=R_ring, length=20,circle_pad_width=10, square_pad_width=100,square_pad_rotation = -90, mode_number=mode_num, duty_cycle=0.3, layer=layer_metal)

    Pulley_coupler = A << sw.ring_coupling_pulley_zy(w_ring=W_ring, w_bus=W_bus_780, R=R_ring, g=gap_vis,
                                                     angle=pulley_angle / 2,
                                                     layer=layer_ring)
    Pulley_coupler.rotate(180)
    # no need this as we are operating at original points if we want to create another one we just shift the whole device
    # Pulley_coupler.move(destination=Ring_point_couple)

    A.rotate(-90)

    # lastly we add tapers
    Taper_in_1550 = A << sw.sbend_taper(w1=w_1550_out, w2=W_bus_1550, l=500, layer=layer_ring)
    Taper_in_1550.move((-1052, -600))
    # qp(A)
    Taper_out_1550 = A << sw.sbend_taper(w1=W_bus_1550, w2=w_1550_out, l=500, layer=layer_ring)
    Taper_out_1550.move((700, 650))
    # qp(A)
    Taper_out_780 = A << sw.sbend_taper(w1=W_bus_780, w2=w_780_out, l=500, layer=layer_ring)
    Taper_out_780.move((700, 650 - D_fiber_array))
    # qp(A)
    Taper_in_780 = A << sw.sbend_taper(w1=w_780_out, w2=W_bus_780, l=500, layer=layer_ring)
    Taper_in_780.move((-1052, -600 - D_fiber_array))

    # now connect the taper to the coupler with route
    route_in_1550 = A << pr.route_smooth(Ring_point_couple.ports[2], Taper_in_1550.ports[2], width=W_bus_1550,
                                         radius=R_out,
                                         layer=layer_ring)
    route_out_1550 = A << pr.route_smooth(Ring_point_couple.ports[1], Taper_out_1550.ports[1], width=W_bus_1550,
                                          radius=R_out,
                                          layer=layer_ring)
    route_in_780 = A << pr.route_smooth(Pulley_coupler.ports[2], Taper_in_780.ports[2], width=W_bus_780, radius=R_out,
                                        layer=layer_ring)
    route_out_780 = A << pr.route_smooth(Pulley_coupler.ports[1], Taper_out_780.ports[1], width=W_bus_780, radius=R_out,
                                         layer=layer_ring)

    # qp(A)
    # here we also need to give the input output waveguide
    WG_1550_in = A << sw.wg(width1=w_1550_out, width2=w_1550_out, length=2005 + colume_idx * (detaX+extra_move), layer=layer_ring)
    WG_1550_in.connect(2, destination=Taper_in_1550.ports[1])
    WG_780_in = A << sw.wg(width1=w_780_out, width2=w_780_out, length=2005 + colume_idx * (detaX+extra_move), layer=layer_ring)
    WG_780_in.connect(2, destination=Taper_in_780.ports[1])
    # qp(A)
    WG_1550_out = A << sw.wg(width1=w_1550_out, width2=w_1550_out, length=2500 + (total_colum - colume_idx) * (detaX+extra_move)-total_colum*extra_move,
                             layer=layer_ring)
    WG_1550_out.connect(1, destination=Taper_out_1550.ports[2])
    WG_780_out = A << sw.wg(width1=w_780_out, width2=w_780_out, length=2500 + (total_colum - colume_idx) * (detaX+extra_move)-total_colum*extra_move,
                            layer=layer_ring)
    WG_780_out.connect(1, destination=Taper_out_780.ports[2])
    if (colume_idx == 0) | (colume_idx == 8):
        A << sw.alignment_mark(layer=layer_ebeam_mark)

    #qp(A)
    #A.write_gds(filename='Test_chip.gds',unit=1e-6, precision=1e-9)


    # %%
    # label #as we are going to sweep the width we try also sweep the gap here
    txt = ('R_ring {:.2f}\n'.format(R_ring) +
       'R_couple_ir {:.3f}\n'.format(R_couple) +
       'W_ring {:.3f}\n'.format(W_ring) +
       'W_bus_ir {:.3f}\n'.format(W_bus_1550) +
       'W_bus_vis {:.3f}\n'.format(W_bus_780) +
       'g_ir {:.2f}\n'.format(gap_ir) +
       'g_vis {:.2f}\n'.format(gap_vis))

    LABEL1 = pg.text(text=txt, size=20, justify='center',
                 layer=layer_text, font='DEPLOF')
    label1 = A << LABEL1
    label1.movex(-30)
    label1.movey(150 + 300)

    return A
gap_780 = np.linspace(0.1,1.2,12)
gap_1550 = np.linspace(0.1,1.2,12)
D1 =Device('1550_coupler')
for i, gap_ir in enumerate(gap_1550):
    mydev = D1 << ring_coupler_1550(gap_ir,i,len(gap_1550))
    mydev.movex(i * detaX)
    mydev.movey(-i * detaY/2 )
DIE<<D1

D2 = Device('780_coupler')
for i, gap_vis in enumerate(gap_780):
    mydev = D2 << ring_coupler_780(gap_vis=gap_vis,pulley_angle=29,colume_idx=i,total_colum=len(gap_780))
    mydev.movex(i * detaX)
    mydev.movey(-i * detaY/2 )
D2.movey(-3000)
DIE<<D2

D3 = Device('Add_drop_coupler')
for i, gap_vis in enumerate(gap_780):
    mydev = D3 << single_device(gap_1550[i],gap_vis,29,i,len(gap_780))
    mydev.movex(i * (detaX+50))
    mydev.movey(-i * detaY )
D3.movey(-6000)
DIE << D3
marker = D<<sw.alignment_mark(layer=layer_ebeam_mark)
qp(DIE)
DIE.write_gds(filename=name+'.gds',unit=1e-6, precision=1e-10)



# pg.grid inconvenient, it is suitable for rectangular shaped device.