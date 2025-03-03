import os

from phidl import Device, Layer, make_device, Path, CrossSection
from phidl import quickplot as qp
import phidl.geometry as pg
import phidl.routing as pr
import phidl.path as pp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端


import scipy.io
import scipy.special as sc
from scipy.constants import pi
import numpy as np
from datetime import *


import iqpgds.iqplib as il

#layer definition
from iqpgds.iqplayer import layermap


def euler_rect_ring_pump_couple(width=1.6, l_st = 545, w_bus = 1.6, gap = 0.6, Rmin = 40, add_straight =0,layer =layermap['wg']):
    D = Device()
    wg1 = D<< il.wg(width1=width,width2=width,length=l_st,layer=layer)
    wg2 = D<< il.wg(width1=width,width2=width,length=l_st,layer=layer)
    wg_bus = D<<il.wg(width1=w_bus,width2=w_bus,length=2*Rmin,layer=layer)
    wg_bus.rotate(-90)
    wg_out1 = D<<il.wg(width1=width,width2=width,length=l_st+300,layer=layer)
    wg_out2 = D<<il.wg(width1=width,width2=width,length=l_st+300,layer=layer)


    euler1 = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                          anticlockwise=True, layer=layer)
    euler1_l = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                          anticlockwise=True, layer=layer)

    euler1_l = euler1_l.mirror((0,0),(0,1))
    euler2 = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                          anticlockwise=True, layer=layer)
    euler2.mirror(euler2.ports[2].center,(euler2.ports[2].center[0]-1,euler2.ports[2].center[1]+1))

    euler2_l = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                          anticlockwise=True, layer=layer)
    euler2_l.mirror(euler2_l.ports[2].center, (euler2_l.ports[2].center[0] - 1, euler2_l.ports[2].center[1] + 1))
    euler2_l = euler2_l.mirror((0,0),(0,1))

    euler4 = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                          anticlockwise=False, layer=layer)
    euler4.mirror(euler4.ports[2].center,(euler4.ports[2].center[0]+1,euler4.ports[2].center[1]+1))
    euler4.move(euler4.ports[1].center,euler2.ports[1].center)

    euler4_l = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                          anticlockwise=False, layer=layer)
    euler4_l.mirror(euler4_l.ports[2].center, (euler4_l.ports[2].center[0] + 1, euler4_l.ports[2].center[1] + 1))

    euler4_l.mirror((0,0),(0,1))

    euler3 = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                          anticlockwise=True, layer=layer)
    euler3.mirror((0,0),(1,0))
    euler3.move(euler3.ports[2].center,euler4.ports[2].center)

    euler3_l =  D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                          anticlockwise=True, layer=layer)
    euler3_l.mirror((0,0),(1,0))
    euler3_l.mirror((0,0),(0,1))








    if add_straight>0:
        wg_add1 = D<< il.wg(width1=width,width2=width,length=add_straight,layer=layer).rotate(-90)
        wg_add2 = D<< il.wg(width1=width,width2=width,length=add_straight,layer=layer).rotate(-90)

        wg1.connect(port=2, destination=euler1.ports[1])
        euler2.connect(port=2, destination=euler1.ports[2])
        wg_add1.connect(port=2,destination=euler2.ports[1])
        euler4.connect(port=1, destination=wg_add1.ports[1])
        euler3.connect(port=2, destination=euler4.ports[2])
        euler3.rotate(-90, center=euler4.ports[2].center)
        wg2.connect(port=2, destination=euler3.ports[1])
        euler1_l.connect(port=1, destination=wg1.ports[1])
        euler2_l.connect(port=2, destination=euler1_l.ports[2])
        wg_add2.connect(port=2,destination=euler2_l.ports[1])
        euler4_l.connect(port=1, destination=wg_add2.ports[1])
        euler3_l.connect(port=2, destination=euler4_l.ports[2])
        euler3_l.rotate(90, center=euler4_l.ports[2].center)
        wg_bus.move(wg_bus.center, (euler2.ports[1].center[0] + width / 2 + w_bus / 2 + gap, euler2.ports[1].center[1]))

        wg_out1.move(wg_out1.ports[2].center, (wg2.ports[2].center[0] - 50, wg2.ports[2].center[1] + 50))
        wg_out2.move(wg_out2.ports[2].center, (wg_out1.ports[2].center[0], wg_out1.ports[2].center[1] - 3 * 127))

        D << pr.route_smooth(wg_bus.ports[1], wg_out1.ports[2], radius=80, layer=layer)
        D << pr.route_smooth(wg_bus.ports[2], wg_out2.ports[2], radius=80, layer=layer)


    else:
        wg1.connect(port=2,destination=euler1.ports[1])
        euler2.connect(port=2,destination=euler1.ports[2])
        euler4.connect(port=1,destination=euler2.ports[1])
        euler3.connect(port=2,destination=euler4.ports[2])
        euler3.rotate(-90,center = euler4.ports[2].center)
        wg2.connect(port=2,destination=euler3.ports[1])
        euler1_l.connect(port=1,destination=wg1.ports[1])
        euler2_l.connect(port=2, destination = euler1_l.ports[2])
        euler4_l.connect(port=1,destination=euler2_l.ports[1])
        euler3_l.connect(port=2,destination=euler4_l.ports[2])
        euler3_l.rotate(90, center=euler4_l.ports[2].center)
        wg_bus.move(wg_bus.center,(euler2.ports[1].center[0]+width/2+w_bus/2+gap,euler2.ports[1].center[1]))

        wg_out1.move(wg_out1.ports[2].center,(wg2.ports[2].center[0]-50,wg2.ports[2].center[1]+50))
        wg_out2.move(wg_out2.ports[2].center, (wg_out1.ports[2].center[0], wg_out1.ports[2].center[1] -3*127))

        D<<pr.route_smooth(wg_bus.ports[1],wg_out1.ports[2],radius=80,layer = layer)
        D<<pr.route_smooth(wg_bus.ports[2], wg_out2.ports[2], radius=80,layer = layer)


    D.add_port(1,wg_out1.ports[1].center,width,180)
    D.add_port(2,wg_out2.ports[1].center,width,180)

    return D


def euler_rect_ring_DC (width=1.6, l_st = 545, w_bus = 1.6, gap = 0.6, Rmin = 40, add_straight =0,layer =layermap['wg']):
    D = Device()
    wg1 = D << il.wg(width1=width, width2=width, length=l_st, layer=layer)
    wg2 = D << il.wg(width1=width, width2=width, length=l_st, layer=layer)
    wg_bus = D << il.wg(width1=w_bus, width2=w_bus, length=l_st, layer=layer)

    wg_out1 = D << il.wg(width1=width, width2=width, length=l_st + 300, layer=layer)

    euler1 = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                anticlockwise=True, layer=layer)
    euler1_l = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                  anticlockwise=True, layer=layer)

    euler1_l = euler1_l.mirror((0, 0), (0, 1))
    euler2 = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                anticlockwise=True, layer=layer)
    euler2.mirror(euler2.ports[2].center, (euler2.ports[2].center[0] - 1, euler2.ports[2].center[1] + 1))

    euler2_l = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                  anticlockwise=True, layer=layer)
    euler2_l.mirror(euler2_l.ports[2].center, (euler2_l.ports[2].center[0] - 1, euler2_l.ports[2].center[1] + 1))
    euler2_l = euler2_l.mirror((0, 0), (0, 1))

    euler4 = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                anticlockwise=False, layer=layer)
    euler4.mirror(euler4.ports[2].center, (euler4.ports[2].center[0] + 1, euler4.ports[2].center[1] + 1))
    euler4.move(euler4.ports[1].center, euler2.ports[1].center)

    euler4_l = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                  anticlockwise=False, layer=layer)
    euler4_l.mirror(euler4_l.ports[2].center, (euler4_l.ports[2].center[0] + 1, euler4_l.ports[2].center[1] + 1))

    euler4_l.mirror((0, 0), (0, 1))

    euler3 = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                anticlockwise=True, layer=layer)
    euler3.mirror((0, 0), (1, 0))
    euler3.move(euler3.ports[2].center, euler4.ports[2].center)

    euler3_l = D << il.euler_bend(width1=width, width2=width, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                  anticlockwise=True, layer=layer)
    euler3_l.mirror((0, 0), (1, 0))
    euler3_l.mirror((0, 0), (0, 1))

    if add_straight > 0:
        wg_add1 = D << il.wg(width1=width, width2=width, length=add_straight, layer=layer).rotate(-90)
        wg_add2 = D << il.wg(width1=width, width2=width, length=add_straight, layer=layer).rotate(-90)

        wg1.connect(port=2, destination=euler1.ports[1])
        euler2.connect(port=2, destination=euler1.ports[2])
        wg_add1.connect(port=2, destination=euler2.ports[1])
        euler4.connect(port=1, destination=wg_add1.ports[1])
        euler3.connect(port=2, destination=euler4.ports[2])
        euler3.rotate(-90, center=euler4.ports[2].center)
        wg2.connect(port=2, destination=euler3.ports[1])
        euler1_l.connect(port=1, destination=wg1.ports[1])
        euler2_l.connect(port=2, destination=euler1_l.ports[2])
        wg_add2.connect(port=2, destination=euler2_l.ports[1])
        euler4_l.connect(port=1, destination=wg_add2.ports[1])
        euler3_l.connect(port=2, destination=euler4_l.ports[2])
        euler3_l.rotate(90, center=euler4_l.ports[2].center)
        wg_bus.move(wg_bus.center, (euler2.ports[1].center[0] + width / 2 + w_bus / 2 + gap, euler2.ports[1].center[1]))

        wg_out1.move(wg_out1.ports[2].center, (wg2.ports[2].center[0] - 50, wg2.ports[2].center[1] + 50))
        wg_out2.move(wg_out2.ports[2].center, (wg_out1.ports[2].center[0], wg_out1.ports[2].center[1] - 3 * 127))

        D << pr.route_smooth(wg_bus.ports[1], wg_out1.ports[2], radius=80, layer=layer)
        D << pr.route_smooth(wg_bus.ports[2], wg_out2.ports[2], radius=80, layer=layer)


    else:
        wg1.connect(port=2, destination=euler1.ports[1])
        euler2.connect(port=2, destination=euler1.ports[2])
        euler4.connect(port=1, destination=euler2.ports[1])
        euler3.connect(port=2, destination=euler4.ports[2])
        euler3.rotate(-90, center=euler4.ports[2].center)
        wg2.connect(port=2, destination=euler3.ports[1])
        euler1_l.connect(port=1, destination=wg1.ports[1])
        euler2_l.connect(port=2, destination=euler1_l.ports[2])
        euler4_l.connect(port=1, destination=euler2_l.ports[1])
        euler3_l.connect(port=2, destination=euler4_l.ports[2])
        euler3_l.rotate(90, center=euler4_l.ports[2].center)
        wg_bus.move(wg_bus.center, (wg1.center[0],wg1.center[1]-width/2-w_bus/2-gap))
        euler_outr1 = D << il.euler_bend(width1=w_bus, width2=w_bus, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                anticlockwise=False, layer=layer)
        euler_outr1.connect(port=1,destination=wg_bus.ports[2])
        euler_outr2 = D << il.euler_bend(width1=w_bus, width2=w_bus, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                         anticlockwise=False, layer=layer)
        euler_outr2.rotate(180)
        euler_outr2.connect(port=2,destination=euler_outr1.ports[2])
        euler_outl1 = D << il.euler_bend(width1=w_bus, width2=w_bus, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                         anticlockwise=False, layer=layer)
        euler_outl1.mirror((0,0),(0,1))
        euler_outl2 = D << il.euler_bend(width1=w_bus, width2=w_bus, R0=Rmin * np.sqrt(pi), angle1=0, angle2=45,
                                         anticlockwise=False, layer=layer)
        euler_outl2.rotate(180)
        euler_outl2.mirror((0,0),(0,1))

        euler_outl1.connect(port=1,destination=wg_bus.ports[1])
        euler_outl2.connect(port=2,destination=euler_outl1.ports[2])

        wg_out1.move(wg_out1.ports[2].center, (wg2.ports[2].center[0] - 50, euler_outl2.ports[1].center[1] + 3*127))
        #
        #
        D << pr.route_smooth(euler_outr2.ports[1], wg_out1.ports[2], radius=80, layer=layer)
        lbu = wg_out1.ports[1].center[0] - euler_outl2.ports[1].center[0]

        wg_out2bu = D<<il.wg(width1=w_bus,width2=w_bus,length=lbu,layer=layer)
        wg_out2bu.connect(port=2,destination=euler_outl2.ports[1])

    D.add_port(1, wg_out1.ports[1].center, width, 180)
    D.add_port(2, wg_out2bu.ports[1].center, width, 180)

    return D


D= Device()
#D<<euler_rect_ring_pump_couple()
D<<euler_rect_ring_DC()
qp(D)

# %%
# %%
D.flatten()
fname = str(datetime.now()).split(' ')[0].replace(
    '-', '')+'_'+'euler_coupler_test'
print(fname)
if 1:
    D.write_gds(filename=fname+'.gds', unit=1e-6, precision=1e-9, cellname = 'CHIP1')
