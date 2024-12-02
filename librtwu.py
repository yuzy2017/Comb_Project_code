import os


from phidl import Device, Layer, make_device, Path, CrossSection
from phidl import quickplot as qp
from phidl import set_quickplot_options
import phidl.geometry as pg
import phidl.routing as pr
import phidl.path as pp
from phidl.utilities import write_svg
import gdspy


import scipy.io
import scipy.special as sc
from scipy.constants import pi
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from datetime import *
import operator

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
# %%
# basic function, do not change

"""
1.创建欧拉曲线函数
"""
def _euler_curve(R0=1, angle1=10, angle2=90,num_seg=1000):
    angle1 = angle1/180*pi
    angle2 = angle2/180*pi
    s = np.linspace(R0*np.sqrt(2*angle1), R0*np.sqrt(2*angle2), num_seg)
    S, C = sc.fresnel(s/np.sqrt(pi)/R0)    # Fresnel integrals
    y = S*np.sqrt(pi)*R0
    x = C*np.sqrt(pi)*R0
    return x, y


x, y=_euler_curve(R0=0.5, angle1=0, angle2=90)     # 画图示例这个函数输出的结果就是一条euler curve 的坐标。angle1控制起始角度，angle2控制终止角度，R0控制曲线缩放

# plt.plot(x, y)
# plt.show()



"""
2.创建欧拉曲线拉伸出来指定宽度的波导Path对象
"""
def euler_bend(width1=1, width2=2, R0=50, angle1=0, angle2=90, num_seg=1000, anticlockwise=False, layer=1):
    '''euler bend
    '''
    if anticlockwise == True:
        x, y = _euler_curve(R0, angle1, angle2,num_seg)
        dx = 1e-9
        x = np.append(x, x[-1]+dx*np.cos(angle2/180*pi))
        y = np.append(y, y[-1]+dx*np.sin(angle2/180*pi))
        P = Path(np.array((x, y)).T)    #将欧拉曲线的 x 和 y 坐标值数组转换为 Path 对象，并将其赋值给变量 P。现在，P 是一个包含了欧拉曲线路径信息的 Path 对象。
        P1 = P.extrude(width=[width1, width2])     #这段代码对路径 P 进行了拉伸操作，创建了一个新的 Device 对象 P1。在这里，width=[width1, width2] 参数指定了拉伸的宽度
        P1.flatten(single_layer=layer)
        euler_len = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))  # Compute length
        euler_detay = abs(y[-1]-y[0])
        euler_detax = abs(x[-1] - x[0])
        port1 = P1.add_port(1, (x[0], y[0]), width1, angle1+180)    #这行代码在欧拉弯曲器件 P1 上添加了一个端口，端口编号为 1。该端口位于欧拉曲线的起始点 (x[0], y[0]) 处，具有宽度 width1，角度为 angle1+180。通常，角度是以相对于水平轴的逆时针方向为正方向的。
        port2 = P1.add_port(2, (x[-1], y[-1]), width2, angle2)
        port3 = P1.add_port(3, (x[0], y[0]), width1, angle1+270)
        port4 = P1.add_port(4, (x[-1], y[-1]), width2, angle2+270)
        return P1, euler_len, euler_detax, euler_detay
    if anticlockwise == False:
        x, y = _euler_curve(R0, angle1, angle2)
        dx = 1e-9
        x = np.append(x, x[-1]+dx*np.cos(angle2/180*pi))
        y = -np.append(y, y[-1]+dx*np.sin(angle2/180*pi))
        P = Path(np.array((x, y)).T)
        P1 = P.extrude(width=[width1, width2])
        P1.flatten(single_layer=layer)
        euler_len = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))  # Compute length
        euler_detay = abs(y[-1]-y[0])
        euler_detax = abs(x[-1] - x[0])
        port1 = P1.add_port(1, (x[0], y[0]), width1, angle1+180)
        port2 = P1.add_port(2, (x[-1], y[-1]), width2, angle2+180)
        port3 = P1.add_port(3, (x[0], y[0]), width1, angle1+90)
        port4 = P1.add_port(4, (x[-1], y[-1]), width2, angle2+90)
        return P1, euler_len, euler_detax, euler_detay


# P1, euler_len, euler_detax, euler_detay=euler_bend(width1=1, width2=2, R0=50, angle1=0, angle2=90, anticlockwise=False, layer=1)
# # print(euler_len)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=True,zoom_factor=True,interactive_zoom=True)
# qp(P1)

"""
3.创建基于欧拉曲线的euler_bend_gap波导Path对象
"""

def euler_bend_gap(width_ring=1, width_bus=5, R0=50, angle1=0, angle2=90, anticlockwise=True, num_seg=1000, gap1=2.8, gap2=0.8, p=1, layer=1):
    if anticlockwise == True:
        x, y = _euler_curve(R0, angle1, angle2, num_seg)
        dx = 1e-9
        x = np.append(x, x[-1]+dx*np.cos(angle2/180*pi))
        y = np.append(y, y[-1]+dx*np.sin(angle2/180*pi))

        gap = np.linspace(gap1, gap2, len(x))   # gap随着曲线不同位置变化
        x2 = [x[0]]
        y2 = [y[0]-gap[0]-width_ring/2-width_bus/2]
        list_idx = range(len(x)-2)
        for idx in list_idx:
            slope_tangent = (y[idx+1]-y[idx])/(x[idx+1]-x[idx])
            slope_normal = -1/slope_tangent
            gap_euler = gap[idx] + width_ring/2 + width_bus/2
            x2 = np.append(
                x2, x[idx] + np.sqrt(gap_euler**2/(1+slope_normal**2)))
            y2 = np.append(
                y2, y[idx] - np.sqrt(gap_euler**2/(1+1/slope_normal**2)))
        x2 = np.append(x2, x[len(x)-1]+gap[len(x)-1]+width_ring/2+width_bus/2)
        y2 = np.append(y2, y[len(x)-1])

        # for small R0, the curve bends inwards, make adjustment to make sure the max x is the last element in the list
        max_index, max_value = max(enumerate(x2), key=operator.itemgetter(1))
        list_idx_max = range(max_index, len(x), 1)
        for idx in list_idx_max:
            x2[idx] = x2[max_index]

        angle_p = np.arctan((y2[round((1-p)*len(x2))+1] - y2[round((1-p)*len(x2))])/(
            x2[round((1-p)*len(x2))+1] - x2[round((1-p)*len(x2))]))/pi*180  # 起始点p处的角度

        P = Path(np.array((x2[round((1-p)*len(x2)):round(len(x2))],
                 y2[round((1-p)*len(x2)):round(len(x2))])).T)   # 根据x，y数组创建路径对象，p确定起始点是从x哪个位置开始，x0到max（x）的长度除以总的x长度是p
        P1 = P.extrude(width=width_bus)   #根据path拉伸出等宽的波导
        P1.flatten(single_layer=layer)
        port1 = P1.add_port(
            1, (x2[round((1-p)*len(x2))], y2[round((1-p)*len(x2))]), width_bus, angle_p+180)
        port2 = P1.add_port(2, (x2[-1], y2[-1]), width_bus, angle2)
        port3 = P1.add_port(
            3, (x2[round((1-p)*len(x2))], y2[round((1-p)*len(x2))]), width_bus, angle_p+90)
        port4 = P1.add_port(4, (x2[-1], y2[-1]), width_bus, angle2+90)

        # print('angle_p='+str(90-angle_p))
        return P1, 90-angle_p

    if anticlockwise == False:
        x, y = _euler_curve(R0, angle1, angle2)
        dx = 1e-9
        x = np.append(x, x[-1]+dx*np.cos(angle2/180*pi))
        # change here: + --> -
        y = -np.append(y, y[-1]+dx*np.sin(angle2/180*pi))

        gap = np.linspace(gap1, gap2, len(x))
        x2 = [x[0]]
        y2 = [y[0]+gap[0]+width_ring/2+width_bus/2]
        list_idx = range(len(x)-2)
        for idx in list_idx:
            slope_tangent = (y[idx+1]-y[idx])/(x[idx+1]-x[idx])
            slope_normal = -1/slope_tangent
            gap_euler = gap[idx] + width_ring/2 + width_bus/2
            x2 = np.append(
                x2, x[idx] + np.sqrt(gap_euler**2/(1+slope_normal**2)))
            # change here: - --> +
            y2 = np.append(
                y2, y[idx] + np.sqrt(gap_euler**2/(1+1/slope_normal**2)))
        x2 = np.append(x2, x[len(x)-1]+gap[len(x)-1]+width_ring/2+width_bus/2)
        y2 = np.append(y2, y[len(x)-1])

        # for small R0, the curve bends inwards, make adjustment to make sure the max x is the last element in the list
        max_index, max_value = max(enumerate(x2), key=operator.itemgetter(1))
        list_idx_max = range(max_index, len(x), 1)
        for idx in list_idx_max:
            x2[idx] = x2[max_index]

        angle_p = np.arctan((y2[round((1-p)*len(x2))+1] - y2[round((1-p)*len(x2))])/(
            x2[round((1-p)*len(x2))+1] - x2[round((1-p)*len(x2))]))/pi*180

        P = Path(np.array((x2[round((1-p)*len(x2)):round(len(x2))],
                 y2[round((1-p)*len(x2)):round(len(x2))])).T)
        P1 = P.extrude(width=width_bus)
        P1.flatten(single_layer=layer)
        port1 = P1.add_port(
            1, (x2[round((1-p)*len(x2))], y2[round((1-p)*len(x2))]), width_bus, angle_p+180)
        port2 = P1.add_port(2, (x2[-1], y2[-1]), width_bus, angle2+180)
        port3 = P1.add_port(
            3, (x2[round((1-p)*len(x2))], y2[round((1-p)*len(x2))]), width_bus, angle_p+270)
        port4 = P1.add_port(4, (x2[-1], y2[-1]), width_bus, angle2+90)

        # print('angle_p='+str(90+angle_p))
        return P1, 90+angle_p


# P1, angle_p=euler_bend_gap(width_ring=1, width_bus=2, R0=50, angle1=0, angle2=90, anticlockwise=True, gap1=2.8, gap2=0.8, p=0.5, layer=1)
# print(angle_p)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=True,zoom_factor=True,interactive_zoom=True)
# qp(P1)
#
# P1, angle_p=euler_bend_gap(width_ring=1, width_bus=2, R0=50, angle1=30, angle2=90, anticlockwise=True, gap1=2.8, gap2=0.8, p=0.5, layer=1)
# print (angle_p)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=False,zoom_factor=True,interactive_zoom=True)
# qp(P1)
#
# P1, angle_p=euler_bend_gap(width_ring=1, width_bus=2, R0=50, angle1=50, angle2=90, anticlockwise=True, gap1=2.8, gap2=0.8, p=0.5, layer=1)
# print (angle_p)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=True,zoom_factor=True,interactive_zoom=True)
# qp(P1)



"""
4.创建直波导
"""

def wg(width1=1, width2=2, length=10.0, layer=1):
    '''straight waveguide
    w: width, l: length
    '''
    P = Path()
    P.append(pp.straight(length=length))          #添加一条直线路径
    P1 = P.extrude(width=[width1, width2])
    P1.flatten(single_layer=layer)
    port1 = P1.add_port(1, (0, 0), width1, 180)
    port2 = P1.add_port(2, (length, 0), width2, 0)
    return P1

# P1=wg(width1=1, width2=2, length=10.0, layer=1)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=False,zoom_factor=True,interactive_zoom=True)
# qp(P1)
#
"""
5.创建圆环形波导
"""

def arc(width1=1, width2=2, angle1=0, angle2=90, R0=4, layer=1):
    '''arc waveguide
    R0: radius
    '''
    P = Path()
    P.append(pp.arc(radius=R0, angle=np.abs(angle2-angle1)))   # Circular arc

    P1 = P.extrude(width=[width1, width2])
    P1.flatten(single_layer=layer)
    port1 = P1.add_port(1, (0, 0), width1, 180)
    port2 = P1.add_port(2, (R0*np.sqrt(2-2*np.cos(np.abs(angle2-angle1)/180*pi))*np.cos((90-(180-np.abs(angle2-angle1))/2)/180*pi), R0 *
                        np.sqrt(2-2*np.cos(np.abs(angle2-angle1)/180*pi))*np.sin((90-(180-np.abs(angle2-angle1))/2)/180*pi)), width2, np.abs(angle2-angle1))
    return P1
#
#
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=True,zoom_factor=True,interactive_zoom=True)
# P1=arc(width1=1, width2=2, angle1=0, angle2=90, R0=4, layer=1)
# qp(P1)


"""
6. 创建十字形marker和十字指针形中间带方块
"""
#
def alignment_mark(width=5.0, length=100.0, center_square=False,
                   center_size=2.0, min_dimension=1.0,
                   bridge_length=3.0, layer=1):
    """generate a single cross alignment mark for ebeam lithography"""
    if center_square == False:                                  # 创建十字形marker
        S1 = pg.rectangle((length, width), layer=layer)    # 创建矩形几何
        S1.center = (0, 0)                                      # 创建一个矩形几何形状后，将其移动到新的位置
        S2 = pg.rectangle((width, length), layer=layer)
        S2.center = (0, 0)
        D = pg.boolean(S1, S2, 'a+b', layer=layer)
        D.name = 'mark'
        return D
    if center_square == True:                                  # 创建方形marker
        # add a small center square as a test of lithography resolution
        D = Device('mark')
        D.add_ref(pg.compass((center_size, center_size)))    # 创建方形marker
        BRIDGE = Device()
        BRIDGE.add_polygon([[-min_dimension/2, -width/2, -width/2, width/2, width/2, min_dimension/2],
                            [0, bridge_length, length/2-center_size, length/2-center_size, bridge_length, 0]])  # 通过顶点坐标x,y添加一个多边形形状
        D.add_ref(BRIDGE).move((0, center_size/2))
        D.add_ref(BRIDGE).mirror((0, 0), (1, 0)).move((0, -center_size/2))
        D.add_ref(BRIDGE).rotate(90).move((-center_size/2, 0))
        D.add_ref(BRIDGE).rotate(-90).move((center_size/2, 0))
        D.flatten(single_layer=layer)
        return D


# P1=alignment_mark(width=5.0, length=100.0, center_square=False,center_size=2.0, min_dimension=1.0,bridge_length=3.0, layer=1)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=False,zoom_factor=True,interactive_zoom=True)
# qp(P1)

"""
7.创建一个euler racetrack
"""

def racetrack_euler(w1=2, w2=0.8, l=461, R=150,num_seg=1000, layer=6):
    # racetrack formed by euler bends and straignt wg
    # w1 = racetrack width, w2 = width at the euler bend joint (far end of coupling)
    # l = length of straigh arms
    # R = radius of the euler bend

    D = Device()
    # racetrack
    WG1 = wg(width1=w1, width2=w1, length=l, layer=layer)
    WG2 = wg(width1=w1, width2=w1, length=l, layer=layer)
    BEND1 = euler_bend(width1=w1, width2=w1, R0=R, angle1=0,
                       angle2=90, num_seg=num_seg, anticlockwise=True, layer=layer)
    BEND2 = euler_bend(width1=w1, width2=w1, R0=R, angle1=0,
                       angle2=90, num_seg=num_seg, anticlockwise=False, layer=layer)
    BEND3 = euler_bend(width1=w1, width2=w2, R0=R, angle1=0,
                       angle2=90, num_seg=num_seg, anticlockwise=True, layer=layer)
    BEND4 = euler_bend(width1=w1, width2=w2, R0=R, angle1=0,
                       angle2=90, num_seg=num_seg, anticlockwise=False, layer=layer)

    bend1 = D << BEND1[0]
    bend2 = D << BEND2[0]
    wg1 = D << WG1
    bend3 = D << BEND3[0]
    bend4 = D << BEND4[0]
    wg2 = D << WG2

    bend1.rotate(0)
    bend1.movex(l/2)
    bend1.movey(0)
    bend2.connect(port=2, destination=bend1.ports[2])
    wg1.connect(port=2, destination=bend2.ports[1], overlap=0)
    bend3.connect(port=1, destination=wg1.ports[1])
    bend4.connect(port=2, destination=bend3.ports[2])
    wg2.connect(port=1, destination=bend4.ports[1])

    return D

#
# P1=racetrack_euler(w1=2, w2=10, l=461, R=150, layer=6)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=True,zoom_factor=True,interactive_zoom=True)
# qp(P1)


"""
8.创建一个由三个直波导，四个30度圆环连接成的y splitter
"""

def ysplitter(w=1, l=5, R=5, angle=30, layer=1):
    '''y splitter
    '''
    D = Device()
    wg0 = D << wg(width1=w, width2=w, length=l, layer=layer)
    wg1 = D << wg(width1=w, width2=w, length=l, layer=layer)
    wg2 = D << wg(width1=w, width2=w, length=l, layer=layer)
    arc1a = D << arc(width1=w, width2=w, angle1=0,
                     angle2=angle, R0=R, layer=layer)
    arc1b = D << arc(width1=w, width2=w, angle1=0,
                     angle2=angle, R0=R, layer=layer)
    arc2a = D << arc(width1=w, width2=w, angle1=0,
                     angle2=angle, R0=R, layer=layer)
    arc2b = D << arc(width1=w, width2=w, angle1=0,
                     angle2=angle, R0=R, layer=layer)

    arc1a.connect(port=1, destination=wg0.ports[2])
    arc1b.connect(port=2, destination=arc1a.ports[2])
    arc2a.connect(port=2, destination=wg0.ports[2])
    arc2b.connect(port=1, destination=arc2a.ports[1])
    wg1.connect(port=1, destination=arc1b.ports[1])
    wg2.connect(port=1, destination=arc2b.ports[2])

    port1 = D.add_port(1, wg0.ports[1].midpoint, w, 180)
    port2 = D.add_port(2, wg1.ports[2].midpoint, w, 0)
    port3 = D.add_port(3, wg2.ports[2].midpoint, w, 0)

    return D
#
#
# P1=ysplitter(w=1, l=5, R=5, angle=30, layer=1)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=False,zoom_factor=True,interactive_zoom=True)
# qp(P1)

"""
10.创建一个高度为height,长度为50的余弦曲线的bend连接
"""
def sbend(length=50.0, height=30.0, width=1.0, layer=0):
    '''s-bend.
    '''
    t = np.linspace(0, 1, 300)
    x = t*length                      # 参数t创建x值
    y = height/2*(1-np.cos(t*np.pi))  # 参数t创建0-pi之间的y值
    D = Path(np.array((x, y)).T)  # 将欧拉曲线的 x 和 y 坐标值数组转换为 Path 对象，并将其赋值给变量 P。现在，P 是一个包含了欧拉曲线路径信息的 Path 对象。
    D = D.extrude(width=width)
    D.flatten(single_layer=layer)
    port1 = D.add_port(1, (x[0], y[0]), width,180)
    port2 = D.add_port(2, (x[-1], y[-1]), width, 0)
    return D


# P1=sbend(length=50.0, height=30.0, width=1.0, layer=0)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=False,zoom_factor=True,interactive_zoom=True)
# qp(P1)



"""
11.用D.add_polygon函数创建一个轮廓满足余弦函数的taper,入口宽度为w1,出口宽度为30，长度为l
"""
def sbend_taper(w1=10.0, w2=30.0, l=200, layer=1):
    '''s-bend taper.
    '''
    D = Device()
    t = np.linspace(0, 1, 300)
    x = t*l
    y = (w2-w1)/4*(1-np.cos(t*np.pi))

    xpts = np.concatenate([x, np.flip(x)])            # 创建多边形轮廓的x坐标，从小到大，再从大到小
    ypts = np.concatenate([y+w1/2, -np.flip(y)-w1/2]) # 创建多边形轮廓的y坐标，从小到大，再从大到小
    xylist = np.column_stack((xpts, ypts))
    D.add_polygon(xylist, layer=1)
    D.flatten(single_layer=layer)
    D.add_port(1, (D.xmin, 0), w1, 180)
    D.add_port(2, (D.xmax, 0), w2, 0)
    return D
#
#
# D=sbend_taper(w1=10.0, w2=30.0, l=200, layer=1)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=False,zoom_factor=True,interactive_zoom=True)
# qp(D)


"""
12.创建coplanar waveguide电极，中心电极宽度w_center，边上两个电极宽度w_gnd，电极之间间隔为gap，长度为l
"""
def cpw(w_center=13.0, gap=7.0, w_gnd=180.0, l=500.0, layer=1):
    '''straignt cpw, oriented horizontally'''
    D = Device()
    GND = wg(w_gnd, w_gnd, l)
    CENTER = wg(w_center, w_center, l)  # center conductor
    c = D.add_ref(CENTER)
    D.add_ref(GND).movey(w_center/2+gap+w_gnd/2)
    D.add_ref(GND).movey(-w_center/2-gap-w_gnd/2)
    D.flatten(single_layer=layer)
    D.add_port(c.ports[1])
    D.add_port(c.ports[2])
    return D


# D=cpw(w_center=13.0, gap=7.0, w_gnd=180.0, l=500.0, layer=1)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=False,zoom_factor=True,interactive_zoom=True)
# qp(D)


#%%############################################################################
###############################################################################
# components = combination of basics, not edition

"""
13.alignment_mark_array创建四个或者八个十字形marker
"""

def alignment_mark_array(width=.5, length=300, center_square=False,
                         offset=[(300, 500), (300, -500),
                                 (-300, 500), (-300, -500)],
                         coordinates=[(-7000, -6000), (-7000, 6000),
                                      (7000, -6000), (7000, 6000)],
                         layer=1, double_marks=True):
    D = Device()
    AM = alignment_mark(width=width, length=length,
                        center_square=center_square, layer=layer)
    B = pg.rectangle((.005, .005), layer=99)
    B.center = (0, 0)

    D.add_ref(AM).center = coordinates[0]
    D.add_ref(AM).center = coordinates[1]
    D.add_ref(AM).center = coordinates[2]
    D.add_ref(AM).center = coordinates[3]

    D.add_ref(B).center = coordinates[0]
    D.add_ref(B).center = coordinates[1]
    D.add_ref(B).center = coordinates[2]
    D.add_ref(B).center = coordinates[3]

    if double_marks:
        D.add_ref(AM).center = np.add(coordinates[0], offset[0])
        D.add_ref(AM).center = np.add(coordinates[1], offset[1])
        D.add_ref(AM).center = np.add(coordinates[2], offset[2])
        D.add_ref(AM).center = np.add(coordinates[3], offset[3])
    D.flatten()
    return D
#
# D=alignment_mark_array(width=.5, length=300, center_square=False,
#                          offset=[(300, 500), (300, -500),
#                                  (-300, 500), (-300, -500)],
#                          coordinates=[(-7000, -6000), (-7000, 6000),
#                                       (7000, -6000), (7000, 6000)],
#                          layer=1, double_marks=False)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=False,zoom_factor=True,interactive_zoom=True)
# qp(D)
"""
14.generate_poling_finger创建由finger_shape决定顶端是round，pointy还是直接矩形的整个poling 电极
"""

def generate_poling_finger(period=10, w_finger=3.0, l_finger=20,
                           total_length=2e3, gap=20, lead_width=300,
                           finger_on_ground=0, finger_shape='round',
                           text_on=1, layer=1):
    #finger_on_ground = 0

    # period = 10 #poling period
    # w_finger = 3 #finger width
    # l_finger = 20 #finger width
    # total_length = 2e3  # total length

    # gap = 10 #gap between signal and ground
    # finger_shape can be 'rectangle', 'pointy', 'round'

    # lead_width = 300 #width of the lead that connects all the fingers
    lead_length = total_length

    D = Device()

    if finger_shape == 'round':
        FINGER = pg.rectangle((w_finger, l_finger-w_finger), layer)
        FINGER.add_ref(pg.circle(radius=w_finger/2)
                       ).movey(FINGER.ymax).movex(w_finger/2)
        FINGER = pg.union(FINGER, layer=layer)

    elif finger_shape == 'pointy':
        h_triangle = 2*w_finger
        FINGER = pg.rectangle((w_finger, l_finger-h_triangle), layer)
        FINGER.add_polygon([(FINGER.xmin, FINGER.ymax), (FINGER.xmax,
                           FINGER.ymax), (FINGER.x, FINGER.ymax+h_triangle)])
        FINGER = pg.union(FINGER, layer=layer)

    else:
        FINGER = pg.rectangle((w_finger, l_finger), layer)

    LEAD = pg.rectangle((lead_length, lead_width), layer)

    SINGLE = Device()
    for i in range(int(total_length//period)):
        SINGLE.add_ref(FINGER).movex(period*i)

    SINGLE.add_ref(LEAD).move((0, -lead_width))
    SINGLE.ymax = 0

    D.add_ref(SINGLE)

    if finger_on_ground:
        D.add_ref(SINGLE).mirror((1, 0)).movey(gap)
    else:
        D.add_ref(LEAD).movey(gap)

    if text_on:
        txt = 'p={:.0f} w={:.0f} lf={:.0f} l={:.1f}mm g = {:.1f}'.format(period, w_finger, l_finger,
                                                                         total_length/1e3, gap)
        xmin = D.xmin
        ymin = D.ymin
        t = D.add_ref(pg.text(text=txt, size=50,
                      justify='left', layer=13))
        t.move((xmin, ymin-t.ysize))

    D.flatten(single_layer=layer)

    return D
#
# D=generate_poling_finger(period=10, w_finger=3.0, l_finger=20,
#                            total_length=2e3, gap=20, lead_width=300,
#                            finger_on_ground=0, finger_shape='a',
#                            text_on=1, layer=1)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=False,zoom_factor=True,interactive_zoom=True)
# qp(D)


"""
15.generate_poling_finger_narrow_trace创建由finger_shape决定顶端是round，pointy还是直接矩形的整个poling 电极,和上一个的区别是多了两个伸出来的pad
"""
def generate_poling_finger_narrow_trace(period=10, w_finger=3.0, l_finger=20,
                                        total_length=2e3, gap=20, lead_width=50,
                                        pad_size=150, finger_on_ground=0, finger_shape='round',
                                        text_on=1, overlap=0, layer_metal=1, layer_ebeam_metal=5):
    '''
    small trace with pads on the side


    #finger_on_ground = 0

    # period = 10 #poling period
    # w_finger = 3 #finger width
    # l_finger = 20 #finger width
    # total_length = 2e3  # total length

    # gap = 10 #gap between signal and ground
    #finger_shape can be 'rectangle', 'pointy', 'round'
    #overlap between finger and lead for hybrid
    #lead_width = 50 #width of the lead that connects all the fingers
    '''
    lead_length = total_length

    D = Device()

    if finger_shape == 'round':
        FINGER = pg.rectangle(
            (w_finger, l_finger-w_finger/2+overlap), layer_metal)
        FINGER.add_ref(pg.circle(radius=w_finger/2)
                       ).movey(FINGER.ymax).movex(w_finger/2)
        FINGER = pg.union(FINGER, layer=layer_ebeam_metal)
        FINGER.movey(-overlap)

    elif finger_shape == 'pointy':
        h_triangle = 2*w_finger
        FINGER = pg.rectangle(
            (w_finger, l_finger-h_triangle+overlap), layer_metal)
        FINGER.add_polygon([(FINGER.xmin, FINGER.ymax), (FINGER.xmax,
                           FINGER.ymax), (FINGER.x, FINGER.ymax+h_triangle)])
        FINGER = pg.union(FINGER, layer=layer_ebeam_metal)
        FINGER.movey(-overlap)

    else:
        FINGER = pg.rectangle((w_finger, l_finger+overlap), layer_ebeam_metal)
        FINGER.movey(-overlap)

    LEAD = pg.rectangle((lead_length, lead_width), layer_metal)

    PAD = pg.rectangle((pad_size, pad_size), layer_metal)

    SINGLE = Device()
    for i in range(int(total_length//period)):
        SINGLE.add_ref(FINGER).movex(period*i)

    lead = SINGLE.add_ref(LEAD).move((0, -lead_width))

    SINGLE.ymax = 0

    # D.add_ref(SINGLE)
    #D.add_ref(PAD).move((PAD.xmin, PAD.ymax),(lead.xmax, lead.ymax))

    # if finger_on_ground:
    #     D.add_ref(SINGLE).mirror((1,0)).movey(gap)
    #     D.add_ref(PAD).move((PAD.xmax, PAD.ymin), (D.xmin, gap+l_finger))
    # else:
    #     D.add_ref(LEAD).movey(gap)
    #     D.add_ref(PAD).move((PAD.xmax, PAD.ymin), (D.xmin, gap))

    D.add_ref(SINGLE)
    D.add_ref(PAD).move((PAD.xmin+pad_size, PAD.ymax +
                         lead_width), (lead.xmax, lead.ymax))

    if finger_on_ground:
        D.add_ref(SINGLE).mirror((1, 0)).movey(gap)
        D.add_ref(PAD).move((PAD.xmax, PAD.ymin),
                            (D.xmin+pad_size, gap+l_finger+lead_width))
    else:
        D.add_ref(LEAD).movey(gap)
        D.add_ref(PAD).move((PAD.xmax, PAD.ymin),
                            (D.xmin+pad_size, gap+lead_width))

    if text_on:
        txt = 'p={:.2f} w={:.2f} lf={:.1f} \nl={:.0f} g = {:.1f}'.format(period, w_finger, l_finger,
                                                                         total_length, gap)
        xmin = (D.xmin+D.xmax)*0.5
        ymin = D.ymin+pad_size/2
        t = D.add_ref(pg.text(text=txt, size=30,
                      justify='center', layer=layer_text))
        t.move((xmin, ymin))

    D.flatten()

    return D
#
#
# D=generate_poling_finger_narrow_trace(period=10, w_finger=3.0, l_finger=20,
#                                         total_length=2e3, gap=20, lead_width=50,
#                                         pad_size=150, finger_on_ground=0, finger_shape='round',
#                                         text_on=1, overlap=0, layer_metal=1, layer_ebeam_metal=5)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=False,zoom_factor=True,interactive_zoom=True)
# qp(D)

"""
16.electrode_straight创建直电极
"""

def electrode_straight (lead_width=20, lead_length=100, pad_length=50, pad_width=100, gap=10, layer=6):

    D = Device()

    WG1 = wg(width1=lead_width, width2=lead_width,
             length=lead_length, layer=layer)
    WG2 = wg(width1=pad_width, width2=pad_width, length=pad_length, layer=layer)
    WG3 = wg(width1=lead_width, width2=lead_width,
             length=lead_length, layer=layer)
    WG4 = wg(width1=pad_width, width2=pad_width, length=pad_length, layer=layer)

    wg1 = D << WG1
    wg2 = D << WG2
    wg3 = D << WG3
    wg4 = D << WG4

    wg1.move((-lead_length/2, -lead_width/2-gap/2))
    wg2.move((-pad_length/2, -lead_width/2-gap/2-lead_width/2-pad_width/2))
    wg3.move((-lead_length/2, +lead_width/2+gap/2))
    wg4.move((-pad_length/2, lead_width/2+gap/2+lead_width/2+pad_width/2))

    return D
#
#
# D=electrode_straight(lead_width=20, lead_length=100, pad_length=50, pad_width=100, gap=10, layer=6)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=True,zoom_factor=True,interactive_zoom=True)
# qp(D)


"""
17.racetrack_euler_coupling_pulley创建
"""
def racetrack_euler_coupling_pulley(w1=1, w2=1, w_bus=1, l=461,l_stra_bus=100, R_arc=100, R_eul=200,angle_eu_c=60, num_seg=1000, g1=10, g2=5, p=0.5, layer=6):

    D = Device()
    PP1 = Device()

    # racetrack
    BEND1 = euler_bend(width1=w1, width2=w1, R0=R_eul, angle1=0,
                       angle2=90, num_seg=num_seg, anticlockwise=True, layer=layer)
    BEND2 = euler_bend(width1=w1, width2=w1, R0=R_eul, angle1=0,           # 参与耦合的racetrack那段弯曲处的宽度为w2 2um和直线处一样
                       angle2=90, num_seg=num_seg, anticlockwise=False, layer=layer)
    BEND3 = euler_bend(width1=w1, width2=w2, R0=R_eul, angle1=0,
                       angle2=90, num_seg=num_seg, anticlockwise=True, layer=layer)     # 不参与耦合的racetrack那段弯曲处的宽度为w2 0.8um
    BEND4 = euler_bend(width1=w1, width2=w2, R0=R_eul, angle1=0,           # 不参与耦合的racetrack那段弯曲处的宽度为w2 0.8um
                       angle2=90, num_seg=num_seg, anticlockwise=False, layer=layer)
    euler_len_total = BEND1[1]+BEND2[1]+BEND3[1]+BEND4[1]                  #计算racetrack弯曲部分总长度
    l_wg = (l-euler_len_total)/2                                           #计算racetrack单个直波导长度
    WG1 = wg(width1=w1, width2=w1, length=l_wg, layer=layer)
    WG2 = wg(width1=w1, width2=w1, length=l_wg, layer=layer)

    # gap around euler bend, i.e. the bus waveguide in the coupling region
    BUS_GAP1 = euler_bend_gap(width_ring=w1, width_bus=w_bus, R0=R_eul, angle1=angle_eu_c,
                              angle2=90, anticlockwise=True, num_seg=num_seg, gap1=g1, gap2=g2, p=p, layer=layer)
    BUS_GAP2 = euler_bend_gap(width_ring=w1, width_bus=w_bus, R0=R_eul, angle1=angle_eu_c,
                              angle2=90, anticlockwise=False, num_seg=num_seg, gap1=g1, gap2=g2, p=p, layer=layer)
    PP1 << BUS_GAP1[0]
    PP1 << BUS_GAP2[0]

    # bus waveguide
    BUS_ARC1 = arc(width1=w_bus, width2=w_bus, angle1=0,
                   angle2=BUS_GAP1[1], R0=R_arc, layer=layer)
    BUS_ARC2 = arc(width1=w_bus, width2=w_bus, angle1=0,
                   angle2=BUS_GAP1[1], R0=R_arc, layer=layer)
    BUS_WG1 = wg(width1=w_bus, width2=w_bus, length=l_stra_bus, layer=layer)
    BUS_WG2 = wg(width1=w_bus, width2=w_bus, length=l_stra_bus, layer=layer)

    bend1 = D << BEND1[0]
    bend2 = D << BEND2[0]
    wg1 = D << WG1
    bend3 = D << BEND3[0]
    bend4 = D << BEND4[0]
    wg2 = D << WG2

    bus_gap1 = D << BUS_GAP1[0]
    bus_gap2 = D << BUS_GAP2[0]
    bus_arc1 = D << BUS_ARC1
    bus_arc2 = D << BUS_ARC2
    bus_wg1 = D << BUS_WG1
    bus_wg2 = D << BUS_WG2

    bend1.rotate(0)
    bend1.movex(l_wg/2)       # bend1的起始点是位于原点，将它往右移动半个直波导长度，使得racetrack 位于x的中心
    bend1.movey(-BEND1[3])     # bend1的起始点是位于原点，将它往下移动euler bend的y方向高度，使得racetrack 位于y的中心
    bend2.connect(port=2, destination=bend1.ports[2])
    wg1.connect(port=2, destination=bend2.ports[1], overlap=0)
    bend3.connect(port=1, destination=wg1.ports[1])
    bend4.connect(port=2, destination=bend3.ports[2])
    wg2.connect(port=1, destination=bend4.ports[1])

    bus_gap1.connect(
        port=4, destination=bend1.ports[4], overlap=-g2 - w_bus/2 - w1/2)   # racetrack bend和耦合区域bend 1通过水平方向的端口以-g2-w_bus/2-w1/2的间隔对齐
    bus_gap2.connect(port=2, destination=bus_gap1.ports[2])
    bus_arc1.connect(port=1, destination=bus_gap1.ports[1])
    bus_arc2.connect(port=2, destination=bus_gap2.ports[1])
    bus_wg1.connect(port=1, destination=bus_arc1.ports[2])
    bus_wg2.connect(port=1, destination=bus_arc2.ports[1])

    port1 = D.add_port(1, bus_wg1.ports[2].midpoint, w_bus, 270)
    port2 = D.add_port(2, bus_wg2.ports[2].midpoint, w_bus, 90)

    return D, euler_len_total, l_wg, BUS_GAP1[1], BEND1[3], PP1


# D, euler_len_total, l_wg, BUS_GAP1_value,y_mincor, PP1 = racetrack_euler_coupling_pulley(w1=1, w2=1, w_bus=1, l=13010,l_stra_bus=103, R_arc=200, R_eul=128.7342, angle_eu_c=50, num_seg=10000, g1=2, g2=0.2, p=0.5, layer=6)
# print("BUS_GAP1[1] 的值为:", BUS_GAP1_value)
# print("racetrack straight waveguide length:", l_wg)
# print("racetrack euler waveguide length:", euler_len_total)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=False,zoom_factor=True,interactive_zoom=True)
# qp(PP1)a
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=True,zoom_factor=True,interactive_zoom=True)
# qp(D)

# D=racetrack_euler_coupling_pulley(w1=1, w2=1, w_bus=1, l=461,R0=150, R=100, g1=2.8, g2=0.8, p=0.5, layer=6)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=True,zoom_factor=True,interactive_zoom=True)
# qp(D)

"""
18.racetrack_electrode创建racetrack电极
"""

def racetrack_electrode (lead_length=3000, lead_width=25, pad_length=120, side_pad_width=75, overlap_length=3, gap=10, racetrack_width=200, layer=6):
    side_pad_width = side_pad_width+overlap_length
    middle_pad_width = racetrack_width-gap-2*lead_width+2*overlap_length
    D = Device()

    WG1 = wg(width1=lead_width, width2=lead_width,length=lead_length, layer=layer)
    WG2 = wg(width1=lead_width, width2=lead_width,length=lead_length, layer=layer)
    WG3 = wg(width1=lead_width, width2=lead_width,length=lead_length, layer=layer)
    WG4 = wg(width1=lead_width, width2=lead_width,length=lead_length, layer=layer)

    PAD1 = wg(width1=side_pad_width, width2=side_pad_width, length=pad_length, layer=layer)
    PAD2 = wg(width1=middle_pad_width, width2=middle_pad_width, length=pad_length, layer=layer)
    PAD3 = wg(width1=side_pad_width, width2=side_pad_width, length=pad_length, layer=layer)


    wg1 = D << WG1
    wg2 = D << WG2
    wg3 = D << WG3
    wg4 = D << WG4
    pad1 = D << PAD1
    pad2 = D << PAD2
    pad3 = D << PAD3


    wg1.move((-lead_length/2, -3*lead_width/2-gap-middle_pad_width/2+overlap_length))
    wg2.move((-lead_length/2, -lead_width/2-middle_pad_width/2+overlap_length))
    wg3.move((-lead_length/2, +lead_width/2+middle_pad_width/2-overlap_length))
    wg4.move((-lead_length/2, +3*lead_width/2+gap+middle_pad_width/2-overlap_length))


    pad1.move((-pad_length/2, -2*lead_width-gap-side_pad_width/2-middle_pad_width/2+2*overlap_length))
    pad2.move((-pad_length/2, 0))
    pad3.move((-pad_length/2, +side_pad_width/2+middle_pad_width/2+2*lead_width+gap-2*overlap_length))
    # D.movey((0,-gap/2-lead_width/2-middle_pad_width/2))
    return D


# D=racetrack_electrode(lead_length=3000, lead_width=25, pad_length=120, side_pad_width=75, overlap_length=5, gap=13,racetrack_width=200, layer=6)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=True,zoom_factor=True,interactive_zoom=True)
# qp(D)

"""
19.racetrack_electrode创建Z cut poling finger
"""
def generate_poling_finger_z_cut (period=12, w_finger=6.0, l_finger=20,
                                  total_length=2e3, lead_width=20,
                                  pad_size=150, finger_shape='round', overlap_finger=3, overlap_pad=30, layer_metal=0, layer_ebeam_metal=1):
    l_finger=l_finger+w_finger/2
    lead_length = total_length

    D = Device()

    if finger_shape == 'round':
        FINGER = pg.rectangle(
            (w_finger, l_finger-w_finger/2+overlap_finger), layer_metal)
        FINGER.add_ref(pg.circle(radius=w_finger/2)
                       ).movey(FINGER.ymax).movex(w_finger/2)
        FINGER = pg.union(FINGER, layer=layer_ebeam_metal)
        FINGER.movey(-overlap_finger)

    elif finger_shape == 'pointy':
        h_triangle = 2*w_finger
        FINGER = pg.rectangle(
            (w_finger, l_finger-h_triangle+overlap_finger), layer_metal)
        FINGER.add_polygon([(FINGER.xmin, FINGER.ymax), (FINGER.xmax,
                           FINGER.ymax), (FINGER.x, FINGER.ymax+h_triangle)])
        FINGER = pg.union(FINGER, layer=layer_ebeam_metal)
        FINGER.movey(-overlap_finger)

    else:
        FINGER = pg.rectangle((w_finger, l_finger+overlap_finger), layer_ebeam_metal)
        FINGER.movey(-overlap_finger)

    LEAD = pg.rectangle((lead_length, lead_width), layer_metal)

    PAD = pg.rectangle((pad_size, pad_size+overlap_pad), layer_metal)

    SINGLE = Device()
    for i in range(int(total_length//period)):
        SINGLE.add_ref(FINGER).movex(period*i)

    lead = SINGLE.add_ref(LEAD).move((0, -lead_width))

    SINGLE.ymax = 0


    D.add_ref(SINGLE)
    D.add_ref(PAD).move((PAD.xmin+pad_size, PAD.ymax +
                         lead_width), (lead.xmax, lead.ymax+overlap_pad))

    # if finger_on_ground:                      # 加上接地的对称一半电极
    #     D.add_ref(SINGLE).mirror((1, 0)).movey(gap)
    #     D.add_ref(PAD).move((PAD.xmax, PAD.ymin),
    #                         (D.xmin+pad_size, gap+l_finger+lead_width))
    # else:
    #     D.add_ref(LEAD).movey(gap)
    #     D.add_ref(PAD).move((PAD.xmax, PAD.ymin),
    #                         (D.xmin+pad_size, gap+lead_width))

    # if text_on:
    #     txt = 'p={:.2f} w={:.2f} lf={:.1f} \nl={:.0f} g = {:.1f}'.format(period, w_finger, l_finger,
    #                                                                      total_length, gap)
    #     xmin = (D.xmin+D.xmax)*0.5
    #     ymin = D.ymin+pad_size/2
    #     t = D.add_ref(pg.text(text=txt, size=30,
    #                   justify='center', layer=layer_text))
    #     t.move((xmin, ymin))

    D.flatten()

    return D, l_finger-overlap_finger, overlap_finger


# D, l_finger, overlap_finger=generate_poling_finger_z_cut (period=12, w_finger=6.0, l_finger=20,
#                                   total_length=2e3, lead_width=20,
#                                   pad_size=150, finger_shape='round',
#                                   overlap_finger=3,overlap_pad=5, layer_metal=0, layer_ebeam_metal=1)
# print (l_finger)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=True,zoom_factor=True,interactive_zoom=True)
# qp(D)


"""
19.创建整个device的marker,20um小方格
"""
def s20_device_marker(size=20, gap=20,region=1,x0=0, y0=0, layer=1):

    D = Device()
    if region == 1:
        WG1 = wg(width1=size, width2=size, length=size, layer=layer)
        WG2 = wg(width1=size, width2=size, length=size, layer=layer)
        WG3 = wg(width1=size, width2=size, length=size, layer=layer)
        WG4 = wg(width1=size, width2=size, length=size, layer=layer)
        WG5 = wg(width1=size, width2=size, length=size, layer=layer)
        wg1 = D << WG1
        wg2 = D << WG2
        wg3 = D << WG3
        wg4 = D << WG4
        wg5 = D << WG5
        wg1.move((x0, y0 + size / 2))
        wg2.move((x0, y0 + gap + 3 * size / 2))
        wg3.move((x0 + gap + size, y0 + size / 2))
        wg4.move((x0 + gap + size, y0 + gap + 3 * size / 2))
        wg5.move((x0 + 2 * gap + 2 * size, y0 + size / 2))
    elif region == 2:
        WG1 = wg(width1=size, width2=size, length=size, layer=layer)
        WG2 = wg(width1=size, width2=size, length=size, layer=layer)
        WG3 = wg(width1=size, width2=size, length=size, layer=layer)
        WG4 = wg(width1=size, width2=size, length=size, layer=layer)
        WG5 = wg(width1=size, width2=size, length=size, layer=layer)
        WG6 = wg(width1=size, width2=size, length=size, layer=layer)
        wg1 = D << WG1
        wg2 = D << WG2
        wg3 = D << WG3
        wg4 = D << WG4
        wg5 = D << WG5
        wg6 = D << WG6
        wg1.move((x0-size, y0 + size / 2))
        wg2.move((x0-size, y0 + gap + 3 * size / 2))
        wg3.move((x0 - gap - 2*size, y0 + size / 2))
        wg4.move((x0 - gap - 2*size, y0 + gap + 3 * size / 2))
        wg5.move((x0 - 2 * gap - 3 * size, y0 + size / 2))
        wg6.move((x0 - 2 * gap - 3 * size, y0 + gap + 3 * size / 2))
    elif region == 3:
        WG1 = wg(width1=size, width2=size, length=size, layer=layer)
        WG2 = wg(width1=size, width2=size, length=size, layer=layer)
        WG3 = wg(width1=size, width2=size, length=size, layer=layer)
        WG4 = wg(width1=size, width2=size, length=size, layer=layer)
        WG5 = wg(width1=size, width2=size, length=size, layer=layer)
        WG6 = wg(width1=size, width2=size, length=size, layer=layer)
        WG7 = wg(width1=size, width2=size, length=size, layer=layer)
        wg1 = D << WG1
        wg2 = D << WG2
        wg3 = D << WG3
        wg4 = D << WG4
        wg5 = D << WG5
        wg6 = D << WG6
        wg7 = D << WG7
        wg1.move((x0 - size, y0 - size / 2))
        wg2.move((x0 - size, y0 - gap - 3 * size / 2))
        wg3.move((x0 - gap - 2 * size, y0 - size / 2))
        wg4.move((x0 - gap - 2 * size, y0 - gap - 3 * size / 2))
        wg5.move((x0 - 2 * gap - 3 * size, y0 - size / 2))
        wg6.move((x0 - 2 * gap - 3 * size, y0 - gap - 3 * size / 2))
        wg7.move((x0 - 3 * gap - 4 * size, y0 - size / 2))
    elif region == 4:
        WG1 = wg(width1=size, width2=size, length=size, layer=layer)
        WG2 = wg(width1=size, width2=size, length=size, layer=layer)
        WG3 = wg(width1=size, width2=size, length=size, layer=layer)
        WG4 = wg(width1=size, width2=size, length=size, layer=layer)
        WG5 = wg(width1=size, width2=size, length=size, layer=layer)
        WG6 = wg(width1=size, width2=size, length=size, layer=layer)
        WG7 = wg(width1=size, width2=size, length=size, layer=layer)
        WG8 = wg(width1=size, width2=size, length=size, layer=layer)
        wg1 = D << WG1
        wg2 = D << WG2
        wg3 = D << WG3
        wg4 = D << WG4
        wg5 = D << WG5
        wg6 = D << WG6
        wg7 = D << WG7
        wg8 = D << WG8
        wg1.move((x0 - size, y0 - size / 2))
        wg2.move((x0 - size, y0 - gap - 3 * size / 2))
        wg3.move((x0 - gap - 2 * size, y0 - size / 2))
        wg4.move((x0 - gap - 2 * size, y0 - gap - 3 * size / 2))
        wg5.move((x0 - 2 * gap - 3 * size, y0 - size / 2))
        wg6.move((x0 - 2 * gap - 3 * size, y0 - gap - 3 * size / 2))
        wg7.move((x0 - 3 * gap - 4 * size, y0 - size / 2))
        wg8.move((x0 - 3 * gap - 4 * size,  y0 - gap - 3 * size / 2))

    return D

# D=s20_device_marker(size=20, gap=20,region=4,x0=0, y0=0, layer=1)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=True,zoom_factor=True,interactive_zoom=True)
# qp(D)


"""
20.创建整个device的global marker,4个20um小方格
"""
def s20_device_marker(size=20, gap=20,region=1,x0=0, y0=0, layer=1):

    D = Device()
    if region == 1:
        WG1 = wg(width1=size, width2=size, length=size, layer=layer)
        WG2 = wg(width1=size, width2=size, length=size, layer=layer)
        WG3 = wg(width1=size, width2=size, length=size, layer=layer)
        WG4 = wg(width1=size, width2=size, length=size, layer=layer)
        WG5 = wg(width1=size, width2=size, length=size, layer=layer)
        wg1 = D << WG1
        wg2 = D << WG2
        wg3 = D << WG3
        wg4 = D << WG4
        wg5 = D << WG5
        wg1.move((x0, y0 + size / 2))
        wg2.move((x0, y0 + gap + 3 * size / 2))
        wg3.move((x0 + gap + size, y0 + size / 2))
        wg4.move((x0 + gap + size, y0 + gap + 3 * size / 2))
        wg5.move((x0 + 2 * gap + 2 * size, y0 + size / 2))
    elif region == 2:
        WG1 = wg(width1=size, width2=size, length=size, layer=layer)
        WG2 = wg(width1=size, width2=size, length=size, layer=layer)
        WG3 = wg(width1=size, width2=size, length=size, layer=layer)
        WG4 = wg(width1=size, width2=size, length=size, layer=layer)
        WG5 = wg(width1=size, width2=size, length=size, layer=layer)
        WG6 = wg(width1=size, width2=size, length=size, layer=layer)
        wg1 = D << WG1
        wg2 = D << WG2
        wg3 = D << WG3
        wg4 = D << WG4
        wg5 = D << WG5
        wg6 = D << WG6
        wg1.move((x0-size, y0 + size / 2))
        wg2.move((x0-size, y0 + gap + 3 * size / 2))
        wg3.move((x0 - gap - 2*size, y0 + size / 2))
        wg4.move((x0 - gap - 2*size, y0 + gap + 3 * size / 2))
        wg5.move((x0 - 2 * gap - 3 * size, y0 + size / 2))
        wg6.move((x0 - 2 * gap - 3 * size, y0 + gap + 3 * size / 2))
    elif region == 3:
        WG1 = wg(width1=size, width2=size, length=size, layer=layer)
        WG2 = wg(width1=size, width2=size, length=size, layer=layer)
        WG3 = wg(width1=size, width2=size, length=size, layer=layer)
        WG4 = wg(width1=size, width2=size, length=size, layer=layer)
        WG5 = wg(width1=size, width2=size, length=size, layer=layer)
        WG6 = wg(width1=size, width2=size, length=size, layer=layer)
        WG7 = wg(width1=size, width2=size, length=size, layer=layer)
        wg1 = D << WG1
        wg2 = D << WG2
        wg3 = D << WG3
        wg4 = D << WG4
        wg5 = D << WG5
        wg6 = D << WG6
        wg7 = D << WG7
        wg1.move((x0 - size, y0 - size / 2))
        wg2.move((x0 - size, y0 - gap - 3 * size / 2))
        wg3.move((x0 - gap - 2 * size, y0 - size / 2))
        wg4.move((x0 - gap - 2 * size, y0 - gap - 3 * size / 2))
        wg5.move((x0 - 2 * gap - 3 * size, y0 - size / 2))
        wg6.move((x0 - 2 * gap - 3 * size, y0 - gap - 3 * size / 2))
        wg7.move((x0 - 3 * gap - 4 * size, y0 - size / 2))
    elif region == 4:
        WG1 = wg(width1=size, width2=size, length=size, layer=layer)
        WG2 = wg(width1=size, width2=size, length=size, layer=layer)
        WG3 = wg(width1=size, width2=size, length=size, layer=layer)
        WG4 = wg(width1=size, width2=size, length=size, layer=layer)
        WG5 = wg(width1=size, width2=size, length=size, layer=layer)
        WG6 = wg(width1=size, width2=size, length=size, layer=layer)
        WG7 = wg(width1=size, width2=size, length=size, layer=layer)
        WG8 = wg(width1=size, width2=size, length=size, layer=layer)
        wg1 = D << WG1
        wg2 = D << WG2
        wg3 = D << WG3
        wg4 = D << WG4
        wg5 = D << WG5
        wg6 = D << WG6
        wg7 = D << WG7
        wg8 = D << WG8
        wg1.move((x0 - size, y0 - size / 2))
        wg2.move((x0 - size, y0 - gap - 3 * size / 2))
        wg3.move((x0 - gap - 2 * size, y0 - size / 2))
        wg4.move((x0 - gap - 2 * size, y0 - gap - 3 * size / 2))
        wg5.move((x0 - 2 * gap - 3 * size, y0 - size / 2))
        wg6.move((x0 - 2 * gap - 3 * size, y0 - gap - 3 * size / 2))
        wg7.move((x0 - 3 * gap - 4 * size, y0 - size / 2))
        wg8.move((x0 - 3 * gap - 4 * size,  y0 - gap - 3 * size / 2))

    return D

"""
21.创建整个device的global marker,3个20um小方格，一个50 um大方格
"""
def sqm(width=2, layer=1):              # 创建中心在原点的方块

    P = Path()
    P.append(pp.straight(length=width))          #添加一条直线路径
    P1 = P.extrude(width=[width, width])
    P1.flatten(single_layer=layer)
    P1.move((-width/2,0))
    return P1


# P = sqm(width=2, layer=1)
# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=True,zoom_factor=True,interactive_zoom=True)
# qp(P)

def s20_50_global_marker(size1=20,size2=50,distance=50,gap=10,region=1,x0=0, y0=0, layer=1):

    x1=x0+distance+size1/2
    y1=y0+distance+size1/2
    x2=x0+distance+size1+gap+size2/2
    y2=y0+distance+size1/2
    x3=x0+distance+size1+gap+size1/2
    y3=y0+distance+size1+size2/2+gap
    x4=x0+distance+5*size1/2+2*gap
    y4=y0+distance+size1+size2/2+gap
    D = Device()
    if region == 1:
        SQ1 = sqm(width=size1, layer=layer)
        SQ2 = sqm(width=size2, layer=layer)
        SQ3 = sqm(width=size1, layer=layer)
        SQ4 = sqm(width=size1, layer=layer)
        sq1 = D << SQ1
        sq2 = D << SQ2
        sq3 = D << SQ3
        sq4 = D << SQ4

        sq1.move((x1, y1))
        sq2.move((x2, y2))
        sq3.move((x3, y3))
        sq4.move((x4, y4))
    elif region == 2:
        SQ1 = sqm(width=size1, layer=layer)
        SQ2 = sqm(width=size2, layer=layer)
        SQ3 = sqm(width=size1, layer=layer)
        SQ4 = sqm(width=size1, layer=layer)
        sq1 = D << SQ1
        sq2 = D << SQ2
        sq3 = D << SQ3
        sq4 = D << SQ4
        sq1.move((-x1, y1))
        sq2.move((-x2, y2))
        sq3.move((-x3, y3))
        sq4.move((-x4, y4))
    elif region == 3:
        SQ1 = sqm(width=size1, layer=layer)
        SQ2 = sqm(width=size2, layer=layer)
        SQ3 = sqm(width=size1, layer=layer)
        SQ4 = sqm(width=size1, layer=layer)
        sq1 = D << SQ1
        sq2 = D << SQ2
        sq3 = D << SQ3
        sq4 = D << SQ4
        sq1.move((-x1, -y1))
        sq2.move((-x2, -y2))
        sq3.move((-x3, -y3))
        sq4.move((-x4, -y4))
    elif region == 4:
        SQ1 = sqm(width=size1, layer=layer)
        SQ2 = sqm(width=size2, layer=layer)
        SQ3 = sqm(width=size1, layer=layer)
        SQ4 = sqm(width=size1, layer=layer)
        sq1 = D << SQ1
        sq2 = D << SQ2
        sq3 = D << SQ3
        sq4 = D << SQ4
        sq1.move((x1, -y1))
        sq2.move((x2, -y2))
        sq3.move((x3, -y3))
        sq4.move((x4, -y4))

    return D

# layer_global_marker=1
# D = Device()
# device_marker1 = s20_50_global_marker(size1=20,size2=50,distance=80,gap=10,region=1,x0=0, y0=0, layer=layer_global_marker)#
# device_marker2 = s20_50_global_marker(size1=20,size2=50,distance=80,gap=10,region=2,x0=0, y0=0, layer=layer_global_marker)#
# device_marker3 = s20_50_global_marker(size1=20,size2=50,distance=80,gap=10,region=3,x0=0, y0=0, layer=layer_global_marker)#
# device_marker4 = s20_50_global_marker(size1=20,size2=50,distance=80,gap=10,region=4,x0=0, y0=0, layer=layer_global_marker)#
# D << device_marker1
# D << device_marker2
# D << device_marker3
# D << device_marker4

"""
22.创建racetrack的local marker,3个20um小方格
"""
def s20_racetrack_marker(size1=20,distance_x=50,distance_y=50,gap=500,gap2=100,region=1,x0=0, y0=0, layer=1):

    x1=x0+distance_x+size1/2
    y1=y0+distance_y+size1/2
    x2=x0+distance_x+3*size1/2+gap
    y2=y0+distance_y+size1/2
    x3=x0+distance_x+size1/2
    y3=y0+distance_y+3*size1/2+gap

    D = Device()
    if region == 1:
        SQ1 = sqm(width=size1, layer=layer)
        SQ2 = sqm(width=size1, layer=layer)
        SQ3 = sqm(width=size1, layer=layer)
        sq1 = D << SQ1
        sq2 = D << SQ2
        sq3 = D << SQ3

        sq1.move((x1, y1))
        sq2.move((x2, y2))
        sq3.move((x3, y3))

        CROSS = alignment_mark(width=5.0, length=50.0, center_square=False,center_size=2.0, min_dimension=1.0,bridge_length=3.0, layer=layer)
        cross = D << CROSS
        cross.move((x1+gap2, y1+gap2))

    elif region == 2:
        SQ1 = sqm(width=size1, layer=layer)
        SQ2 = sqm(width=size1, layer=layer)
        SQ3 = sqm(width=size1, layer=layer)
        sq1 = D << SQ1
        sq2 = D << SQ2
        sq3 = D << SQ3

        sq1.move((-x1, y1))
        sq2.move((-x2, y2))
        sq3.move((-x3, y3))

        CROSS = alignment_mark(width=5.0, length=50.0, center_square=False,center_size=2.0, min_dimension=1.0,bridge_length=3.0, layer=layer)
        cross = D << CROSS
        cross.move((-x1-gap2, y1+gap2))
    elif region == 3:
        SQ1 = sqm(width=size1, layer=layer)
        SQ2 = sqm(width=size1, layer=layer)
        SQ3 = sqm(width=size1, layer=layer)
        sq1 = D << SQ1
        sq2 = D << SQ2
        sq3 = D << SQ3

        sq1.move((-x1, -y1))
        sq2.move((-x2, -y2))
        sq3.move((-x3, -y3))

        CROSS = alignment_mark(width=5.0, length=50.0, center_square=False,center_size=2.0, min_dimension=1.0,bridge_length=3.0, layer=layer)
        cross = D << CROSS
        cross.move((-x1-gap2, -y1-gap2))
    elif region == 4:
        SQ1 = sqm(width=size1, layer=layer)
        SQ2 = sqm(width=size1, layer=layer)
        SQ3 = sqm(width=size1, layer=layer)
        sq1 = D << SQ1
        sq2 = D << SQ2
        sq3 = D << SQ3

        sq1.move((x1, -y1))
        sq2.move((x2, -y2))
        sq3.move((x3, -y3))

        CROSS = alignment_mark(width=5.0, length=50.0, center_square=False,center_size=2.0, min_dimension=1.0,bridge_length=3.0, layer=layer)
        cross = D << CROSS
        cross.move((x1+gap2, -y1-gap2))

    return D

layer_global_marker=1
D = Device()
racetrack_marker1 = s20_racetrack_marker(size1=20,distance_x=50,distance_y=50,gap=50,gap2=100,region=1,x0=0, y0=0, layer=1)#
racetrack_marker2 = s20_racetrack_marker(size1=20,distance_x=50,distance_y=50,gap=50,gap2=100,region=2,x0=0, y0=0, layer=1)#
racetrack_marker3 = s20_racetrack_marker(size1=20,distance_x=50,distance_y=50,gap=50,gap2=100,region=3,x0=0, y0=0, layer=1)#
racetrack_marker4 = s20_racetrack_marker(size1=20,distance_x=50,distance_y=50,gap=50,gap2=100,region=4,x0=0, y0=0, layer=1)#
D << racetrack_marker1
D << racetrack_marker2
D << racetrack_marker3
D << racetrack_marker4

# set_quickplot_options(show_ports=True,show_subports=True,label_aliases=True,new_window=True,blocking=True,zoom_factor=True,interactive_zoom=True)
# qp(D)

"""
22.创建ring resonator 包含耦合区域
"""
def ring_resonator(width=1, R_ring=100, gap_s=0.3, gap_e=6, layer=1):

    D = Device()
    ring = arc(width1=width, width2=width, angle1=0, angle2=360, R0=R_ring, layer=layer)
    Ring = D << ring.move((0, -R_ring))

    arc_gap2 = np.arccos(((R_ring+width/2)-(gap_e-gap_s)/2)/(R_ring+width/2))
    degrees_arc_gap2 = np.degrees(arc_gap2)
    arc_coupling = arc(width1=width, width2=width, angle1=-degrees_arc_gap2, angle2=degrees_arc_gap2, R0=R_ring, layer=layer)
    arc_coupling.rotate(-degrees_arc_gap2,center=(0, R_ring))
    arc_coupling.move((0, width/2))
    arc_coupling.mirror((0, 0), (1, 0))
    arc_coupling.move((0, -R_ring-width / 2-gap_s))
    a_coupling = D << arc_coupling

    arc_coupling_left = arc(width1=width, width2=width, angle1=-degrees_arc_gap2/2, angle2=degrees_arc_gap2/2, R0=2*R_ring,layer=layer)
    a_coupling_left = D << arc_coupling_left.mirror((0, 0), (1, 0))
    arc_coupling_right= arc(width1=width, width2=width, angle1=-degrees_arc_gap2/2, angle2=degrees_arc_gap2/2, R0=2*R_ring,layer=layer)
    a_coupling_right = D << arc_coupling_right.mirror((0, 0), (1, 0))

    arc_coupling_left1 = arc(width1=width, width2=width, angle1=-degrees_arc_gap2/2, angle2=degrees_arc_gap2/2, R0=2*R_ring,layer=layer)
    a_coupling_left1 = D << arc_coupling_left1
    arc_coupling_right1= arc(width1=width, width2=width, angle1=-degrees_arc_gap2/2, angle2=degrees_arc_gap2/2, R0=2*R_ring,layer=layer)
    a_coupling_right1 = D << arc_coupling_right1

    arc_coupling_left2 = arc(width1=width, width2=width, angle1=0, angle2=degrees_arc_gap2, R0=2*R_ring,layer=layer)
    a_coupling_left2 = D << arc_coupling_left2
    arc_coupling_right2= arc(width1=width, width2=width, angle1=0, angle2=degrees_arc_gap2, R0=2*R_ring,layer=layer)
    a_coupling_right2 = D << arc_coupling_right2

    # WG1 = wg(width1=width, width2=width, length=300, layer=layer)
    # wg1 = D << WG1
    # wg1.connect(port=2, destination=ring_resonator.ports[1])

    # euler_bend_left2,euler_len, euler_detax, euler_detay =  euler_bend(width1=width, width2=width, R0=3*R_ring, angle1=0, angle2=90-3*degrees_arc_gap2, num_seg=1000, anticlockwise=True, layer=layer)
    # E_coupling_left2 = D << euler_bend_left2
    # euler_bend_right2,euler_len, euler_detax, euler_detay =  euler_bend(width1=width, width2=width, R0=3*R_ring, angle1=0, angle2=90-3*degrees_arc_gap2, num_seg=1000, anticlockwise=False, layer=layer)
    # E_coupling_right2 = D << euler_bend_right2.mirror((0, 0), (1, 0))

    a_coupling_left.connect(port=2, destination=a_coupling.ports[1])
    a_coupling_right.connect(port=1, destination=a_coupling.ports[2])
    a_coupling_left1.connect(port=2, destination=a_coupling_left.ports[1])
    a_coupling_right1.connect(port=1, destination=a_coupling_right.ports[2])
    a_coupling_left2.connect(port=2, destination=a_coupling_left1.ports[1])
    a_coupling_right2.connect(port=1, destination=a_coupling_right1.ports[2])
    # a_coupling = D << arc_coupling.rotate(-degrees_arc_gap2)
    port1 = D.add_port(1, a_coupling_left2.ports[1].midpoint, width, 180)
    port2 = D.add_port(2, a_coupling_right2.ports[2].midpoint, width, 0)

    return D

"""
22.创建1550nm TE mode grating 包含耦合区域
"""
def TE_grating_coupler(gap = .4):
    GR = pg.import_gds('Grating_model_wu.gds')
    D = Device()
    G = pg.extract(GR, [54]); G.flatten(single_layer=layer_grating)
    D << G
    port1 = D.add_port(1, (-190.5, -545.7578), 0.8, 90)
    return D



# D = s20_50_global_marker(size1=20,size2=50, region=3,x0=0, y0=0, layer=1)
# D.write_gds('alignment_mark.gds', unit=1e-6, precision=1e-9)

"""
23.创建circular poling finger
"""


def circular_poling_finger(R_ring=89, length=20,circle_pad_width=10, square_pad_width=100,square_pad_rotation = -90, mode_number=46, duty_cycle=0.3, layer=186):
    D = Device()
    ang1 = duty_cycle*360 / mode_number
    ang2 = 360 / mode_number
    for i in range(mode_number):
        arc_element = arc(width1=length, width2=length, angle1=0, angle2=ang1, R0=R_ring, layer=layer)
        arc_element.mirror((0, 0), (1, 0))
        arc_element.move((0, R_ring))
        arc_element.rotate(ang2*i,center=(0, 0))
        D << arc_element

    CIR_pad_radius = R_ring+length/2+circle_pad_width/2
    CIR_pad = arc(width1=circle_pad_width, width2=circle_pad_width, angle1=0, angle2=360, R0 = CIR_pad_radius, layer=187)
    D << CIR_pad.move((0, -CIR_pad_radius))

    pad_ang = np.arcsin(square_pad_width/2/CIR_pad_radius)
    pad_ang_degrees = np.degrees(pad_ang)*2
    square_pad_radius = R_ring + length / 2 + circle_pad_width / 2+square_pad_width/2
    arc_square_pad = arc(width1=square_pad_width, width2=square_pad_width, angle1=0, angle2=pad_ang_degrees, R0=square_pad_radius, layer=188)
    arc_square_pad.move((0, -square_pad_radius))
    arc_square_pad.rotate(-pad_ang_degrees/2,center=(0, 0))
    arc_square_pad.rotate(square_pad_rotation, center=(0, 0))
    # square_pad = wg(width1=square_pad_width, width2=square_pad_width, length=square_pad_width, layer=1)
    D << arc_square_pad
    return D
#

"""
24. 创建全局的marker
"""
#
def device_global_marker(layer=1, move_x=0, move_y=0):
    
                                      # 创建十字形marker
    a = 200
    width=2.0 
    length=200.0
    S1 = pg.rectangle((length, width), layer=layer)    # 创建矩形几何
    S1.center = (move_x, move_y)                                      # 创建一个矩形几何形状后，将其移动到新的位置
    S2 = pg.rectangle((width, length), layer=layer)
    S2.center = (move_x, move_y)
    D = pg.boolean(S1, S2, 'a+b', layer=layer)
    s_big = pg.rectangle((100, 100), layer=layer)
    D << s_big.move((150+move_x,-50+move_y))
    s_mid1 = pg.rectangle((20, 20), layer=layer)
    D << s_mid1.move((290+move_x,-10+move_y))
    s_mid2 = pg.rectangle((20, 20), layer=layer)
    D << s_mid2.move((190+move_x,90+move_y))
    s_small = pg.rectangle((10, 10), layer=layer)
    D << s_small.move((295+move_x,95+move_y))
    #
    S1 = pg.rectangle((length, width), layer=layer)    # 创建矩形几何
    S1.center = (-move_x-a, -move_y)                                      # 创建一个矩形几何形状后，将其移动到新的位置
    S2 = pg.rectangle((width, length), layer=layer)
    S2.center = (-move_x-a, -move_y)
    D1 = pg.boolean(S1, S2, 'a+b', layer=layer)
    s_big = pg.rectangle((100, 100), layer=layer)
    D1 << s_big.move((-250-move_x-a,50-100-move_y))
    s_mid1 = pg.rectangle((20, 20), layer=layer)
    D1 << s_mid1.move((-310-move_x-a,10-20-move_y))
    s_mid2 = pg.rectangle((20, 20), layer=layer)
    D1 << s_mid2.move((-210-move_x-a,-90-20-move_y))
    s_small = pg.rectangle((10, 10), layer=layer)
    D1 << s_small.move((-305-move_x-a,-95-10-move_y))
    D << D1

    #
    S1 = pg.rectangle((length, width), layer=layer)    # 创建矩形几何
    S1.center = (move_x, -move_y)                                      # 创建一个矩形几何形状后，将其移动到新的位置
    S2 = pg.rectangle((width, length), layer=layer)
    S2.center = (move_x, -move_y)
    D2 = pg.boolean(S1, S2, 'a+b', layer=layer)
    s_big = pg.rectangle((100, 100), layer=layer)
    D2 << s_big.move((150+move_x,50-100-move_y))
    s_mid1 = pg.rectangle((20, 20), layer=layer)
    D2 << s_mid1.move((290+move_x,10-20-move_y))
    s_mid2 = pg.rectangle((20, 20), layer=layer)
    D2 << s_mid2.move((190+move_x,-90-20-move_y))
    s_small = pg.rectangle((10, 10), layer=layer)
    D2 << s_small.move((295+move_x,-95-10-move_y))
    D << D2

    #
    S1 = pg.rectangle((length, width), layer=layer)    # 创建矩形几何
    S1.center = (-move_x-a, move_y)                                      # 创建一个矩形几何形状后，将其移动到新的位置
    S2 = pg.rectangle((width, length), layer=layer)
    S2.center = (-move_x-a, move_y)
    D3 = pg.boolean(S1, S2, 'a+b', layer=layer)
    s_big = pg.rectangle((100, 100), layer=layer)
    D3 << s_big.move((-250-move_x-a,-50+move_y))
    s_mid1 = pg.rectangle((20, 20), layer=layer)
    D3 << s_mid1.move((-310-move_x-a,-10+move_y))
    s_mid2 = pg.rectangle((20, 20), layer=layer)
    D3 << s_mid2.move((-210-move_x-a,90+move_y))
    s_small = pg.rectangle((10, 10), layer=layer)
    D3 << s_small.move((-305-move_x-a,95+move_y))
    D << D3
    D.move((-(D.xmax + D.xmin) / 2, -(D.ymax + D.ymin) / 2))
    D.name = 'mark'
    return D
    




# T = circular_poling_finger(R_ring=89, length=20,circle_pad_width=10, square_pad_width=100,square_pad_rotation = -90, mode_number=46, duty_cycle=0.3, layer=186)
#
# set_quickplot_options(show_ports=True, show_subports=True, label_aliases=True, new_window=True, blocking=True,
#                       zoom_factor=True, interactive_zoom=True)
# qp(T)