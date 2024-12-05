# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:27:05 2023

@author: SH
modified by jy
"""

import os

from phidl import Device, Layer, make_device, Path, CrossSection
from phidl import quickplot as qp
from numpy import sin,cos
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


#%%############################################################################
###############################################################################
# Basics, strictly no edition

def _euler_curve(R0=1, angle1=10, angle2=90):
    angle1 = angle1/180*pi
    angle2 = angle2/180*pi
    s = np.linspace(R0*np.sqrt(2*angle1), R0*np.sqrt(2*angle2), 600)
    S, C = sc.fresnel(s/np.sqrt(pi)/R0)
    y = S*np.sqrt(pi)*R0
    x = C*np.sqrt(pi)*R0
    return x, y


def euler_bend(width1=1, width2=2, R0=50, angle1=0, angle2=90, anticlockwise=True, layer=1):
    '''euler bend
    '''
    if anticlockwise == True:
        x, y = _euler_curve(R0, angle1, angle2)
        dx = 1e-9
        x = np.append(x, x[-1]+dx*np.cos(angle2/180*pi))
        y = np.append(y, y[-1]+dx*np.sin(angle2/180*pi))
        P = Path(np.array((x, y)).T)
        P1 = P.extrude(width=[width1, width2])
        P1.flatten(single_layer=layer)
        port1 = P1.add_port(1, (x[0], y[0]), width1, angle1+180)
        port2 = P1.add_port(2, (x[-1], y[-1]), width2, angle2)
        print(abs(y[0]-y[-1]))
        # port3 = P1.add_port(3, (x[0], y[0]), width1, angle1+270)
        # port4 = P1.add_port(4, (x[-1], y[-1]), width2, angle2+270)
        # qp(P1)
        return P1
    if anticlockwise == False:
        x, y = _euler_curve(R0, angle1, angle2)
        dx = 1e-9
        x = np.append(x, x[-1]+dx*np.cos(angle2/180*pi))
        y = -np.append(y, y[-1]+dx*np.sin(angle2/180*pi))
        P = Path(np.array((x, y)).T)
        P1 = P.extrude(width=[width1, width2])
        P1.flatten(single_layer=layer)
        port1 = P1.add_port(1, (x[0], y[0]), width1, angle1+180)
        port2 = P1.add_port(2, (x[-1], y[-1]), width2, angle2+180)
        # port3 = P1.add_port(3, (x[0], y[0]), width1, angle1+90)
        # port4 = P1.add_port(4, (x[-1], y[-1]), width2, angle2+180)
        return P1


def euler_bend_gap(width_ring=1, width_bus=2, R0=50, angle1=0, angle2=90, anticlockwise=True, gap1=2.8, gap2=0.8, p=1, layer=1):
    if anticlockwise == True:
        x, y = _euler_curve(R0, angle1, angle2)
        dx = 1e-9
        x = np.append(x, x[-1]+dx*np.cos(angle2/180*pi))
        y = np.append(y, y[-1]+dx*np.sin(angle2/180*pi))

        gap = np.linspace(gap1, gap2, len(x))
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
            x2[round((1-p)*len(x2))+1] - x2[round((1-p)*len(x2))]))/pi*180

        P = Path(np.array((x2[round((1-p)*len(x2)):round(len(x2))],
                 y2[round((1-p)*len(x2)):round(len(x2))])).T)
        P1 = P.extrude(width=width_bus)
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


def wg(width1=1, width2=2, length=10.0, layer=1):
    '''straight waveguide
    w: width, l: length
    '''
    P = Path()
    P.append(pp.straight(length=length))          # Straight section
    P1 = P.extrude(width=[width1, width2])
    P1.flatten(single_layer=layer)
    port1 = P1.add_port(1, (0, 0), width1, 180)
    port2 = P1.add_port(2, (length, 0), width2, 0)
    return P1


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
def arc_zy(width1=1, width2=2, angle1=0, angle2=90, R0=4, layer=1):
    '''arc waveguide
    R0: radius
    ZY version only support same width
    Same lah no different than the arc function deif
    '''
    P1 = pg.arc(radius=R0,width=width1,theta=angle2-angle1,start_angle=angle1,angle_resolution=0.1,layer=layer)
    P1.rotate(-angle1).movex(-R0)
    P1.rotate(-90+angle1)
    return P1

def euler_bend_180(w_wg=1, R0=200, stright_length = 100, out_length = 10,layer = 1):
    D = Device()
    
    bend_u = D<< euler_bend(width1=w_wg, width2=w_wg, R0=R0, angle1=0, angle2=90, anticlockwise=False, layer=layer).mirror()
    bend_d = D<< euler_bend(width1=w_wg, width2=w_wg, R0=R0, angle1=0, angle2=90, anticlockwise=False, layer=layer)
    WG1 = D <<wg(width1=w_wg, width2=w_wg, length=stright_length, layer=layer)
    WG_u = D <<wg(width1=w_wg, width2=w_wg, length=out_length, layer=layer)
    WG_d = D <<wg(width1=w_wg, width2=w_wg, length=out_length, layer=layer)
    # wg2 = D << wg(width1=w, width2=w, length=l1, layer=layer)
    bend_u.connect(port=2, destination=WG1.ports[1])
    bend_d.connect(port=2, destination=WG1.ports[2])
    WG_u.connect(port=1, destination=bend_u.ports[1])
    WG_d.connect(port=1, destination=bend_d.ports[1])
    
    port1 = D.add_port(name=1, port=WG_u.ports[2])
    port2 = D.add_port(name=2, port=WG_d.ports[2])
    
    return D


def bend_180(w_wg=1, R0=200, stright_length = 100, out_length = 10,layer = 1):
    D = Device()
    
    bend_u = D<< arc(width1=w_wg, width2=w_wg, angle1=0, angle2=90, R0=R0, layer=layer, orientation = 0).mirror()
    bend_d = D<< arc(width1=w_wg, width2=w_wg, angle1=0, angle2=90, R0=R0, layer=layer, orientation = 0)
    WG1 = D <<wg(width1=w_wg, width2=w_wg, length=stright_length, layer=layer)
    WG_u = D <<wg(width1=w_wg, width2=w_wg, length=out_length, layer=layer)
    WG_d = D <<wg(width1=w_wg, width2=w_wg, length=out_length, layer=layer)
    # wg2 = D << wg(width1=w, width2=w, length=l1, layer=layer)
    bend_u.connect(port=2, destination=WG1.ports[1])
    bend_d.connect(port=2, destination=WG1.ports[2])
    WG_u.connect(port=1, destination=bend_u.ports[1])
    WG_d.connect(port=1, destination=bend_d.ports[1])
    
    port1 = D.add_port(name=1, port=WG_u.ports[2])
    port2 = D.add_port(name=2, port=WG_d.ports[2])
    
    return D

def bend_180_wg_distance(w_wg=1, R0=200, wg_distance = 127*3, out_length = 10,layer = 1):
    D = Device()
    stright_length = wg_distance-2*R0
    if stright_length>0:
        bend_u = D<< arc(width1=w_wg, width2=w_wg, angle1=0, angle2=90, R0=R0, layer=layer, orientation = 0).mirror()
        bend_d = D<< arc(width1=w_wg, width2=w_wg, angle1=0, angle2=90, R0=R0, layer=layer, orientation = 0)
        WG1 = D <<wg(width1=w_wg, width2=w_wg, length=stright_length, layer=layer)
        WG_u = D <<wg(width1=w_wg, width2=w_wg, length=out_length, layer=layer)
        WG_d = D <<wg(width1=w_wg, width2=w_wg, length=out_length, layer=layer)
        # wg2 = D << wg(width1=w, width2=w, length=l1, layer=layer)
        bend_u.connect(port=2, destination=WG1.ports[1])
        bend_d.connect(port=2, destination=WG1.ports[2])
        WG_u.connect(port=1, destination=bend_u.ports[1])
        WG_d.connect(port=1, destination=bend_d.ports[1])
        
        port1 = D.add_port(name=1, port=WG_u.ports[2])
        port2 = D.add_port(name=2, port=WG_d.ports[2])
        
    else:
        print("2*R0 should < wg_distance")
    
    return D


def alignment_mark(width=5.0, length=100.0, center_square=False,
                   center_size=2.0, min_dimension=1.0,
                   bridge_length=3.0, layer=1):
    """generate a single cross alignment mark for ebeam lithography"""
    if center_square == False:
        S1 = pg.rectangle((length, width), layer=layer)
        S1.center = (0, 0)
        S2 = pg.rectangle((width, length), layer=layer)
        S2.center = (0, 0)
        D = pg.boolean(S1, S2, 'a+b', layer=layer)
        D.name = 'mark'
        return D
    if center_square == True:
        # add a small center square as a test of lithography resolution
        D = Device('mark')
        D.add_ref(pg.compass((center_size, center_size)))
        BRIDGE = Device()
        BRIDGE.add_polygon([[-min_dimension/2, -width/2, -width/2, width/2, width/2, min_dimension/2],
                            [0, bridge_length, length/2-center_size, length/2-center_size, bridge_length, 0]])
        D.add_ref(BRIDGE).move((0, center_size/2))
        D.add_ref(BRIDGE).mirror((0, 0), (1, 0)).move((0, -center_size/2))
        D.add_ref(BRIDGE).rotate(90).move((-center_size/2, 0))
        D.add_ref(BRIDGE).rotate(-90).move((center_size/2, 0))
        D.flatten(single_layer=layer)
        return D


def racetrack_euler(w1=2, w2=0.8, l=461, R=150, layer=6):
    # racetrack formed by euler bends and straignt wg
    # w1 = racetrack width, w2 = width at the euler bend joint (far end of coupling)
    # l = length of straigh arms
    # R = radius of the euler bend

    D = Device()
    # racetrack
    WG1 = wg(width1=w1, width2=w1, length=l, layer=layer)
    WG2 = wg(width1=w1, width2=w1, length=l, layer=layer)
    BEND1 = euler_bend(width1=w1, width2=w1, R0=R, angle1=0,
                       angle2=90, anticlockwise=True, layer=layer)
    BEND2 = euler_bend(width1=w1, width2=w1, R0=R, angle1=0,
                       angle2=90, anticlockwise=False, layer=layer)
    BEND3 = euler_bend(width1=w1, width2=w2, R0=R, angle1=0,
                       angle2=90, anticlockwise=True, layer=layer)
    BEND4 = euler_bend(width1=w1, width2=w2, R0=R, angle1=0,
                       angle2=90, anticlockwise=False, layer=layer)

    bend1 = D << BEND1
    bend2 = D << BEND2
    wg1 = D << WG1
    bend3 = D << BEND3
    bend4 = D << BEND4
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


def directional_coupler(w=1, l=5, R=5, wg_distance=5, gap=5, l_couple=10, layer=1):
    '''directional coupler/ beam splitter with controllable waveguide distance
    '''
    dy = (wg_distance - gap - w) / 4
    # Calculate the angle from dy and R
    theta = np.arccos(1 - dy / R)
    angle =  np.degrees(theta)

    D = Device()
    wg2 = D << wg(width1=w, width2=w, length=l, layer=layer)
    wg1 = D << wg(width1=w, width2=w, length=l, layer=layer)
    wg_couple = D << wg(width1=w, width2=w, length=l_couple, layer=layer)
    
    arc1a = D << arc(width1=w, width2=w, angle1=0, angle2=angle, R0=R, layer=layer)
    arc1b = D << arc(width1=w, width2=w, angle1=0, angle2=angle, R0=R, layer=layer)
    arc2a = D << arc(width1=w, width2=w, angle1=0, angle2=angle, R0=R, layer=layer)
    
    # arc1a.mirror((-1, 0), (1, 0))
    # arc1b.mirror((-1, 0), (1, 0))
    
    arc2b = D << arc(width1=w, width2=w, angle1=0, angle2=angle, R0=R, layer=layer)

    arc1a.connect(port=1, destination=wg_couple.ports[2])
    arc1b.connect(port=2, destination=arc1a.ports[2])
    arc2a.connect(port=2, destination=wg_couple.ports[1])
    arc2b.connect(port=1, destination=arc2a.ports[1])
    wg1.connect(port=1, destination=arc1b.ports[1])
    wg2.connect(port=1, destination=arc2b.ports[2])
    
    port_left = D.add_port(1, wg1.ports[2].midpoint, w, 0)
    port_right = D.add_port(2, wg2.ports[2].midpoint, w, 180)
    
    P = Device()
    d = gap + w
    upper = (P << D).movey(d / 2)
    lower = (P << D).mirror((-1, 0), (1, 0)).movey(-d / 2)
    
    P.add_port(name=3, port=upper.ports[1])
    P.add_port(name=1, port=upper.ports[2])
    P.add_port(name=4, port=lower.ports[1])
    P.add_port(name=2, port=lower.ports[2])
    # qp(P)
    
    return P

def directional_coupler_angle(w=1, l=5, R=5, angle=30, gap=5, l_couple = 10, layer=1):
    '''directional coupler/ beam splitter
    '''
    D = Device()
    wg2 = D << wg(width1=w, width2=w, length=l, layer=layer)
    wg1 = D << wg(width1=w, width2=w, length=l, layer=layer)
    wg_couple = D << wg(width1=w, width2=w, length=l_couple, layer=layer)
    # wg3 = D << wg(width1=w, width2=w, length=l, layer=layer)
    # wg_couple1 = D << wg(width1=w, width2=w, length=l_couple, layer=layer)
    # wg_couple2 = D << wg(width1=w, width2=w, length=l_couple, layer=layer)
    
    arc1a = D << arc(width1=w, width2=w, angle1=0,
                      angle2=angle, R0=R, layer=layer)
    arc1b = D << arc(width1=w, width2=w, angle1=0,
                      angle2=angle, R0=R, layer=layer)
    arc2a = D << arc(width1=w, width2=w, angle1=0,
                      angle2=angle, R0=R, layer=layer)
    
    # arc1a.mirror((-1,0),(1,0))
    # arc1b.mirror((-1,0),(1,0))
    
    arc2b = D << arc(width1=w, width2=w, angle1=0,
                      angle2=angle, R0=R, layer=layer)

    arc1a.connect(port=1, destination=wg_couple.ports[2])
    arc1b.connect(port=2, destination=arc1a.ports[2])
    arc2a.connect(port=2, destination=wg_couple.ports[1])
    
    arc2b.connect(port=1, destination=arc2a.ports[1])
    wg1.connect(port=1, destination=arc1b.ports[1])
    wg2.connect(port=1, destination=arc2b.ports[2])
    port_left = D.add_port(1, wg1.ports[2].midpoint, w, 0)
    port_right = D.add_port(2, wg2.ports[2].midpoint, w, 180)
    P = Device()
    d = gap+w
    upper = (P<<D).movey(d/2)
    lower = (P<<D).mirror((-1,0),(1,0)).movey(-d/2)
    P.add_port(name=3, port=upper.ports[1])
    P.add_port(name=1, port=upper.ports[2])
    P.add_port(name=4, port=lower.ports[1])
    P.add_port(name=2, port=lower.ports[2])
    
    return P

# def directional_coupler_euler(w=1, l1=5, l2=5, R=5, angle=30, gap=5, l_couple = 10, layer=1):
#     '''directional coupler/ beam splitter
#     '''
#     D = Device()
#     wg2 = D << wg(width1=w, width2=w, length=l1, layer=layer)
#     wg1 = D << wg(width1=w, width2=w, length=l2, layer=layer)
#     wg_couple = D << wg(width1=w, width2=w, length=l_couple, layer=layer)
#     # wg3 = D << wg(width1=w, width2=w, length=l, layer=layer)
#     # wg_couple1 = D << wg(width1=w, width2=w, length=l_couple, layer=layer)
#     # wg_couple2 = D << wg(width1=w, width2=w, length=l_couple, layer=layer)
    
#     arc1a = D << euler_bend(width1=w, width2=w, R0=R, angle1=0,
#                         angle2=angle, anticlockwise=True, layer=layer)
#     arc1b = D << euler_bend(width1=w, width2=w, R0=R, angle1=0,
#                         angle2=angle, anticlockwise=True, layer=layer)
#     arc2a = D << euler_bend(width1=w, width2=w, R0=R, angle1=0,
#                         angle2=angle, anticlockwise=True, layer=layer)
#     arc1a.mirror((-1,0),(1,0))
#     arc1b.mirror((-1,0),(1,0))
#     arc2b = D << euler_bend(width1=w, width2=w, R0=R, angle1=0,
#                         angle2=angle, anticlockwise=True, layer=layer)

#     arc1a.connect(port=2, destination=wg_couple.ports[2])
#     arc1b.connect(port=1, destination=arc1a.ports[1])
#     arc2a.connect(port=2, destination=wg_couple.ports[1])
#     arc2b.connect(port=1, destination=arc2a.ports[1])
    
#     wg1.connect(port=1, destination=arc1b.ports[2])
#     wg2.connect(port=1, destination=arc2b.ports[2])
#     port_left = D.add_port(1, wg1.ports[2].midpoint, w, 0)
#     port_right = D.add_port(2, wg2.ports[2].midpoint, w, 180)
    
#     P = Device()
#     d = gap+w
#     upper = (P<<D).movey(d/2)
#     lower = (P<<D).mirror((-1,0),(1,0)).movey(-d/2)
#     P.add_port(name=4, port=upper.ports[1])
#     P.add_port(name=1, port=upper.ports[2])
#     P.add_port(name=3, port=lower.ports[1])
#     P.add_port(name=2, port=lower.ports[2])
    
#     return P


def directional_coupler_euler(w=1, l1=5, l2=5, R=5, wg_distance=10, gap=5, l_couple=10, layer=1):
    '''directional coupler/ beam splitter with controllable waveguide distance'''
    # Calculate the required y-coordinate difference for a single S bend
    dy_target = (wg_distance - gap - w) / 8
    print(dy_target)
    # Function to find the angle for a given dy
    def find_angle_for_dy(R, dy_target):
        angle = 30  # Initial guess for the angle
        for _ in range(100):  # Limit the number of iterations
            _, dy = _euler_curve(R, 0, angle)
            dy = abs(dy[0] - dy[-1])
            if abs(dy - dy_target) < 1e-9:
                break
            angle *= dy_target / dy  # Adjust the angle proportionally
        return angle

    angle = find_angle_for_dy(R, dy_target)
    
    # angle = 10
    D = Device()
    wg2 = D << wg(width1=w, width2=w, length=l1, layer=layer)
    wg1 = D << wg(width1=w, width2=w, length=l2, layer=layer)
    wg_couple = D << wg(width1=w, width2=w, length=l_couple, layer=layer)
    
    arc1a = D << euler_bend(width1=w, width2=w, R0=R, angle1=0, angle2=angle, anticlockwise=True, layer=layer)
    arc1b = D << euler_bend(width1=w, width2=w, R0=R, angle1=0, angle2=angle, anticlockwise=True, layer=layer)
    arc2a = D << euler_bend(width1=w, width2=w, R0=R, angle1=0, angle2=angle, anticlockwise=True, layer=layer)
    arc1a.mirror((-1,0),(1,0))
    arc1b.mirror((-1,0),(1,0))
    arc2b = D << euler_bend(width1=w, width2=w, R0=R, angle1=0, angle2=angle, anticlockwise=True, layer=layer)

    arc1a.connect(port=2, destination=wg_couple.ports[2])
    arc1b.connect(port=1, destination=arc1a.ports[1])
    arc2a.connect(port=2, destination=wg_couple.ports[1])
    arc2b.connect(port=1, destination=arc2a.ports[1])
    
    wg1.connect(port=1, destination=arc1b.ports[2])
    wg2.connect(port=1, destination=arc2b.ports[2])
    port_left = D.add_port(1, wg1.ports[2].midpoint, w, 0)
    port_right = D.add_port(2, wg2.ports[2].midpoint, w, 180)
    # qp(D)
    P = Device()
    d = gap+w
    upper = (P<<D).movey(d/2)
    lower = (P<<D).mirror((-1,0),(1,0)).movey(-d/2)
    P.add_port(name=3, port=upper.ports[1])
    P.add_port(name=1, port=upper.ports[2])
    P.add_port(name=4, port=lower.ports[1])
    P.add_port(name=2, port=lower.ports[2])
    
    return P

# def directional_coupler_euler(w=1, l1=5, l2=5, R=5, wg_distance=10, gap=5, l_couple=10, layer=1):
#     '''directional coupler/ beam splitter with controllable waveguide distance'''
    
#     # Initial guess for the angle
#     angle = 30
    
#     # Create the initial structure and calculate dy_final
#     def create_structure(angle):
#         D = Device()
#         wg2 = D << wg(width1=w, width2=w, length=l1, layer=layer)
#         wg1 = D << wg(width1=w, width2=w, length=l2, layer=layer)
#         wg_couple = D << wg(width1=w, width2=w, length=l_couple, layer=layer)
        
#         arc1a = D << euler_bend(width1=w, width2=w, R0=R, angle1=0, angle2=angle, anticlockwise=True, layer=layer)
#         arc1b = D <<euler_bend(width1=w, width2=w, R0=R, angle1=0, angle2=angle, anticlockwise=True, layer=layer)
#         arc2a = D <<euler_bend(width1=w, width2=w, R0=R, angle1=0, angle2=angle, anticlockwise=True, layer=layer)
        
#         arc1a.mirror((-1, 0), (1, 0))
#         arc1b.mirror((-1, 0), (1, 0))
        
#         arc2b = D <<euler_bend(width1=w, width2=w, R0=R, angle1=0, angle2=angle, anticlockwise=True, layer=layer)

#         arc1a.connect(port=2, destination=wg_couple.ports[2])
#         arc1b.connect(port=1, destination=arc1a.ports[1])
#         arc2a.connect(port=2, destination=wg_couple.ports[1])
#         arc2b.connect(port=1, destination=arc2a.ports[1])
        
#         wg1.connect(port=1, destination=arc1b.ports[2])
#         wg2.connect(port=1, destination=arc2b.ports[2])
        
#         port_left = D.add_port(1, wg1.ports[2].midpoint, w, 0)
#         port_right = D.add_port(2, wg2.ports[2].midpoint, w, 180)
        
#         P = Device()
#         d = gap + w
#         upper = (P << D).movey(d / 2)
#         lower = (P << D).mirror((-1, 0), (1, 0)).movey(-d / 2)
        
#         P.add_port(name=3, port=upper.ports[1])
#         P.add_port(name=1, port=upper.ports[2])
#         P.add_port(name=4, port=lower.ports[1])
#         P.add_port(name=2, port=lower.ports[2])
        
#         return P

#     # Iteratively adjust the angle
#     iteration = 0
#     dy_final = 0
#     while abs(dy_final - wg_distance) > 1e-3 and iteration < 100:
#         P = create_structure(angle)
#         dy_final = abs(P.ports[1].midpoint[1] - P.ports[3].midpoint[1])
        
#         if abs(dy_final - wg_distance) < 1e-3:
#             break
#         angle *= wg_distance / dy_final  # Adjust the angle proportionally
#         iteration += 1
    
#     return P


def grating(period,
            number_of_teeth,
            fill_frac,
            width,
            lda=1,
            sin_theta=0,
            focus_distance=-1,
            focus_width=-1,
            evaluations=50,
            layer=1
            ):
    '''
    Straight or focusing grating.

    period          : grating period
    number_of_teeth : number of teeth in the grating
    fill_frac       : filling fraction of the teeth with respect to the period
    width           : width of the grating
    direction       : one of {'+x', '-x', '+y', '-y'}
    lda             : free-space wavelength
    sin_theta       : sine of incidence angle
    focus_distance  : focus distance (negative for straight grating)
    focus_width     : if non-negative, the focusing area is included in the
                      result (usually for negative resists) and this is the
                      width of the waveguide connecting to the grating
    evaluations     : number of parametric evaluations of `path.parametric`

    Return `PolygonSet`
    '''
    position = (0, 0)
    datatype = 0
    if focus_distance < 0:
        print('hi!')
        path = gdspy.L1Path(
            (position[0] - 0.5 * width, position[1] + 0.5 *
             (number_of_teeth - 1 + fill_frac) * period),
            '+x',
            period * fill_frac, [width], [],
            number_of_teeth,
            period,
            layer=layer,
            datatype=datatype)
    else:
        neff = lda / float(period) + sin_theta
        qmin = int(focus_distance / float(period) + 0.5)
        path = gdspy.Path(period * fill_frac, position)
        max_points = 199 if focus_width < 0 else 2 * evaluations
        c3 = neff**2 - sin_theta**2
        w = 0.5 * width
        for q in range(qmin, qmin + number_of_teeth):
            c1 = q * lda * sin_theta
            c2 = (q * lda)**2
            path.parametric(
                lambda t: (width * t - w, (c1 + neff * np.sqrt(
                    c2 - c3 * (width * t - w)**2)) / c3),
                number_of_evaluations=evaluations,
                max_points=max_points,
                layer=layer,
                datatype=datatype)
            path.x = position[0]
            path.y = position[1]
        if focus_width >= 0:
            path.polygons[0] = np.vstack(
                (path.polygons[0][:evaluations, :],
                 ([position] if focus_width == 0 else
                  [(position[0] + 0.5 * focus_width, position[1]),
                   (position[0] - 0.5 * focus_width, position[1])])))
            path.fracture()
    D = Device()
    [D.add_polygon(p, layer=layer) for p in path.polygons]
    D.flatten(single_layer=layer)
    D.add_port(1, (0, D.ymin), focus_width, -90)

    return D


def sbend(length=50.0, height=30.0, width=1.0, layer=0):
    '''s-bend. 
    '''
    t = np.linspace(0, 1, 300)
    x = t*length
    y = height/2*(1-np.cos(t*np.pi))
    D = pr.point_path(points=np.array((x, y)).T, width=width, layer=layer)
    D.flatten(single_layer=layer)
    D.ports[1].orientation = 180
    D.ports[2].orientation = 0
    return D


def sbend_taper(w1=10.0, w2=30.0, l=200, layer=1):
    '''s-bend taper. 
    '''
    D = Device()
    t = np.linspace(0, 1, 300)
    x = t*l
    y = (w2-w1)/4*(1-np.cos(t*np.pi))

    xpts = np.concatenate([x, np.flip(x)])
    ypts = np.concatenate([y+w1/2, -np.flip(y)-w1/2])
    xylist = np.column_stack((xpts, ypts))
    D.add_polygon(xylist, layer=1)
    D.flatten(single_layer=layer)
    D.add_port(1, (D.xmin, 0), w1, 180)
    D.add_port(2, (D.xmax, 0), w2, 0)
    return D


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


#%%############################################################################
###############################################################################
# components = combination of basics, not edition


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
                      justify='left', layer=layer_text))
        t.move((xmin, ymin-t.ysize))

    D.flatten(single_layer=layer)

    return D


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


def electrode_straight(lead_width=20, lead_length=100, pad_size1=50, pad_size2=100, gap=10, layer=6):

    D = Device()

    WG1 = wg(width1=lead_width, width2=lead_width,
             length=lead_length, layer=layer)
    WG2 = wg(width1=pad_size2, width2=pad_size2, length=pad_size1, layer=layer)
    WG3 = wg(width1=lead_width, width2=lead_width,
             length=lead_length, layer=layer)
    WG4 = wg(width1=pad_size2, width2=pad_size2, length=pad_size1, layer=layer)

    wg1 = D << WG1
    wg2 = D << WG2
    wg3 = D << WG3
    wg4 = D << WG4

    wg1.move((-lead_length/2, -lead_width/2-gap/2))
    wg2.move((-pad_size1/2, -lead_width/2-gap/2-lead_width/2-pad_size2/2))
    wg3.move((-lead_length/2, +lead_width/2+gap/2))
    wg4.move((-pad_size1/2, lead_width/2+gap/2+lead_width/2+pad_size2/2))

    return D


def racetrack_euler_coupling_pulley(w1=2, w2=0.8, w_bus=0.6, l=461, R=150, g1=2.8, g2=0.8, p=0.5, layer=6):
    # racetrack formed by euler bends and straignt wg
    # w1 = racetrack width, w2 = width at the euler bend joint (far end of coupling)
    # w_bus = bus wg width
    # l = length of straigh arms
    # R = radius of the euler bend
    # p = extend of euler bend. p=1 is 90 deg bend, p=0 is 0 deg bend
    # g1 = gap at p=1, if p<1, the largest gap at coupling is reduced correspondingly
    # g2 = gap at 0 deg angle, normally the narrowest gap

    D = Device()
    # racetrack
    WG1 = wg(width1=w1, width2=w1, length=l, layer=layer)
    WG2 = wg(width1=w1, width2=w1, length=l, layer=layer)
    BEND1 = euler_bend(width1=w1, width2=w1, R0=R, angle1=0,
                       angle2=90, anticlockwise=True, layer=layer)
    BEND2 = euler_bend(width1=w1, width2=w1, R0=R, angle1=0,
                       angle2=90, anticlockwise=False, layer=layer)
    BEND3 = euler_bend(width1=w1, width2=w2, R0=R, angle1=0,
                       angle2=90, anticlockwise=True, layer=layer)
    BEND4 = euler_bend(width1=w1, width2=w2, R0=R, angle1=0,
                       angle2=90, anticlockwise=False, layer=layer)

    # gap around euler bend
    BUS_GAP1 = euler_bend_gap(width_ring=w1, width_bus=w_bus, R0=R, angle1=0,
                              angle2=90, anticlockwise=True, gap1=g1, gap2=g2, p=p, layer=layer)
    BUS_GAP2 = euler_bend_gap(width_ring=w1, width_bus=w_bus, R0=R, angle1=0,
                              angle2=90, anticlockwise=False, gap1=g1, gap2=g2, p=p, layer=layer)

    # bus waveguide
    BUS_ARC1 = arc(width1=w_bus, width2=w_bus, angle1=0,
                   angle2=BUS_GAP1[1], R0=100, layer=layer)
    BUS_ARC2 = arc(width1=w_bus, width2=w_bus, angle1=0,
                   angle2=BUS_GAP1[1], R0=100, layer=layer)
    BUS_WG1 = wg(width1=w_bus, width2=w_bus, length=R, layer=layer)
    BUS_WG2 = wg(width1=w_bus, width2=w_bus, length=R, layer=layer)

    bend1 = D << BEND1
    bend2 = D << BEND2
    wg1 = D << WG1
    bend3 = D << BEND3
    bend4 = D << BEND4
    wg2 = D << WG2

    bus_gap1 = D << BUS_GAP1[0]
    bus_gap2 = D << BUS_GAP2[0]
    bus_arc1 = D << BUS_ARC1
    bus_arc2 = D << BUS_ARC2
    bus_wg1 = D << BUS_WG1
    bus_wg2 = D << BUS_WG2

    bend1.rotate(0)
    bend1.movex(l/2)
    bend1.movey(0)
    bend2.connect(port=2, destination=bend1.ports[2])
    wg1.connect(port=2, destination=bend2.ports[1], overlap=0)
    bend3.connect(port=1, destination=wg1.ports[1])
    bend4.connect(port=2, destination=bend3.ports[2])
    wg2.connect(port=1, destination=bend4.ports[1])

    bus_gap1.connect(
        port=4, destination=bend1.ports[4], overlap=-g2 - w_bus/2 - w1/2)
    bus_gap2.connect(port=2, destination=bus_gap1.ports[2])
    bus_arc1.connect(port=1, destination=bus_gap1.ports[1])
    bus_arc2.connect(port=2, destination=bus_gap2.ports[1])
    bus_wg1.connect(port=1, destination=bus_arc1.ports[2])
    bus_wg2.connect(port=1, destination=bus_arc2.ports[1])

    port1 = D.add_port(1, bus_wg1.ports[2].midpoint, w_bus, 270)
    port2 = D.add_port(2, bus_wg2.ports[2].midpoint, w_bus, 90)

    return D
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


def ring_coupling_pulley(w_ring=2, w_bus=0.6, R=150, g=1, angle=10, layer=6):
    D = Device()
    l_wg = np.max((R*(1-2*sin(angle/180*pi)),5))
    RING = pg.ring(radius=R, width=w_ring, angle_resolution=0.1, layer=layer)
    ARC_WHOLE = arc(width1=w_bus, width2=w_bus, angle1=0,
               angle2=2*angle, R0=R+w_ring/2+w_bus/2+g, layer=layer).rotate(-angle,center=(0,R+w_ring/2+w_bus/2+g))
    ARC1A = arc(width1=w_bus, width2=w_bus, angle1=0,
                angle2=angle, R0=R+w_ring/2+w_bus/2+g, layer=layer)
    WG1 = wg(width1=w_bus, width2=w_bus, length=l_wg, layer=layer)
    ARC2A = arc(width1=w_bus, width2=w_bus, angle1=0,
                angle2=angle, R0=R+w_ring/2+w_bus/2+g, layer=layer)
    WG2 = wg(width1=w_bus, width2=w_bus, length=l_wg, layer=layer)

    ring = D << RING
    arc_whole = D<< ARC_WHOLE
    arc1a = D << ARC1A
    wg1 = D << WG1
    arc2a = D << ARC2A
    wg2 = D << WG2

    arc_whole.movey(-R-g-w_ring/2-w_bus/2)
    arc1a.connect(port=2, destination=arc_whole.ports[2])
    wg1.connect(port=1, destination=arc1a.ports[1])

    arc2a.connect(port=1, destination=arc_whole.ports[1])
    wg2.connect(port=1, destination=arc2a.ports[2])

    port1 = D.add_port(1, wg1.ports[2].midpoint, w_bus, 0)
    port2 = D.add_port(2, wg2.ports[2].midpoint, w_bus, 180)

    return D
def ring_coupling_pulley_taper(w_ring=2, w_bus=0.7,w_thin=0.3, R=150, g=1, angle=10, layer=6):
    #this version I shorten the length of the straight waveguide
    D = Device()
    l_wg = np.max((R * (1 - 2*sin(angle / 180 * pi)), 5))
    RING = pg.ring(radius=R, width=w_ring, angle_resolution=0.1, layer=layer)
    ARC1 = arc(width1=w_thin, width2=w_bus, angle1=0,
               angle2=angle, R0=R + w_ring / 2 + w_bus / 2 + g, layer=layer)
    ARC1A = arc(width1=w_bus, width2=w_bus, angle1=0,
                angle2=angle, R0=R + w_ring / 2 + w_bus / 2 + g, layer=layer)
    WG1 = wg(width1=w_bus, width2=w_bus, length=l_wg, layer=layer)
    ARC2 = arc(width1=w_bus, width2=w_thin, angle1=0,
               angle2=angle, R0=R + w_ring / 2 + w_bus / 2 + g, layer=layer)
    ARC2A = arc(width1=w_bus, width2=w_bus, angle1=0,
                angle2=angle, R0=R + w_ring / 2 + w_bus / 2 + g, layer=layer)
    WG2 = wg(width1=w_bus, width2=w_bus, length=l_wg, layer=layer)

    ring = D << RING
    arc1 = D << ARC1
    arc1a = D << ARC1A
    wg1 = D << WG1
    arc2 = D << ARC2
    arc2a = D << ARC2A
    wg2 = D << WG2

    arc1.movey(-R - g - w_ring / 2 - w_bus / 2)
    arc1a.connect(port=2, destination=arc1.ports[2])
    wg1.connect(port=1, destination=arc1a.ports[1])

    arc2.connect(port=2, destination=arc1.ports[1])
    arc2a.connect(port=1, destination=arc2.ports[1])
    wg2.connect(port=1, destination=arc2a.ports[2])

    port1 = D.add_port(1, wg1.ports[2].midpoint, w_bus, 0)
    port2 = D.add_port(2, wg2.ports[2].midpoint, w_bus, 180)

    return D
def ring_coupling_symmetric(w_ring=2, w_bus=0.8, R=150, g=1, layer=6):
    # is this the tapered pulley coupler?
    D = Device()
    RING = pg.ring(radius=R, width=w_ring, angle_resolution=0.1, layer=layer)
    ARC1 = arc(width1=w_bus, width2=w_bus, angle1=0,
               angle2=15, R0=R+w_ring/2+w_bus/2+g, layer=layer)
    ARC1A = arc(width1=w_bus, width2=w_bus, angle1=0,
                angle2=15, R0=R+w_ring/2+w_bus/2+g, layer=layer)
    WG1 = wg(width1=w_bus, width2=w_bus, length=R, layer=layer)
    ARC2 = arc(width1=w_bus, width2=w_bus, angle1=0,
               angle2=15, R0=R+w_ring/2+w_bus/2+g, layer=layer)
    ARC2A = arc(width1=w_bus, width2=w_bus, angle1=0,
                angle2=15, R0=R+w_ring/2+w_bus/2+g, layer=layer)
    WG2 = wg(width1=w_bus, width2=w_bus, length=R, layer=layer)

    ring = D << RING
    arc1 = D << ARC1
    arc1a = D << ARC1A
    wg1 = D << WG1
    arc2 = D << ARC2
    arc2a = D << ARC2A
    wg2 = D << WG2

    arc1.rotate(180)
    arc1.movey(-R-g-w_ring/2-w_bus/2)
    arc1a.connect(port=2, destination=arc1.ports[2])
    wg1.connect(port=1, destination=arc1a.ports[1])

    arc2.connect(port=2, destination=arc1.ports[1])
    arc2a.connect(port=1, destination=arc2.ports[1])
    wg2.connect(port=1, destination=arc2a.ports[2])

    port1 = D.add_port(1, wg1.ports[2].midpoint, w_bus, 180)
    port2 = D.add_port(2, wg2.ports[2].midpoint, w_bus, 0)

    return D

def ring_coupling_point(width=1,w_bus=0.8, R_ring=100,R_couple=100, gap_s=0.3, gap_e=6, layer=6):
        D = Device()
        Ring = D<<pg.ring(radius=R_ring, width=width, angle_resolution=0.1, layer=layer)

        arc_gap2 = np.arccos(((R_couple + w_bus / 2) - (gap_e - gap_s) / 2) / (R_couple + w_bus / 2))
        degrees_arc_gap2 = np.degrees(arc_gap2)
        R_couple = R_couple +width/2+w_bus/2+gap_s
        arc_coupling = arc(width1=w_bus, width2=w_bus, angle1=-degrees_arc_gap2, angle2=degrees_arc_gap2, R0=R_couple,
                           layer=layer)
        arc_coupling.rotate(-degrees_arc_gap2, center=(0, R_couple))
        arc_coupling.move((0, w_bus / 2))
        arc_coupling.mirror((0, 0), (1, 0))
        arc_coupling.move((0, -R_ring - width / 2 - gap_s))
        a_coupling = D << arc_coupling

        arc_coupling_left = arc(width1=w_bus, width2=w_bus, angle1=-degrees_arc_gap2 / 2, angle2=degrees_arc_gap2 / 2,
                                R0=2 * R_couple, layer=layer)
        a_coupling_left = D << arc_coupling_left.mirror((0, 0), (1, 0))
        arc_coupling_right = arc(width1=w_bus, width2=w_bus, angle1=-degrees_arc_gap2 / 2, angle2=degrees_arc_gap2 / 2,
                                R0=2 * R_couple, layer=layer)
        a_coupling_right = D << arc_coupling_right.mirror((0, 0), (1, 0))

        arc_coupling_left1 = arc(width1=w_bus, width2=w_bus, angle1=-degrees_arc_gap2 / 2, angle2=degrees_arc_gap2 / 2,
                                R0=2 * R_couple, layer=layer)
        a_coupling_left1 = D << arc_coupling_left1
        arc_coupling_right1 = arc(width1=w_bus, width2=w_bus, angle1=-degrees_arc_gap2 / 2, angle2=degrees_arc_gap2 / 2,
                                R0=2 * R_couple, layer=layer)
        a_coupling_right1 = D << arc_coupling_right1

        arc_coupling_left2 = arc(width1=w_bus, width2=w_bus, angle1=0, angle2=degrees_arc_gap2, R0=2 * R_couple,
                                 layer=layer)
        a_coupling_left2 = D << arc_coupling_left2
        arc_coupling_right2 = arc(width1=w_bus, width2=w_bus, angle1=0, angle2=degrees_arc_gap2, R0=2 * R_couple,
                                 layer=layer)
        a_coupling_right2 = D << arc_coupling_right2

        # WG1 = wg(width1=width, width2=width, length=300, layer=layer)
        # wg1 = D << WG1
        # wg1.connect(port=2, destination=ring_resonator.ports[1])

        # euler_bend_left2,euler_len, euler_detax, euler_detay =  euler_bend(width1=width, width2=width, R0=3*R_ring, angle1=0, angle2=90-3*degrees_arc_gap2, num_seg=1000, anticlockwise=True, layer=layer)
        # E_coupling_left2 = D << euler_bend_left2
        # euler_bend_right2,euler_len, euler_detax, euler_detay =  euler_bend(width1=width, width2=width, R0=3*R_ring, angle1=0, angle2=90-3*degrees_arc_gap2, num_seg=1000, anticlockwise=False, layer=layer)
        # E_coupling_right2 = D << euler_bend_right2.mirror((0, 0), (1, 0))

        a_coupling_left.connect(port=2, destination=a_coupling.ports[1])
       # print(a_coupling.ports)

        a_coupling_right.connect(port=1, destination=a_coupling.ports[2])
        a_coupling_left1.connect(port=2, destination=a_coupling_left.ports[1])
        a_coupling_right1.connect(port=1, destination=a_coupling_right.ports[2])
        a_coupling_left2.connect(port=2, destination=a_coupling_left1.ports[1])
        a_coupling_right2.connect(port=1, destination=a_coupling_right1.ports[2])
        # a_coupling = D << arc_coupling.rotate(-degrees_arc_gap2)
        port1 = D.add_port(1, a_coupling_left2.ports[1].midpoint, width, 180)
        port2 = D.add_port(2, a_coupling_right2.ports[2].midpoint, width, 0)

        return D


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


def wg_poling(w1=2, w2=2, l=5000, poling_period=4.74, w_finger=4.74*0.4, l_finger=20, gap=15, lead_width=40, pad_size=150, finger_on_ground=0, finger_shape='pointy', text_on=1, overlap=0, layer=6):

    D = Device()
    POLING_FINGERS = generate_poling_finger_narrow_trace(period=poling_period, w_finger=w_finger, l_finger=l_finger, total_length=l, gap=gap, lead_width=lead_width, pad_size=pad_size,
                                                         finger_on_ground=finger_on_ground, finger_shape=finger_shape, text_on=text_on, overlap=overlap, layer_metal=layer_metal, layer_ebeam_metal=layer_ebeam_metal)

    POLING_WG1 = wg(width1=w1, width2=w1, length=l, layer=layer)

    poling_fingers = D << POLING_FINGERS
    poling_wg1 = D << POLING_WG1

    poling_fingers.move((-l/2, -gap/2))
    poling_wg1.movex(-l/2)

    port1 = D.add_port(1, poling_wg1.ports[1].midpoint, w1, 180)
    port2 = D.add_port(2, poling_wg1.ports[2].midpoint, w1, 0)

    return D
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


def wg_mod(w1=1, w2=2, lead_width=20, lead_length=100, pad_size1=50, pad_size2=100, gap=10, layer=6):
    D = Device()
    pad = D << electrode_straight(lead_width=lead_width, lead_length=lead_length,
                                  pad_size1=pad_size1, pad_size2=pad_size2, gap=gap, layer=layer_metal)
    wg1 = D << wg(width1=w1, width2=w1, length=lead_length, layer=layer)

    wg1.movex(-lead_length/2)

    port1 = D.add_port(1, wg1.ports[1].midpoint, w1, 180)
    port2 = D.add_port(2, wg1.ports[2].midpoint, w1, 0)

    return D


def cpw_with_pad(w_pad=60.0, g_pad=30.0, l_pad=100.0, w_cpw=30.0,
                 g_cpw=5.0, l_expander=300.0, l_mod=3000, w_metal=500.0,
                 layer=1):
    D = Device()
    D1 = Device()
    i1 = D1.add_ref(wg(w_pad, w_pad, l_pad))
    i2 = D1.add_ref(sbend_taper(w1=w_pad, w2=w_cpw, l=l_expander))
    i2.connect(1, i1.ports[2])

    D2 = Device()
    o1 = D2.add_ref(wg(w_pad+2*g_pad, w_pad+2*g_pad, l_pad))
    #o2 = D2.add_ref(pg.taper(l_expander, w_pad+2*g_pad, w_cpw+2*g_cpw, o1.ports[2]))
    o2 = D2.add_ref(sbend_taper(w1=w_pad+2*g_pad,
                    w2=w_cpw+2*g_cpw, l=l_expander))
    o2.connect(1, o1.ports[2])
    D3 = pg.boolean(D2, D1, 'a-b')

    RECT = pg.bbox([(D3.xmin, -w_metal/2), (D3.xmax, w_metal/2)])
    PAD = pg.boolean(RECT, D3, 'a-b')
    PAD.add_port(1, (PAD.xmin, 0), w_pad, 180)
    PAD.add_port(2, (PAD.xmax, 0), w_cpw, 0)
    pad1 = D.add_ref(PAD)
    c = D.add_ref(cpw(w_cpw, g_cpw, w_metal/2-w_cpw/2-g_cpw, l_mod, layer))
    c.connect(1, pad1.ports[2])
    D.add_ref(PAD).connect(2, c.ports[2])
    D.center = (0, 0)
    D.flatten(single_layer=layer)
    D.add_port(1, (D.xmin, 0), w_pad, 180)
    D.add_port(2, (D.xmax, 0), w_pad, 0)
    D.add_port('center', (0, 0), w_cpw, 0)
    return D


def electrode_straight_cpw(lead_width=20, lead_length=100, pad_size1=50, pad_size2=100, gap=10, layer=6):

    D = Device()

    WG1 = wg(width1=lead_width, width2=lead_width,
             length=lead_length, layer=layer)
    WG2 = wg(width1=pad_size2, width2=pad_size2, length=pad_size1, layer=layer)
    WG3 = wg(width1=lead_width, width2=lead_width,
             length=lead_length, layer=layer)
    WG4 = wg(width1=pad_size2, width2=pad_size2, length=pad_size1, layer=layer)
    WG5 = wg(width1=pad_size2, width2=pad_size2, length=pad_size1, layer=layer)

    wg1 = D << WG1
    wg2 = D << WG2
    wg3 = D << WG3
    wg4 = D << WG4
    wg5 = D << WG5

    wg1.move((-lead_length/2, -lead_width/2-gap/2))
    wg2.move((-pad_size1/2, -lead_width/2-gap/2-lead_width/2-pad_size2/2))
    wg3.move((-lead_length/2, +lead_width/2+gap/2))
    wg4.move((-pad_size1/2, lead_width/2+gap/2+lead_width/2+pad_size2/2))
    wg5.move((-pad_size1/2, lead_width/2+gap/2 +
             lead_width/2+pad_size2/2 + pad_size2 + gap))

    return D



def apodize_grating(semicircle_radius = 10,
            pitch_range = [0.370, 0.370],
            FF_range = [0.75, 0.25],
            apodize_teeth = 70,
            uniform_teeth = 10,
            taper_width_max=1.5,
            taper_width_min=0.25,
            grating_layer = 3,
            taper_length = 80
            ):
    '''
    edited by JY Yan
    Reference: Optics Express 29.13 (2021): 20205-20216.
    '''
    
    D = Device('Grating')
    # semicircle_radius = 12.5 # for 1550 nm
    # semicircle_radius = 2.5 # for 780 nm
    # grating_layer = 3
    # apodize_teeth = 70
    # uniform_teeth = 10
    # taper_width_max=1.5
    # taper_width_min=0.25
                
    # semicircle_radius = 10
    # pitch_range = [0.370, 0.370]
    # FF_range = [0.75, 0.25]
    pitch_list_apo = np.linspace(pitch_range[0], pitch_range[1], apodize_teeth)
    pitch_list_uni = np.linspace(pitch_range[1], pitch_range[1], uniform_teeth)
    FF_list_apo = np.linspace(FF_range[0], FF_range[1], apodize_teeth) 
    FF_list_uni = np.linspace(FF_range[1], FF_range[1], uniform_teeth)
    
    pitch_list = np.concatenate((pitch_list_apo, pitch_list_uni))
    FF_list = np.concatenate((FF_list_apo, FF_list_uni))
    
    #   
    semicircle_0 = pg.arc(radius = semicircle_radius/2, width = semicircle_radius, theta = 180, layer = grating_layer)
    semicircle_0.add_port(name = 3, midpoint = [0, 0], width = 10, orientation = -90)
    semicircle_0.add_port(name = 4, midpoint = [0, 0], width = 10, orientation = 90)
    semicircle = D<<semicircle_0
    
    r = semicircle_radius - pitch_list[0]*FF_list[0]/2
    for i, pitch in enumerate(pitch_list):
        r = r + pitch
        w = pitch*FF_list[i]
        B = pg.arc(radius = r, width = w, theta = 180, layer = grating_layer)
        B.add_port(name = 3, midpoint = [0, 0], width = 10, orientation = -90)
        teeth = D<<B
        # teeth = D << sw.wg(width1=20, width2=40, length=10, layer=1)
        teeth.connect(port=teeth.ports[3], destination=semicircle.ports[4])
    
    
    
    D.flatten(single_layer=grating_layer)
    D.add_port(1, port = semicircle.ports[3])
    # qp(D)
    
    A = Device('wg')
    wg1 = A << wg(width1=taper_width_max, width2=taper_width_min, length=taper_length, layer=grating_layer)
    grating = A<<D
    
    grating.connect(port=1, destination=wg1.ports[1])
    A.flatten(single_layer=grating_layer)
    A.add_port(1, port = wg1.ports[2])
    # qp(A) 
    
    # fname = "apodize grating"
    # A.write_gds(filename=fname+'.gds', unit=1e-6, precision=1e-9)
    return A

def apodize_grating_finite_angle(semicircle_radius = 10,
            pitch_range = [0.370, 0.370],
            FF_range = [0.75, 0.25],
            apodize_teeth = 70,
            uniform_teeth = 10,
            taper_width_max=1.5,
            taper_width_min=0.25,
            grating_layer = 3,
            taper_length = 80,
            open_angle = 90
            ):
    '''
    edited by JY Yan
    Reference: Optics Express 29.13 (2021): 20205-20216.
    '''
    if not (0 <= open_angle < 180):
        raise ValueError("open_angle must be within the range 0-180 degrees (excluding 180 degrees).")
    # Function implementation goes here
    pass

    D = Device('Grating')
    pitch_list_apo = np.linspace(pitch_range[0], pitch_range[1], apodize_teeth)
    pitch_list_uni = np.linspace(pitch_range[1], pitch_range[1], uniform_teeth)
    FF_list_apo = np.linspace(FF_range[0], FF_range[1], apodize_teeth) 
    FF_list_uni = np.linspace(FF_range[1], FF_range[1], uniform_teeth)
    
    pitch_list = np.concatenate((pitch_list_apo, pitch_list_uni))
    FF_list = np.concatenate((FF_list_apo, FF_list_uni))
    
    #   
    semicircle_0 = pg.arc(radius = semicircle_radius/2, width = semicircle_radius, theta = 180, layer = grating_layer)
    semicircle_0.add_port(name = 3, midpoint = [0, 0], width = 10, orientation = -90)
    semicircle_0.add_port(name = 4, midpoint = [0, 0], width = 10, orientation = 90)
    semicircle = D<<semicircle_0
    
    r = semicircle_radius - pitch_list[0]*FF_list[0]/2
    r_max = r+sum(pitch_list)+pitch_list[-1]
    
    
    for i, pitch in enumerate(pitch_list):
        r = r + pitch
        w = pitch*FF_list[i]
        B = pg.arc(radius = r, width = w, theta = 180, layer = grating_layer)
        B.add_port(name = 3, midpoint = [0, 0], width = 10, orientation = -90)
        teeth = D<<B
        # teeth = D << sw.wg(width1=20, width2=40, length=10, layer=1)
        teeth.connect(port=teeth.ports[3], destination=semicircle.ports[4])
    
    gr = D.flatten(single_layer=grating_layer)
    P = Device('angled')
    xpts = (0,                  0,                 0,-r_max,                                                    -r_max)
    ypts = (-taper_width_max/2, 0, taper_width_max/2, taper_width_max/2+r_max*np.tan(open_angle/2/180*np.pi),   -taper_width_max/2-r_max*np.tan(open_angle/2/180*np.pi))
    poly1 = P.add_polygon([xpts, ypts], layer = 0)
    poly1.rotate(-90)
    
    gr_P = P<<gr
    # qp(D)
    AND = pg.boolean(A = poly1, B = gr_P, operation = 'and', precision = 1e-6,
                    num_divisions = [1,1], layer = 0)
    AND.add_port(name = 1, midpoint = [0, 0], width = taper_width_max, orientation = -90)
    
    E = Device('angled_and')
    grating_angled = E<<AND
    # qp(E)

    
    A = Device('wg')
    wg1 = A << wg(width1=taper_width_max, width2=taper_width_min, length=taper_length, layer=grating_layer)
    grating = A<<E
    
    grating.connect(port=AND.ports[1], destination=wg1.ports[1])
    A.flatten(single_layer=grating_layer)
    A.add_port(1, port = wg1.ports[2])
    
    return A


# def MZI_with_expander(w_wg = .8, w_wg_mod = 1.0, l_mod = 3000, w_cpw = 30.0, 
#                       g_cpw = 5.0, w_pad = 60.0, g_pad = 20.0, l_pad = 100.0,
#                       l_expander = 300.0, splitter_taper_length = 300.0, 
#                       splitter_sbend_length = 600.0, layer = 1):
#     l_taper = 50.0
#     D = Device()
#     L = wg(w_wg,w_wg, length = 1.0)
#     SPLITTER = y_splitter(w0=w_wg,taper_length=splitter_taper_length, s_bend_length = splitter_sbend_length,
#                           separation=w_pad + g_pad)
#     EXP = parallel_expander(s1 = w_pad+g_pad, s2 = w_cpw+g_cpw, l = l_expander, 
#                             w = w_wg)
    
#     TAPER = parallel_tapers(w1 = w_wg, w2 = w_wg_mod, l = l_taper, 
#                             num_of_wgs = 2, spacing = w_cpw+g_cpw)
    
#     L_PAD = parallel_wgs(w = w_wg, l = l_pad, num_of_wgs = 2, 
#                          spacing = w_pad+g_pad)
#     L_MOD = parallel_wgs(w = w_wg_mod, l = l_mod-2*l_taper, num_of_wgs = 2, 
#                          spacing = w_cpw+g_cpw)
    
#     l1 = D.add_ref(L)#lead
#     s1 = D.add_ref(SPLITTER).connect(1, l1.ports[2])#first splitter
#     lpad1 = D.add_ref(L_PAD); lpad1.connect(1, s1.ports[4])
#     exp1 = D.add_ref(EXP); exp1.connect(1, lpad1.ports[2])
#     tap1 = D.add_ref(TAPER); tap1.connect(1, exp1.ports[2])
#     mod1 = D.add_ref(L_MOD); mod1.connect(1, tap1.ports[2])
#     tap2 = D.add_ref(TAPER); tap2.connect(2, mod1.ports[2])
#     exp2 = D.add_ref(EXP); exp2.connect(2, tap2.ports[1])
#     lpad2 = D.add_ref(L_PAD); lpad2.connect(1, exp2.ports[1])
#     s2 = D.add_ref(SPLITTER).connect(4, lpad2.ports[2])
#     l2 = D.add_ref(L); l2.connect(1, s2.ports[1])
    
#     D.center = (0,0)
#     D.flatten(single_layer = layer)
#     D.add_port(1, port = l1.ports[1])
#     D.add_port(2, port = l2.ports[2])
#     D.add_port('center', (0,0), w_cpw, 0)
#     return D

# %%


# D = Device()
# D << cpw_with_pad(w_pad=60.0, g_pad=30.0, l_pad=100.0, w_cpw=30.0,
#                   g_cpw=5.0, l_expander=300.0, l_mod=3000, w_metal=500.0,
#                   layer=1)
# qp(D)

# %%
# fname = str(datetime.now()).split(' ')[0].replace('-','')+'_'+'Basics'
# print(fname)
# if 1:
#     D.write_gds(filename=fname+'.gds', unit=1e-6, precision=1e-9)
