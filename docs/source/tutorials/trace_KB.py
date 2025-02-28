# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2024-09-27"

Created with xrtQook




"""

import numpy as np
import sys, os

# sys.path.append(r"/home/rchernikov/anaconda3/envs/blop312/lib/python3.12/site-packages")
# os.environ['LD_PRELOAD']='/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
# os.environ['LD_PRELOAD']='/usr/lib64/libstdc++.so.6'
os.environ["QT_QPA_PLATFORM"] = "xcb"
# os.environ['PYOPENGL_PLATFORM'] = 'glx'
# os.environ['SDL_VIDEO_X11_FORCE_EGL'] = '1'
sys.path.append("/home/rcherniko/github/xrt")
# print("0")
import xrt.backends.raycing.sources as rsources

# print("")
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun
import time
# print("00")

limits = [[-0.6, 0.6], [-0.45, 0.45]]


def build_histRGB(lb, gb, limits=None, isScreen=False, shape=[256, 256]):
    # print(lb.x, lb.y, lb.z, lb.state)
    good = (lb.state == 1) | (lb.state == 2)
    if isScreen:
        x, y, z = lb.x[good], lb.z[good], lb.y[good]
    else:
        x, y, z = lb.x[good], lb.y[good], lb.z[good]
    goodlen = len(lb.x[good])
    hist2dRGB = np.zeros((shape[1], shape[0], 3), dtype=np.float64)
    hist2d = np.zeros((shape[1], shape[0]), dtype=np.float64)

    if limits is None and goodlen > 0:
        limits = np.array([[np.min(x), np.max(x)], [np.min(y), np.max(y)], [np.min(z), np.max(z)]])

    if goodlen > 0:
        beamLimits = [limits[1], limits[0]] or None
        flux = gb.Jss[good] + gb.Jpp[good]
        hist2d, yedges, xedges = np.histogram2d(y, x, bins=[shape[1], shape[0]], range=beamLimits, weights=flux)
        hist2dRGB = None
    return hist2d, hist2dRGB, limits


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.geometricSource01 = rsources.GeometricSource(
        bl=beamLine, center=[0, 0, 0], nrays=25000, energies=(9000, 100), distE="normal", dx=0.2, dz=0.1, dxprime=0.00015
    )

    # beamLine.toroidMirror01.R=38245.71081889952
    # beamLine.toroidMirror02.R=21035.140950394736

    beamLine.toroidMirror01 = roes.ToroidMirror(
        bl=beamLine,
        center=[0, 10000, 0],
        pitch=r"5deg",
        limPhysX=[-20.0, 20.0],
        limPhysY=[-150.0, 150.0],
        # R=55000,
        R=38245,
        # R=[10000, 2000],
        r=100000000.0,
    )
    print(f"{beamLine.toroidMirror01.R=}")

    beamLine.toroidMirror02 = roes.ToroidMirror(
        bl=beamLine,
        center=[0, 11000, r"auto"],
        pitch=r"5deg",
        yaw=r"10deg",
        positionRoll=r"90deg",
        rotationSequence=r"RyRxRz",
        limPhysX=[-20, 20],
        limPhysY=[-150, 150],
        # R=25000,
        R=21035,
        # R=[11000, 1000],
        r=100000000.0,
    )
    print(f"{beamLine.toroidMirror02.R=}")

    beamLine.screen01 = rscreens.Screen(bl=beamLine, center=[164.87347936545572, 11935, 343.73164815693235])
    # center=[r"auto", 11935, r"auto"])

    return beamLine


def run_process(beamLine):
    time0 = time.time()
    geometricSource01beamGlobal01 = beamLine.geometricSource01.shine()

    toroidMirror01beamGlobal01, toroidMirror01beamLocal01 = beamLine.toroidMirror01.reflect(
        beam=geometricSource01beamGlobal01
    )

    toroidMirror02beamGlobal01, toroidMirror02beamLocal01 = beamLine.toroidMirror02.reflect(beam=toroidMirror01beamGlobal01)

    screen01beamLocal01 = beamLine.screen01.expose(beam=toroidMirror02beamGlobal01)

    outDict = {
        "geometricSource01beamGlobal01": geometricSource01beamGlobal01,
        "toroidMirror01beamGlobal01": toroidMirror01beamGlobal01,
        "toroidMirror01beamLocal01": toroidMirror01beamLocal01,
        "toroidMirror02beamGlobal01": toroidMirror02beamGlobal01,
        "toroidMirror02beamLocal01": toroidMirror02beamLocal01,
        "screen01beamLocal01": screen01beamLocal01,
    }
    print("Tracing takes {:.3f}ms".format(1000 * (time.time() - time0)))
    beamLine.prepare_flow()
    return outDict


rrun.run_process = run_process


def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(label=r"x", limits=limits[0], bins=400, ppb=1, fwhmFormatStr="%.3f"),
        yaxis=xrtplot.XYCAxis(label=r"z", limits=limits[1], bins=300, ppb=1, fwhmFormatStr="%.3f"),
        caxis=xrtplot.XYCAxis(label=r"energy", unit=r"eV", bins=300, ppb=1),
        title=r"plot01",
        aspect="auto",
    )
    plots.append(plot01)
    return plots


def main():
    # print("1")
    beamLine = build_beamline()
    # print("2")
    beamLine.glow()
    # print("3")
    E0 = list(beamLine.geometricSource01.energies)[0]
    beamLine.alignE = E0
    plots = define_plots()
    xrtrun.run_ray_tracing(plots=plots, backend=r"raycing", beamLine=beamLine)


if __name__ == "__main__":
    main()
