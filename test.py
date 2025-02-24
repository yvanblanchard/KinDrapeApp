# Script to define input and call KinDrape

import KinDrape_eff_NR as kd
import time

# Input parameters to KinDrape:
# d: discretization distance 
# Grid: Dimensions of fabric grid (rows, columns)
# Org: x,y origin point on mold
# Ang: Initial draping direction rel. to y-axis
# OrgNode: Origin node of grid (row, column)

# Starting point on the north pole
d = 0.075; Grid = [24,24]; Org = [-d/2, -d/2]; Ang = 0; OrgNode = [11,11]
PreShear = 0.0 ; Plt = False

# Starting point away from the north pole
#d = 0.075; Grid = [24,24]; Org = [0,-0.9]; Ang = 60; OrgNode = [3,3]

# Time and call KinDrape
tic = time.time()
#Node, Shear, ax, fig = kd.KinDrape(d, Grid, Org, Ang, OrgNode) 
Node, Shear, ax, fig = kd.KinDrape_eff_NR(d, Grid, Org, Ang, OrgNode, PreShear, Plt) 

print('Time spent on drape analysis: ', time.time() - tic, ' s')
