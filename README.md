# KinDrapeApp
This is a demonstrator for Kinematics-based composites draping simulation streamlit application.
Using optimization for shearing angle minimization (Newton-Raphson).

## Context
The hand layup process involves manually placing layers (plies) of fiber reinforcement materials (like carbon fiber or glass fiber fabrics) onto a mold surface. Each ply is carefully positioned and pressed down to conform to the mold geometry. The fabric is typically pre-impregnated with resin (prepreg) or resin is applied during layup. The layers are built up according to a specific stacking sequence designed to achieve desired mechanical properties in different directions.

![KinDrape_HandLayup](https://github.com/user-attachments/assets/17239ffc-9b4a-4629-80f1-1aa2098b053e)

Draping simulation is crucial for several reasons:
- Predicts how fabric will deform when conforming to complex surfaces
- Identifies potential manufacturing issues before actual production
- Optimizes the layup process by reducing trial-and-error in production
- Ensures structural performance by maintaining proper fiber orientations and preventing defects that could compromise mechanical properties

## Method
The kinematic draping algorithm simulates fabric deformation by treating the reinforcement as a pin-jointed net of inextensible fibers, where each node must lie on the target surface. Starting from an initial point and direction, it sequentially places nodes along generator paths while minimizing shear angles between fibers, then fills remaining areas using geometric constraints of constant fiber length and surface intersection.

![KinDrape_DrapingAlgorithm](https://github.com/user-attachments/assets/6e800182-e9e2-4881-845f-5e1ed4e997cd)

## Dependencies
- streamlit
- plotly
- matplotlib
- numpy
- scipy

## Usage
streamlit run app.py
- Select mold type: Hemisphere or Saddle
- Adjust settings
- Click on Run simulation
- Shearing angle analysis is displayed

![KinDrape_StreamlitApp](https://github.com/user-attachments/assets/41de4c92-551c-44ce-9a76-3fec84b53734)


## Next improvements
- Guiding (start) curve
- Energy-based algorithm
- Friction between layers
- Darts inclusion,...

## Credits
Krogh, C., Bak, B.L.V., Lindgaard, E. et al. "A simple MATLAB draping code for fiber-reinforced composites with application to optimization of manufacturing process parameters". 
Struct Multidisc Optim 64, 457–471 (2021).
https://doi.org/10.1007/s00158-021-02925-z
