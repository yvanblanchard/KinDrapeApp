import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CloughTocher2DInterpolator as CT2DInt
from KinDrape_eff_NR import KinDrape_eff_NR

def get_surface_function(surface_type):
    """Return the appropriate surface function and interpolator"""
    # Define grid for interpolation
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    if surface_type == "Hemisphere":
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                r = 1.0
                rxy = np.sqrt(X[i,j]**2 + Y[i,j]**2)
                if rxy < r:
                    Z[i,j] = np.sqrt(r**2 - X[i,j]**2 - Y[i,j]**2)
    else:  # Saddle
        Z = 0.5 * (X**2 - Y**2)
    
    # Create interpolator
    F = CT2DInt((X.ravel(), Y.ravel()), Z.ravel(), fill_value=np.min(Z))
    
    return F, X, Y, Z

def create_3d_mesh_plot(Node, CellShear, vertices_list):
    """Create a Plotly 3D mesh plot with shear angle coloring"""
    vertices = np.array(vertices_list)
    i, j, k = [], [], []
    
    # Triangulate quads
    for idx in range(0, len(vertices_list), 4):
        # First triangle
        i.extend([idx, idx+1, idx+2])
        j.extend([idx+1, idx+2, idx])
        k.extend([idx+2, idx, idx+1])
        # Second triangle
        i.extend([idx+2, idx+3, idx])
        j.extend([idx+3, idx, idx+2])
        k.extend([idx, idx+2, idx+3])
    
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i, j=j, k=k,
        colorbar_title='Shear Angle (deg)',
        colorscale='Viridis',
        intensity=np.repeat(CellShear, 4),
        showscale=True
    )
    
    return mesh

def get_surface_function(surface_type):
    """Return the appropriate surface function based on type"""
    if surface_type == "Hemisphere":
        def surface_func(x, y):
            r = 1.0
            rxy = np.sqrt(x**2 + y**2)
            if rxy < r:
                return np.sqrt(r**2 - x**2 - y**2)
            return 0.0
    else:  # Saddle
        def surface_func(x, y):
            return 0.5 * (x**2 - y**2)
    
    return surface_func

def app():
    st.title("Composite Draping Simulation")
    
    # Default values
    DEFAULT_D = 0.075
    DEFAULT_GRID = 24
    DEFAULT_ANG = 0
    DEFAULT_PRESHEAR = 0
    DEFAULT_ORGNODE = [11, 11]
    
    # Sidebar inputs
    st.sidebar.header("Simulation Parameters")
    
    # Surface type selection
    surface_type = st.sidebar.selectbox(
        "Mold Type",
        ["Hemisphere", "Saddle"],
        index=0
    )
    
    # Essential parameters
    d = st.sidebar.number_input(
        "Discretization Distance (d)",
        min_value=0.01,
        max_value=0.2,
        value=DEFAULT_D,
        step=0.005,
        format="%.3f"
    )
    
    grid_size = st.sidebar.number_input(
        "Grid Size",
        min_value=10,
        max_value=50,
        value=DEFAULT_GRID,
        step=1
    )
    
    ang = st.sidebar.slider(
        "Initial Draping Direction (degrees)",
        min_value=-90,
        max_value=90,
        value=DEFAULT_ANG
    )
    
    preshear = st.sidebar.slider(
        "Pre-shear Angle (degrees)",
        min_value=-45,
        max_value=45,
        value=DEFAULT_PRESHEAR
    )
    
    # Run simulation button
    if st.sidebar.button("Run Simulation"):
        with st.spinner("Running draping simulation..."):
            try:
                # Calculate Org based on d
                Org = [-d/2, -d/2]
                Grid = [grid_size, grid_size]

                Node, CellShear, _, _ = KinDrape_eff_NR(
                        d=d,
                        Grid=Grid,
                        Org=Org,
                        Ang=ang,
                        OrgNode=DEFAULT_ORGNODE,
                        PreShear=preshear,
                        Plt=False,
                        surface_type=surface_type
                    )
                
                # Prepare vertices for plotting
                vertices_list = []
                for i in range(Grid[0]-1):
                    for j in range(Grid[1]-1):
                        vertices = [
                            Node[i,j],
                            Node[i+1,j],
                            Node[i+1,j+1],
                            Node[i,j+1]
                        ]
                        vertices_list.extend(vertices)
                
                vertices_array = np.array(vertices_list)
                
                # Create Plotly figure
                fig = go.Figure()
                mesh = create_3d_mesh_plot(Node, CellShear, vertices_array)
                fig.add_trace(mesh)
                
                # Add surface visualization
                x = np.linspace(-1, 1, 100)
                y = np.linspace(-1, 1, 100)
                X, Y = np.meshgrid(x, y)
                surface_func = get_surface_function(surface_type)
                Z = np.array([[surface_func(xi, yi) for xi in x] for yi in y])
                
                surface = go.Surface(
                    x=X, y=Y, z=Z,
                    opacity=0.3,
                    showscale=False,
                    colorscale='Greys'
                )
                fig.add_trace(surface)
                
                # Update layout
                fig.update_layout(
                    title=f"{surface_type} Draping Simulation",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z",
                        aspectmode='data'
                    ),
                    width=800,
                    height=800
                )
                
                # Display statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Maximum Shear Angle", f"{np.nanmax(CellShear):.2f}°")
                with col2:
                    st.metric("Average Shear Angle", f"{np.nanmean(CellShear):.2f}°")
                with col3:
                    st.metric("Standard Deviation", f"{np.nanstd(CellShear):.2f}°")
                
                # Display plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Save and provide download
                results = {
                    'nodes': Node,
                    'shear': CellShear,
                    'parameters': {
                        'd': d,
                        'grid_size': grid_size,
                        'ang': ang,
                        'preshear': preshear,
                        'surface_type': surface_type
                    }
                }
                np.savez('draping_results.npz', **results)
                
                with open('draping_results.npz', 'rb') as f:
                    st.download_button(
                        label="Download Results (NPZ)",
                        data=f,
                        file_name="draping_results.npz",
                        mime="application/octet-stream"
                    )
                
            except Exception as e:
                st.error(f"An error occurred during simulation: {str(e)}")
    
    # Help section
    with st.expander("Help"):
        st.markdown("""
        ### KinDrape
        This is a demonstrator for kinematics-based draping simulation, using optimization (shear angle minimization).
                    
        Credits: 
        Krogh, C., Bak, B.L.V., Lindgaard, E. et al. "A simple MATLAB draping code for fiber-reinforced composites with application to optimization of manufacturing process parameters". Struct Multidisc Optim 64, 457–471 (2021).
        https://doi.org/10.1007/s00158-021-02925-z
        
        ### Parameter Descriptions
        - **Mold Type**: Choose between hemisphere or saddle mold surface
        - **Discretization Distance (d)**: Distance between nodes (default: 0.075)
        - **Grid Size**: Number of nodes in each direction (default: 24)
        - **Initial Draping Direction**: Starting fiber orientation angle (default: 0°)
        - **Pre-shear Angle**: Initial shear deformation (default: 0°)
        
        ### Fixed Parameters
        - Origin Node: [11, 11]
        - Origin Point: [-d/2, -d/2]
        """)

if __name__ == "__main__":
    app()