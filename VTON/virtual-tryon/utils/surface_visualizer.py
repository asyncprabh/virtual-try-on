import numpy as np
import pyvista as pv
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy.interpolate import griddata

class SurfaceVisualizer:
    def __init__(self):
        """Initialize the surface visualizer with default settings"""
        self.grid_size = 100
        self.contour_levels = 20
        self.wireframe_opacity = 0.3
        self.surface_opacity = 0.8
        self.colormap = 'viridis'
        self.plotter = pv.Plotter()
        
    def generate_sample_data(self, complexity=2):
        """Generate sample topographical data"""
        x = np.linspace(-5, 5, self.grid_size)
        y = np.linspace(-5, 5, self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Generate complex surface with multiple features
        Z = (np.sin(X) * np.cos(Y) + 
             np.exp(-(X**2 + Y**2)/8) * 2 +
             np.sin(complexity * np.sqrt(X**2 + Y**2)) * 0.5)
        
        return X, Y, Z
    
    def create_surface_plot(self, X, Y, Z, title="3D Surface Visualization"):
        """Create an interactive 3D surface plot"""
        # Create the surface mesh
        grid = pv.StructuredGrid(X, Y, Z)
        
        # Clear any existing plots
        self.plotter.clear()
        
        # Set up the visualization
        self.plotter.set_background('white')
        self.plotter.add_title(title, font_size=16)
        
        # Add the surface with smooth interpolation
        surf = self.plotter.add_mesh(
            grid,
            scalars=Z.ravel(),
            smooth_shading=True,
            specular=0.5,
            cmap=self.colormap,
            opacity=self.surface_opacity,
            show_edges=False
        )
        
        # Add wireframe overlay
        self.plotter.add_mesh(
            grid,
            style='wireframe',
            color='black',
            opacity=self.wireframe_opacity,
            line_width=0.5
        )
        
        # Add contour lines
        contours = grid.contour()
        self.plotter.add_mesh(
            contours,
            color='black',
            line_width=1,
            opacity=0.5
        )
        
        # Add axes and bounds
        self.plotter.show_bounds(
            grid=True,
            location='outer',
            all_edges=True
        )
        
        # Add orientation widget
        self.plotter.add_orientation_widget(interactive=True)
        
        # Add scalar bar
        self.plotter.add_scalar_bar(
            title='Elevation',
            vertical=True,
            position_x=0.85,
            position_y=0.05,
            width=0.1,
            height=0.7
        )
        
        return self.plotter
    
    def add_interactive_controls(self):
        """Add interactive controls to the visualization"""
        # Camera controls
        self.plotter.camera.zoom(1.5)
        self.plotter.camera.elevation = 45
        self.plotter.camera.azimuth = 45
        
        # Add text displaying controls
        control_text = (
            "Controls:\n"
            "Left Mouse: Rotate\n"
            "Middle Mouse: Pan\n"
            "Right Mouse: Zoom\n"
            "R: Reset Camera"
        )
        self.plotter.add_text(
            control_text,
            position='upper_left',
            font_size=12,
            shadow=True
        )
    
    def export_view(self, filename):
        """Export the current view to an image file"""
        self.plotter.screenshot(filename)
    
    def create_plotly_surface(self, X, Y, Z, title="3D Surface Visualization"):
        """Create an interactive surface plot using Plotly"""
        fig = go.Figure(data=[
            # Surface plot
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale=self.colormap,
                opacity=self.surface_opacity,
                showscale=True,
                contours={
                    "z": {
                        "show": True,
                        "usecolormap": True,
                        "project_z": True,
                        "width": 2
                    }
                }
            ),
            # Wireframe
            go.Scatter3d(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                mode='lines',
                line=dict(color='black', width=1),
                opacity=self.wireframe_opacity,
                showlegend=False
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene={
                'xaxis_title': 'X',
                'yaxis_title': 'Y',
                'zaxis_title': 'Z',
                'camera': {
                    'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}
                }
            },
            width=1000,
            height=800
        )
        
        return fig
    
    def save_plotly_html(self, fig, filename):
        """Save the Plotly figure as an interactive HTML file"""
        fig.write_html(filename)

def create_visualization(output_path, use_plotly=False):
    """Create and save a surface visualization"""
    visualizer = SurfaceVisualizer()
    X, Y, Z = visualizer.generate_sample_data(complexity=3)
    
    if use_plotly:
        # Create and save Plotly visualization
        fig = visualizer.create_plotly_surface(X, Y, Z)
        html_path = f"{output_path}/surface_plot.html"
        visualizer.save_plotly_html(fig, html_path)
        return html_path
    else:
        # Create PyVista visualization
        plotter = visualizer.create_surface_plot(X, Y, Z)
        visualizer.add_interactive_controls()
        image_path = f"{output_path}/surface_plot.png"
        visualizer.export_view(image_path)
        return image_path
