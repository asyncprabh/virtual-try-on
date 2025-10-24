import trimesh
import numpy as np
from pathlib import Path
import os
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import cv2

class ClothingModelProcessor:
    def __init__(self, model_path):
        """
        Initialize the clothing model processor
        Args:
            model_path (str): Path to the 3D model file (.obj, .stl)
        """
        self.model_path = model_path
        self.mesh = None
        self.body_mesh = None
        self.supported_formats = ['.obj', '.stl']
        self.texture = None
        self.uv_coords = None

    def load_model(self):
        """Load the 3D model file"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        file_ext = Path(self.model_path).suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format. Supported formats: {self.supported_formats}")
        
        try:
            self.mesh = trimesh.load(self.model_path)
            return True
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def load_body_mesh(self, body_mesh_path):
        """
        Load the target body mesh for alignment
        Args:
            body_mesh_path (str): Path to the body mesh file
        """
        try:
            self.body_mesh = trimesh.load(body_mesh_path)
        except Exception as e:
            raise Exception(f"Error loading body mesh: {str(e)}")

    def compute_laplacian_matrix(self):
        """Compute Laplacian matrix for mesh deformation"""
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        
        # Construct adjacency matrix
        V = len(vertices)
        A = sparse.lil_matrix((V, V), dtype=np.float32)
        
        for face in faces:
            for i in range(3):
                A[face[i], face[(i + 1) % 3]] = 1
                A[face[(i + 1) % 3], face[i]] = 1
        
        # Compute Laplacian matrix
        D = sparse.diags(A.sum(axis=1).flatten())
        L = D - A
        
        return L.tocsr()

    def deform_to_body(self, constraint_points, target_points, weight=1.0):
        """
        Deform the clothing mesh to fit the body using Laplacian deformation
        Args:
            constraint_points (list): Indices of constraint vertices
            target_points (np.array): Target positions for constraint vertices
            weight (float): Weight for constraint enforcement
        """
        if len(constraint_points) != len(target_points):
            raise ValueError("Number of constraint points must match target points")
        
        # Compute Laplacian matrix
        L = self.compute_laplacian_matrix()
        V = len(self.mesh.vertices)
        
        # Setup linear system
        A = sparse.vstack([L, weight * sparse.eye(V)])
        b = np.zeros((L.shape[0] + V, 3))
        
        # Add positional constraints
        for i, (idx, target) in enumerate(zip(constraint_points, target_points)):
            b[L.shape[0] + idx] = weight * target
        
        # Solve for new vertex positions
        x = spsolve(A.T @ A, A.T @ b)
        self.mesh.vertices = x

    def apply_texture_mapping(self, texture_path):
        """
        Apply texture mapping to the clothing mesh
        Args:
            texture_path (str): Path to texture image
        """
        try:
            # Load texture image
            self.texture = cv2.imread(texture_path)
            if self.texture is None:
                raise ValueError("Failed to load texture image")
            
            # Generate UV coordinates if not present
            if self.mesh.visual.uv is None:
                self.generate_uv_mapping()
            else:
                self.uv_coords = self.mesh.visual.uv
            
            # Apply texture to mesh
            self.mesh.visual = trimesh.visual.TextureVisuals(
                uv=self.uv_coords,
                image=self.texture
            )
        except Exception as e:
            raise Exception(f"Error applying texture: {str(e)}")

    def generate_uv_mapping(self):
        """Generate UV mapping for the mesh using cylindrical projection"""
        vertices = self.mesh.vertices
        
        # Calculate cylindrical coordinates
        center = np.mean(vertices, axis=0)
        vertices_centered = vertices - center
        
        # Convert to cylindrical coordinates
        r = np.sqrt(vertices_centered[:, 0]**2 + vertices_centered[:, 2]**2)
        theta = np.arctan2(vertices_centered[:, 2], vertices_centered[:, 0])
        
        # Generate UV coordinates
        u = (theta + np.pi) / (2 * np.pi)
        v = (vertices_centered[:, 1] - np.min(vertices_centered[:, 1])) / \
            (np.max(vertices_centered[:, 1]) - np.min(vertices_centered[:, 1]))
        
        self.uv_coords = np.column_stack((u, v))

    def adjust_scale(self, scale_factor):
        """Scale the model uniformly"""
        if self.mesh is None:
            raise ValueError("No model loaded")
        self.mesh.apply_scale(scale_factor)

    def align_to_body(self, body_landmarks):
        """
        Align the clothing model to body landmarks
        Args:
            body_landmarks (dict): Dictionary of body landmark coordinates
        """
        if self.mesh is None:
            raise ValueError("No model loaded")
        
        # Calculate transformation matrix based on body landmarks
        clothing_points = np.array([
            self.mesh.bounds[0],  # min bounds
            self.mesh.bounds[1],  # max bounds
            self.mesh.center_mass
        ])
        
        target_points = np.array([
            np.min(list(body_landmarks.values()), axis=0),
            np.max(list(body_landmarks.values()), axis=0),
            np.mean(list(body_landmarks.values()), axis=0)
        ])
        
        # Calculate optimal rigid transformation
        matrix = trimesh.transformations.align_vectors(clothing_points, target_points)
        self.mesh.apply_transform(matrix)

    def smooth_surface(self, iterations=1, lambda_factor=0.5):
        """
        Apply Laplacian smoothing to the mesh surface
        Args:
            iterations (int): Number of smoothing iterations
            lambda_factor (float): Smoothing factor (0-1)
        """
        if self.mesh is None:
            raise ValueError("No model loaded")
        
        vertices = self.mesh.vertices.copy()
        L = self.compute_laplacian_matrix()
        
        for _ in range(iterations):
            # Compute Laplacian coordinates
            delta = L @ vertices
            
            # Update vertices
            vertices = vertices - lambda_factor * (L @ vertices)
        
        self.mesh.vertices = vertices

    def export_model(self, output_path):
        """Export the processed model"""
        if self.mesh is None:
            raise ValueError("No model loaded")
        
        file_ext = Path(output_path).suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported export format. Supported formats: {self.supported_formats}")
        
        try:
            self.mesh.export(output_path)
            return True
        except Exception as e:
            raise Exception(f"Error exporting model: {str(e)}")
