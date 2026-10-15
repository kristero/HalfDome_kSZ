"""Ten deterministic, approximately uniform sightlines on the unit sphere."""
import numpy as np

def viewing_rotations(n=10):
    i=np.arange(n)
    z=1-2*(i+.5)/n
    phi=i*np.pi*(3-np.sqrt(5))
    normals=np.column_stack((np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z))
    matrices=[]
    for normal in normals:
        first=np.cross([0.,0.,1.],normal);first/=np.linalg.norm(first)
        second=np.cross(normal,first)
        matrix=np.vstack((first,second,normal))
        assert np.allclose(matrix@matrix.T,np.eye(3)) and np.linalg.det(matrix)>0
        matrices.append(matrix)
    return np.asarray(matrices)
