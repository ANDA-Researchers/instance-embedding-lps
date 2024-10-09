import numpy as np
import matplotlib.pyplot as plt

def draw_point(pcl, indices=None, name='pcl_img.png', s=1):
    # Assuming `embedding` is your point cloud and `indices` are the indices of the centroids
    embedding_np = pcl  # Convert to numpy array for plotting
    
    # Create a new color array
    colors = np.full(embedding_np.shape[0], 'w')  # All points are blue
    
    # Create a new size array
    sizes = np.full(embedding_np.shape[0], s)  # All points are size 1
    
    if indices is not None:
        indices_np = indices.flatten()  # Flatten the indices array for indexing
        colors[indices_np] = 'r'  # Centroid points are red
        sizes[indices_np] = 10  # Centroid points are size 20

    # Create a 3D plot
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot()

    # Plot the point cloud in 3D
    scatter = ax.scatter(embedding_np[:, 0], embedding_np[:, 1], c=colors, s=sizes)

    # Set background color to black
    ax.set_facecolor('black')
    ax.set_aspect('auto')
    plt.savefig(name)

def write_img(img, name='heatmap.png'):
    plt.imsave(name, img)

def draw_point_with(pcl, indices=None, name='pcl_img.png', s=1, pcl2=None, s2=1):
    # Assuming `embedding` is your point cloud and `indices` are the indices of the centroids
    embedding_np = pcl  # Convert to numpy array for plotting
    
    # Create a new color array
    colors = np.full(embedding_np.shape[0], 'w')  # All points are blue
    
    # Create a new size array
    sizes = np.full(embedding_np.shape[0], s)  # All points are size 1
    
    if indices is not None:
        indices_np = indices.flatten()  # Flatten the indices array for indexing
        colors[indices_np] = 'b'  # Centroid points are red
        sizes[indices_np] = 10  # Centroid points are size 20

    # Create a 3D plot
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot()

    

    # Plot the point cloud in 3D
    scatter = ax.scatter(embedding_np[:, 0], embedding_np[:, 1], c=colors, s=sizes)
    if pcl2 is not None:
        scatter = ax.scatter(pcl2[:, 0], pcl2[:, 1], c='r', s=s2)

    # Set background color to black
    ax.set_facecolor('black')
    ax.set_aspect('auto')

    plt.savefig(name)
