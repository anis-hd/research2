import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

def visualize_results(original, reconstructed, entropy, losses, dists, rates, q_steps):
    """Enhanced visualization with proper vector field alignment"""
    plt.figure(figsize=(20, 15))
    
    # Create consistent sampling grid
    step = 16  # Controls vector density (16 = 16x16 grid)
    X, Y = np.meshgrid(np.arange(0, 256, step), np.arange(0, 256, step))
    
    # Original Flow Analysis
    orig_u = original[0, ::step, ::step, 0]
    orig_v = original[0, ::step, ::step, 1]
    orig_mag = np.sqrt(orig_u**2 + orig_v**2)
    
    # Reconstructed Flow Analysis
    recon_u = reconstructed[0, ::step, ::step, 0]
    recon_v = reconstructed[0, ::step, ::step, 1]
    recon_mag = np.sqrt(recon_u**2 + recon_v**2)
    
    # Error Calculations
    error_mag = np.abs(orig_mag - recon_mag)
    diff_u = orig_u - recon_u
    diff_v = orig_v - recon_v
    
    # 1. Original Motion Vector Field
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    im1 = ax1.quiver(X, Y, orig_u, orig_v, orig_mag, 
                    scale=50, cmap='jet', clim=[0, np.percentile(orig_mag, 95)])
    plt.colorbar(im1, ax=ax1)
    ax1.set_title("Original Motion Vector Field")
    
    # 2. Reconstructed Vector Field
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    im2 = ax2.quiver(X, Y, recon_u, recon_v, recon_mag,
                    scale=50, cmap='jet', clim=[0, np.percentile(orig_mag, 95)])
    plt.colorbar(im2, ax=ax2)
    ax2.set_title("Reconstructed Vector Field")
    
    # 3. Vector Difference Visualization
    ax3 = plt.subplot2grid((3, 3), (1, 0))
    im3 = ax3.quiver(X, Y, diff_u, diff_v, error_mag,
                    cmap='hot_r', scale=30)
    plt.colorbar(im3, ax=ax3)
    ax3.set_title("Vector Differences")
    
    # 4. Magnitude Comparison
    ax4 = plt.subplot2grid((3, 3), (1, 1))
    ax4.scatter(orig_mag.flatten(), recon_mag.flatten(), alpha=0.3)
    ax4.plot([0, orig_mag.max()], [0, orig_mag.max()], 'r--')
    ax4.set_xlabel("Original Magnitude")
    ax4.set_ylabel("Reconstructed Magnitude")
    ax4.set_title("Magnitude Correlation")
    
    # 5. Error Distribution
    ax5 = plt.subplot2grid((3, 3), (1, 2))
    ax5.hist(error_mag.flatten(), bins=50, color='purple')
    ax5.set_xlabel("Magnitude Error")
    ax5.set_title("Error Distribution")
    
    # 6. Training Metrics
    ax6 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    ax6.plot(losses, label='Total Loss')
    ax6.plot(dists, label='Distortion')
    ax6.plot(rates, label='Rate')
    ax6.plot(q_steps, label='Quant Step')
    ax6.legend()
    ax6.set_title("Training Progress")
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('motion_vector_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_in_notebook(original, reconstructed, entropy, losses, dists, rates, q_steps):
    """Interactive version with zoom capabilities and linked views"""
    fig = plt.figure(figsize=(20, 20))
    
    # Main flow visualization
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    ax3 = plt.subplot2grid((3, 3), (1, 0))
    ax4 = plt.subplot2grid((3, 3), (1, 1))
    ax5 = plt.subplot2grid((3, 3), (1, 2))
    ax6 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    
    # Plot main components
    orig_mag = np.sqrt(original[0,...,0]**2 + original[0,...,1]**2)
    recon_mag = np.sqrt(reconstructed[0,...,0]**2 + reconstructed[0,...,1]**2)
    error = np.abs(orig_mag - recon_mag)
    
    # Main magnitude plot with zoom capability
    im1 = ax1.imshow(orig_mag, cmap='jet', picker=True)
    ax1.set_title("Original Motion Vectors")
    
    # Detail view (connected via pick event)
    im2 = ax2.imshow(np.zeros_like(orig_mag), cmap='jet')
    ax2.set_title("Zoom Detail")
    
    # Error map
    im3 = ax3.imshow(error, cmap='hot')
    ax3.set_title("Magnitude Error")
    
    # Entropy map
    im4 = ax4.imshow(entropy.numpy().squeeze().mean(-1), cmap='viridis')
    ax4.set_title("Entropy Distribution")
    
    # Reconstructed flow
    im5 = ax5.imshow(recon_mag, cmap='jet')
    ax5.set_title("Reconstructed Flow")
    
    # Training curves
    ax6.plot(losses, label='Loss')
    ax6.plot(dists, label='Distortion')
    ax6.plot(rates, label='Rate')
    ax6.plot(q_steps, label='Quant Step')
    ax6.legend()
    ax6.set_title("Training Progress")
    
    # Add callback for interactive zoom
    def on_pick(event):
        if event.mouseevent.inaxes != ax1:
            return
            
        xint = int(event.mouseevent.xdata)
        yint = int(event.mouseevent.ydata)
        
        # Update detail view
        crop_size = 32
        detail = orig_mag[
            max(0,yint-crop_size):min(256,yint+crop_size),
            max(0,xint-crop_size):min(256,xint+crop_size)
        ]
        ax2.imshow(detail, cmap='jet', 
                  extent=(xint-crop_size, xint+crop_size, yint+crop_size, yint-crop_size))
        ax2.set_xlim(xint-crop_size, xint+crop_size)
        ax2.set_ylim(yint+crop_size, yint-crop_size)
        
        # Draw connection lines
        for conn in fig.connections:
            fig.connections.remove(conn)
            
        con = ConnectionPatch(
            xyA=(xint,yint), xyB=(xint,yint),
            coordsA="data", coordsB="data",
            axesA=ax1, axesB=ax2,
            color="white", linestyle="--"
        )
        fig.add_artist(con)
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('button_press_event', on_pick)
    plt.tight_layout()
    plt.show()