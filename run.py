import numpy as np
from utils.solvers import *



if __name__ == "__main__":
    jax.config.update("jax_platform_name", "gpu")
    devices = jax.devices()
    print(f"Available Jax devices: {devices}")

    GridSpec = namedtuple("GridSpec", ["N", "h"])
    Lx, Ly, Lz = 1.1, 1.0, 0.005
    Nx, Ny, Nz = 154, 140, 1
    dtype = jnp.float64
    grid_spec = GridSpec(N=(Nx, Ny, Nz), h=(Lx / Nx, Ly / Ny, Lz / Nz))
    
    # Initialise base material
    lmbda_solid = 121.15  #GPa [kN/mm2]   
    mu_solid    = 80.77  
    mu, lmbda   = initialise_material(grid_spec, shape="homogeneous",
                                      mu_mat=mu_solid, lmbda_mat=lmbda_solid, dtype=dtype)
    # Phase-field parameters
    base_gc = 2.7e-3  
    lc = 0.030 
    k_stab = 1e-6
    
    
 
    # Generate loading steps: 
    load_steps = [jnp.array([0.0, eps_yy, 0.0, 0.0, 0.0, 0.0], dtype=dtype) 
                  for eps_yy in np.concatenate((np.linspace(0, 0.005, 10), np.linspace(0.00505, 0.007, 200)))]

    # Determine indices of steps to save based on target macroscopic strains
    eps_values = np.array([np.array(s)[1] for s in load_steps])
    target_strains = [4.6e-3, 5e-3]
    save_steps = []
    for t in target_strains:
        idx = int(np.argmin(np.abs(eps_values - t)))
        save_steps.append(idx)
    save_steps = sorted(set(save_steps + [len(load_steps) - 1]))
    print(f"Will save full 3D fields at steps: {save_steps} corresponding to strains {[eps_values[i] for i in save_steps]}")

    d_hist, eps_hist, sigma_hist, eps_lst, sig_lst = solve_fracture_staggered(
        grid_spec, lmbda, mu, base_gc, lc, load_steps, d_0=None, k=k_stab, save_steps=save_steps
    )
    
    #visuals
    sig_lst = np.array(sig_lst)
    eps_lst = np.array(eps_lst)
    #print(sig_lst.shape)

    plt.figure()
    plt.plot(eps_lst[:,1], sig_lst[:,1], '-*')
    plt.title("Macroscopic Stress-Strain Curve")
    plt.xlabel("Strain (eps_yy)")
    plt.ylabel("Stress (sigma_yy)")
    plt.savefig("newsigma.png", bbox_inches="tight") 
    plt.show()
    

    # Plot and save damage maps for saved steps
    print(f"Available saved damage steps: {sorted(d_hist.keys())}")
    for step in sorted(d_hist.keys()):
        print(f"Plotting damage map for step: {step}")
        dmap = np.array(d_hist[step])
        eps_val = float(np.array(load_steps[step])[1])
        plt.figure()
        plt.imshow(dmap[:, :, 0].T, cmap='coolwarm', vmin=0.0, vmax=1.0)
        plt.colorbar(label="Damage (d)")
        plt.xticks([])
        plt.yticks([])
        img_name = f"dmap_step_{step}_eps_{eps_val:.4e}.png"
        plt.savefig(img_name, bbox_inches="tight")
        np.savez(f"dmap_step_{step}_eps_{eps_val:.4e}.npz", d=dmap, eps=eps_val)
        plt.show()
    
save_filename = "stress_strain_data_0.060h.npz"
np.savez(save_filename, eps=eps_lst, sig=sig_lst)