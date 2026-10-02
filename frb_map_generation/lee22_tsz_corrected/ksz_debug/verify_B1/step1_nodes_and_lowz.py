import numpy as np, sys
sys.path.insert(0, '.')
from b16_direct_np import *
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/verify_B1/'
cr = CacheReplica('/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/caches/tau_b16_sphere4p0_chordmean_interpolator.jld2')
logth, zs, logM, logy = cr.node_values()
print('z grid:', zs[:4], 'dz=', zs[1]-zs[0], 'dA ratio z1/z0 =', dA_mpc(zs[1])[0]/dA_mpc(zs[0])[0])
# 1) validate my direct g at exact cache nodes
lines=[]
maxdev=0
for iz in [0,1,2,5,20,100,255]:
    for im in [0,40,85,127]:
        M=10**logM[im]; z=zs[iz]; th200=theta200(M,z)[0]
        ith = [np.argmin(abs(np.exp(logth)-f*th200)) for f in (0.05,0.3,1,2,3.5)]
        th = np.exp(logth[ith])
        mine = g_direct(th, M, z)
        cache = np.exp(logy[im, iz, ith])
        dev = mine/cache-1; maxdev=max(maxdev, abs(dev).max())
        lines.append(f'node z={z:.4f} logM={logM[im]:.3f}: mine/cache-1 = ' + ' '.join(f'{d:+.2e}' for d in dev))
lines.append(f'max |mine/cache - 1| at nodes = {maxdev:.2e}')
# 2) off-node in z (theta at node, M at node): cache spline vs my direct
for z in [0.004, 0.0125, 0.0206, 0.03, 0.0375, 0.05, 0.0625, 0.1125, 0.2]:
    slab = cr.eval_z(z)
    for M in [1e13, 1e14, 1e15]:
        th200 = theta200(M, z)[0]
        xs = np.array([0.05, 0.3, 1, 2, 3.5])
        th = xs*th200
        c = cr.eval(th, M, z, slab=slab)
        d = g_direct(th, M, z)
        lines.append(f'z={z:.4f} M={M:.0e}: cache/direct-1 at x=0.05,0.3,1,2,3.5: ' + ' '.join(f'{v:+.4f}' for v in c/d-1))
open(OUT+'step1_nodes_and_lowz.txt','w').write('\n'.join(lines)+'\n')
print('\n'.join(lines))
