import numpy as np

# Transforms 'control vector' into a [-1, 1] random uniform vector. (non-random params are 0)
def dat2vec(z):
    x = np.zeros( [13, len(z['ex_scale'])] )
    x[0] = (z['ex_scale'] - .35) / .15
    sgn = np.sign(z['ex_v0'])
    x[1] = sgn * np.sqrt(np.abs(z['ex_v0']))
    
    x[2] = z['fl_hardening'] - 9
    x[3] = np.log(z['fl_lambda0'] / 38888.9) / np.log(10)
    x[4] = np.log(z['fl_mu0'] / 58333.3) / np.log(10)
    x[5] = 2*np.log(z['fl_thetac'] / 2.5e-2) / np.log(10)
    x[6] = 2*np.log(z['fl_thetas'] / 7.3e-3) / np.log(10)
    
    x[7] = (z['m_dx']-.05)/.03
    x[8] = z['m_dy'] - .05
    x[9] = (z['m_x0'] - .45)/.03
    x[10] = (z['m_y0'] - .505) / .045
    
    x[11] = z['t_pause'] - .08
    x[12] = 8*(z['t_seg'] - .375)
    return x.T
