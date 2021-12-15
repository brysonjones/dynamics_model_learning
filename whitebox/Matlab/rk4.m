function xkp1 = rk4(x,u,p,dt)
k1 = dt*quadDynamics(x,u,p);
k2 = dt*quadDynamics(x + k1/2,u,p);
k3 = dt*quadDynamics(x + k2/2,u,p);
k4 = dt*quadDynamics(x + k3,u,p);
xkp1 = x + (1/6)*(k1 + 2*k2 + 2*k3 + k4);
end