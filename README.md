# TAE_repo
# Input parameters
params = {'T': 573, 'nu_v': 6e12, 'nu_g': 6e10, 'nu_i': 6e12,
          'a0': 3.1648e-10, 'b': 0, 'G': 1e-2, 'ti.G_He':  1e-10, 'f': 0.1, 'Em_g': 0.06,
          'Em_v': 1.66, 'Em_i': 0.013, 'Eb_v_g': 4.57, 'Eb_v_2g': 6.65,
          'Eb_2g': 1.03, 'Ef_v': 3.65, 'Z_v': 1, 'Z_i': 1.4, 'rho': 1e14,
          'floor': 1e-18, 'epsilon': 10}

(T, nu_v, nu_g, nu_i, a0, b, G, G_He, f, Em_g, Em_v, Em_i, Eb_v_g, Eb_v_2g,
 Eb_2g, Ef_v, Z_v, Z_i, rho, floor, epsilon) = params.values()

ti = Mp(T, nu_v, nu_g, nu_i, a0, b, G, G_He, f, Em_g, Em_v,
        Em_i, Eb_v_g, Eb_v_2g, Eb_2g, Ef_v, Z_v, Z_i, rho, floor, epsilon)


def system_of_odes(t, y):
    # Assign aliases
    C_v, C_i, C_g, C_gv, C_2gv, C_2g, C_star, C_b, m = y

    # Initialize the derivative array
    dydt = np.zeros_like(y)

    # Rewrite the equations using the aliases
    # System of ODEs
    C_i = max(C_i, ti.floor)  # Reassign y with values enforced to be no less than floor_value

    # The time derivative of vacancy concentration (C_v)
    dydt[0] = (f*G + (ti.beta * ti.e1 + ti.delta) * C_gv
               - (ti.alpha * C_i + ti.beta * C_g
                  + ti.gamma * (ti.C_s_v + C_gv + 2 * C_2g + 3 * C_star)) * C_v)

# The time derivative of self-interstitial atoms concentration (C_i).
    dydt[1] = f*G - ti.alpha * (C_v + C_gv + 2 * C_2g + 3 * C_star + ti.C_s_i) * C_i

# The time derivative of gas atoms concentration (C_g).
    dydt[2] = ti.G_He + ti.beta * (ti.e1 * C_gv + ti.e2 * C_2g + 2 * ti.e3 * C_2g)
    + ti.delta * (C_gv + 2 * C_2g + 2 * (2 * C_2g) + 3 * C_star + m * C_b)
    + ti.alpha * C_i * (C_gv + 3 * C_star)
    - ti.beta * C_g * (C_v + 4 * C_g + C_gv )

#  The time derivative of C_gv (gas-vacancy complex concentration).
    dydt[3] = (ti.beta * ti.e2 * C_2gv + 2 * ti.delta * C_2gv + ti.beta * C_g * C_v
               + 2 * ti.gamma * C_v * C_2gv
               - C_gv * (ti.beta * ti.e1 + ti.delta + ti.alpha * C_i + ti.beta * C_v))

#  The time derivative of C_2gv (di-gas-vacancy complex concentration).
    dydt[4] = 3 * ti.delta * C_star + ti.beta * C_g * C_gv
    + 2 * ti.gamma * C_v * C_2gv
    - C_2gv * (2 * ti.beta * ti.e2 + 2 * ti.delta + 2 * ti.beta * C_v
               + 2 * ti.alpha * C_i + 2 * ti.gamma * C_v)

#  The time derivative of C_2g (di-gas complex concentration).
    dydt[5] = (2 * ti.alpha * C_i * C_2gv + 2 * ti.beta * C_g**2
               - C_2g * (ti.beta * ti.e3 + 2 * ti.delta
                         + 2 * ti.gamma * C_v + 2 * ti.beta * C_g))

# The time derivative of C_star (critical size nucleus concentration).
    dydt[6] = (2 * ti.beta * C_g * (C_2gv + C_2g)
               - 3 * C_star * (ti.delta + ti.alpha * C_i
                               + ti.beta * C_g + ti.gamma * C_v))

#  The time derivative of C_b (bubble concentration).
    dydt[7] = (12 * ti.beta * C_g * C_star + 9 * ti.gamma * C_v * C_star) / m

#    The time derivative of m (average gas atom count in a bubble).
    dydt[8] = ti.epsilon * ti.beta * C_g - ti.delta * m

    # Assign aliases
    y = C_v, C_i, C_g, C_gv, C_2gv, C_2g, C_star, C_b, m
    y = [max(val, ti.floor) for val in y]  # Reassign y with values enforced to be no less than floor_value

    return dydt


# Initial conditions
y0 = [ti.C_v_e, ti.floor, ti.floor, ti.floor, ti.floor, ti.floor,
      ti.floor, ti.floor, 4]

# Time span for the solution


t_span = (1e-8, 1e4)

# # Time points at which to solve the ODE
# t_eval = np.linspace(*t_span, 10000)
t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[1]), 10000)

# Solve the system of ODEs
solution = solve_ivp(system_of_odes, t_span, y0, method='BDF', t_eval=t_eval)

# Example plot for one of the concentrations (C_v)
plt.figure(figsize=(4, 3))
plt.loglog(solution.t, solution.y[0], label='C_v(t)')
plt.loglog(solution.t, solution.y[1], label='C_i(t)')
plt.loglog(solution.t, solution.y[2], label='C_h(t)')
plt.xlim([1e-7, 1e4])
plt.xlabel('Time t')
plt.ylabel('Concentration')
plt.title('Time Evolution of Unoccupied Single Vacancy Concentration')
plt.legend()
plt.show()

