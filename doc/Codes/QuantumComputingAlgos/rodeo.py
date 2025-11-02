import random
import math

def optimize_zetaN(N, E_min, E_max, T, dt):
    if dt < 0:
        print("choose dt > 0")
        return
    if dt > 0:
        if T / dt > math.floor(T / dt) + 1e-10:
            print("T is not an integer multiple of dt")
            return
    # Number of iterations for annealing
    maxIter = 2000
    # Initial temperature
    T0 = 10.0
    # Cooling rate
    cooling = 0.999

    # Initial guess: equally divide T
    t = [random.random() for _ in range(N)]
    s = sum(t)
    if s != 0:
        t = [x / s * T for x in t]
    if dt > 0:
        k = random.randrange(N)
        t = [math.floor(x / dt) * dt for x in t]
        # ensure sum exactly T by adjusting one element
        total_except_k = sum(t[i] for i in range(N) if i != k)
        t[k] = T - total_except_k

    # Evaluate initial cost
    zeta_best = zetaN(t, E_min, E_max, N)
    t_best = t[:]

    T_curr = T0

    for iter in range(1, maxIter + 1):
        # Generate neighbor (perturbation that preserves sum t_i = T)
        perturb = max(0.1, 2 * dt)
        t_new = [t[i] + perturb * random.gauss(0, 1) for i in range(N)]
        t_new = [abs(x) for x in t_new]  # make positive
        s_new = sum(t_new)
        if s_new != 0:
            t_new = [x * T / s_new for x in t_new]  # normalize to satisfy sum = T
        if dt > 0:
            k = random.randrange(N)
            t_new = [math.floor(x / dt) * dt for x in t]
            total_except_k = sum(t_new[i] for i in range(N) if i != k)
            t_new[k] = T - total_except_k

        # Compute new cost
        zeta_new = zetaN(t_new, E_min, E_max, N)

        # Accept or reject
        delta = zeta_new - zeta_best
        if delta < 0 or random.random() < math.exp(-delta / T_curr):
            t = sorted(t_new, reverse=True)
            if zeta_new < zeta_best:
                zeta_best = zeta_new
                t_best = t[:]
                print(iter, zeta_best)
                print(t)

        # Cool down
        T_curr = T_curr * cooling

    # Display results
    print(f"Minimum zeta_N found: {zeta_best:.6f}")
    print("Corresponding t_i values:")
    print(t_best)

def optimize_geometric_zetaN(N, E_min, E_max, T, dt, alpha_array):
    if dt < 0:
        print("choose dt > 0")
        return

    zeta_array = []
    if dt == 0:
        nn = 0
        for alpha in alpha_array:
            nn += 1
            print(nn, len(alpha_array))
            t = [((-1 + alpha) * alpha ** (N - 1) * T) / ((alpha ** N) - 1) * (1 / (alpha ** i))
                 for i in range(N)]
            zeta_val = zetaN(t, E_min, E_max, N)
            zeta_array.append(zeta_val)
    else:
        if T / dt > math.floor(T / dt) + 1e-10:
            print("T is not an integer multiple of dt")
            return
        nn = 0
        for alpha in alpha_array:
            nn += 1
            print(nn, len(alpha_array))
            match = False
            Tcontinuum = T
            while not match:
                factor = ((alpha - 1) * alpha ** (N - 1) * Tcontinuum) / ((alpha ** N) - 1)
                t = [dt * round(factor * (1 / (alpha ** i))) for i in range(N)]
                total = sum(t)
                ratio = total / T
                if abs(ratio - 1) > 1e-10:
                    Tcontinuum = Tcontinuum / ratio
                else:
                    match = True
            zeta_val = zetaN(t, E_min, E_max, N)
            zeta_array.append(zeta_val)

    # Print the results (plotting omitted)
    print("alpha_array:", alpha_array)
    print("zeta_array:", zeta_array)

def sinc(x):
    if abs(x) < 1e-12:
        return 1.0
    else:
        return math.sin(math.pi * x) / (math.pi * x)

def zetaN(t, E_min, E_max, N):
    # Generate all combinations of {-1, 1} for k_i and k_i'
    M = 2 ** N
    states = []
    for i in range(M):
        bits = format(i, 'b').zfill(N)
        state = [2*int(bit) - 1 for bit in bits]
        states.append(state)

    z_min = 0.0
    z_max = 0.0
    for i in range(M):
        k = states[i]
        for j in range(M):
            kp = states[j]
            dot_sum = 0.0
            for d in range(N):
                dot_sum += (k[d] + kp[d]) * t[d]
            z_min += sinc(dot_sum * E_min / 2.0)
            z_max += sinc(dot_sum * E_max / 2.0)

    z = (E_max * z_max - E_min * z_min) / (2 ** (2 * N - 1))
    return z

if __name__ == "__main__":
    # Example usage from MATLAB script
    optimize_geometric_zetaN(10, 0.1, 4, 60, 0.1, [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2])
    optimize_zetaN(10, 0.1, 4, 60, 0)
