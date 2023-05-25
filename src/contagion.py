import random

import numpy as np


def bipartite_sis_to_matrices(
    I, g_x, g_y, b_x, b_y, f, g, s_x, s_y, tmin=0, tmax=20, dt=1, random_seed=None
):
    if random_seed is not None:
        random.seed(random_seed)

    # infect nodes
    n, m = np.shape(I)
    num_timesteps = int((tmax - tmin) / dt)
    x = np.zeros((num_timesteps, n))
    y = np.zeros((num_timesteps, m))
    x[0] = s_x
    y[0] = s_y

    for t in range(num_timesteps - 1):
        x[t + 1] = x[t]
        y[t + 1] = y[t]

        # infect people from the rooms
        for i in range(n):
            if x[t, i] == 1 and random.random() <= g_x[i] * dt:
                x[t + 1, i] += -1
            elif x[t, i] == 0 and random.random() <= b_x[i] * f(I[i], y[t]) * dt:
                x[t + 1, i] += 1

        # update the room infection status from people
        for j in range(m):
            if y[t, j] == 1 and random.random() <= g_y[j] * dt:
                y[t + 1, j] += -1
            elif y[t, j] == 0 and random.random() <= b_y[j] * g(I[:, j], x[t]) * dt:
                y[t + 1, j] += 1
    return x, y


def simulate_c_diff(
    H,
    g_x,
    g_y,
    b_x,
    b_y,
    patient_status,
    s_y,
    tmin=0,
    tmax=20,
    dt=1,
):
    # precompute
    patients = list(H.patients)
    wards = list(H.wards)
    rooms = list(H.rooms)
    # infect nodes
    t = tmin
    time = [t]
    Sp = [len([n for n in patients if patient_status[n] == "S"])]
    Ip = [len([n for n in patients if patient_status[n] == "I"])]
    Rp = [len([n for n in patients if patient_status[n] == "R"])]

    Sr = [len([e for e in rooms if s_y[e] == "S"])]
    Ir = [len([e for e in rooms if s_y[e] == "I"])]

    while t < tmax:
        new_patient_status = patient_status.copy()
        new_s_y = s_y.copy()

        Sp.append(Sp[-1])
        Ip.append(Ip[-1])
        Rp.append(Rp[-1])

        Sr.append(Sr[-1])
        Ir.append(Ir[-1])

        # infect patients from the wards
        for n in patients:
            if patient_status[n] == "I" and random.random() <= g_x * dt:
                new_patient_status[n] = "R"
                Ip[-1] += -1
                Rp[-1] += 1
            elif patient_status[n] == "S":
                for e in H.nodes.memberships(n):
                    if s_y[e] == "I" and random.random() <= b_x * dt:
                        new_patient_status[n] = "I"
                        Sp[-1] += -1
                        Ip[-1] += 1
                        break

        # update the ward infection status from patients
        for e in wards:
            if s_y[e] == "S":
                edge = H.edges.members(e)
                if random.random() <= len(
                    [n for n in edge if patient_status[n] == "I"]
                ) / len(edge):
                    new_s_y[e] = "I"
                    Sr[-1] += -1
                    Ir[-1] += 1

            elif s_y[e] == "I" and random.random() <= g_y * dt:
                new_s_y[e] = "S"
                Ir[-1] += -1
                Sr[-1] += 1

        # update the room infection status from doctors
        for r in rooms:
            if s_y[r] == "S":
                for r1 in H.rooms.neighbors(r):
                    if s_y[r1] == "I" and random.random() <= b_y * dt:
                        new_s_y[r] = "I"
                        Sr[-1] += -1
                        Ir[-1] += 1
                        break
        patient_status = new_patient_status
        s_y = new_s_y

        t += dt
        time.append(t)

    return (
        np.array(time),
        np.array(Sp),
        np.array(Ip),
        np.array(Rp),
        np.array(Sr),
        np.array(Ir),
    )


def simulate_c_diff_sis_flat(
    H,
    gamma,
    beta,
    f,
    s_x,
    tmin=0,
    tmax=20,
    dt=1,
):
    # infect nodes
    t = tmin
    time = [t]
    Sp = [len([n for n in H.nodes if s_x[n] == "S"])]
    Ip = [len([n for n in H.nodes if s_x[n] == "I"])]

    while t < tmax:
        new_s_x = s_x.copy()

        Sp.append(Sp[-1])
        Ip.append(Ip[-1])

        # infect people from the rooms
        for n in H.nodes:
            if s_x[n] == "I" and random.random() <= gamma[n] * dt:
                new_s_x[n] = "S"
                Ip[-1] += -1
                Sp[-1] += 1
            elif s_x[n] == "S":
                for e in H.nodes.memberships(n):
                    if random.random() <= beta[e] * f(H.edges.members(e), s_x) * dt:
                        new_s_x[n] = "I"
                        Sp[-1] += -1
                        Ip[-1] += 1
                        break

        s_x = new_s_x

        t += dt
        time.append(t)

    return np.array(time), np.array(Sp), np.array(Ip)
