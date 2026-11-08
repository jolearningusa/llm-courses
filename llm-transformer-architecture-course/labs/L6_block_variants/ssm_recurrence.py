"""Small algebra fixtures for deck 23's optional Mamba-3 reading.

No trained block, Mamba library or GPU kernel. Python complex numbers represent
one two-real-coordinate state; phase is a discrete per-step angle in radians.
The MIMO helpers isolate the rank-R update/read algebra, not a full Mamba layer.
"""
import cmath
import math


def coefficients(delta, drift, mixing, phase=0.0):
    """alpha=exp(delta*drift+i*phase), beta=(1-mixing)*delta*alpha, gamma=mixing*delta.

    Parameters are fixed fixture inputs, not learned projections. No claim of
    second-order ODE accuracy for arbitrary mixing or step size is made.
    """
    if (not all(math.isfinite(v) for v in (delta, drift, mixing, phase))
            or delta <= 0 or drift > 0 or not 0 <= mixing <= 1):
        raise ValueError('need finite delta>0, drift<=0, mixing in [0,1] and finite phase')
    alpha = cmath.exp(delta * drift + 1j * phase)
    return alpha, (1 - mixing) * delta * alpha, mixing * delta


def stream(values, params, carry=(0j, 0j)):
    """h_t = alpha_t*h_prev + beta_t*v_prev + gamma_t*v_t.

    carry contains BOTH the recurrent state and previous state-input v=B*x.
    It is also returned for continuation across chunks. Empty chunks preserve it.
    """
    if len(values) != len(params):
        raise ValueError('one coefficient triple per input required')
    state, previous = carry
    states = []
    for value, (alpha, beta, gamma) in zip(values, params):
        state = alpha * state + beta * previous + gamma * value
        states.append(state)
        previous = value
    return states, (state, previous)


def reference(values, params, carry=(0j, 0j)):
    """Independent O(T^2) unrolled sum, not a scan or efficient training kernel.

    Each output is recomputed from the initial state and all earlier drives,
    without feeding previously computed states into the next output.
    """
    if len(values) != len(params):
        raise ValueError('one coefficient triple per input required')
    initial, previous = carry
    drives = [beta * (previous if i == 0 else values[i - 1]) + gamma * values[i]
              for i, (_, beta, gamma) in enumerate(params)]
    states = []
    for end in range(len(values)):
        total, weight = 0j, 1 + 0j
        for i in range(end, -1, -1):
            total += weight * drives[i]
            weight *= params[i][0]
        states.append(total + weight * initial)
    return states


def shape(matrix):
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError('nonempty rectangular matrix required')
    return len(matrix), len(matrix[0])


def mimo_update(state, decay, b, x):
    """H:(N,P), B:(N,R), X:(P,R); return decay*H + B*X^T.

    Input/output rank can grow without multiplying the N*P core state size.
    This real-valued, one-step fragment omits the trapezoidal previous drive.
    """
    n, p = shape(state)
    bn, rank = shape(b)
    xp, xr = shape(x)
    if (bn, xp, xr) != (n, p, rank):
        raise ValueError('inconsistent state/input/rank dimensions')
    return [[decay * state[i][j] + sum(b[i][r] * x[j][r] for r in range(rank))
             for j in range(p)] for i in range(n)]


def mimo_read(state, c):
    """C:(N,R); return H^T*C:(P,R). No residual gate or output projection."""
    n, p = shape(state)
    cn, rank = shape(c)
    if cn != n:
        raise ValueError('read projection must match state rows')
    return [[sum(state[i][j] * c[i][r] for i in range(n))
             for r in range(rank)] for j in range(p)]


if __name__ == '__main__':
    params = [coefficients(1.0, 0.0, 0.5)] * 2
    states, carry = stream([2.0, 4.0], params)
    print('real fixture states:', [z.real for z in states])
    print('max unrolled-reference error:', max(abs(a-b) for a, b in
          zip(states, reference([2.0, 4.0], params))))
    print('CPU algebra only; no learned quality or kernel timing measured')
