"""PPC #4 - Calculo Numerico Aplicado / UnB (2026.2).

Modelo termico tridiagonal, algoritmo de Thomas proprio e gradiente
 descendente com passo otimo; sem inversao matricial nem otimizadores prontos.

Execucao: python ppc4.py
Dependencias: numpy, matplotlib
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Parametros conforme Tabela 3 do enunciado
L = 0.30             # m
K = 50.0             # W/(m.K)
AC = 1.0e-4         # m^2
PERIMETRO = 0.040   # m
HCONV = 20.0        # W/(m^2.K)
TINF = 25.0          # Celsius
N = 20
DTREF = 60.0        # K
QREF = 1.0          # W
TOL = 1.0e-6
MAXITER = 10000
LAMBDAS = [0.001, 0.02, 0.2]

DX = L / N
G = K * AC / DX
C = HCONV * PERIMETRO * DX
X = (np.arange(N, dtype=float) + 0.5) * DX
TD = TINF + 60.0 * np.exp(-((X - L/2.0)/(0.20*L))**2)
# A e' tridiagonal: sub e superdiag = -G, diagonal = 2G+C,
# extremidades = G+C (faces externas isoladas).
DIAG = np.full(N, 2.0 * G + C)
DIAG[0] = DIAG[-1] = G + C
LOWER = np.full(N - 1, -G)
UPPER = np.full(N - 1, -G)


def thomas(lower, diagonal, upper, rhs):
    """Resolve sistema tridiagonal pelo algoritmo de Thomas, sem np.linalg.solve."""
    a = np.asarray(lower, dtype=float)
    b = np.asarray(diagonal, dtype=float)
    c = np.asarray(upper, dtype=float)
    d = np.asarray(rhs, dtype=float)
    n = len(b)
    if len(a) != n - 1 or len(c) != n - 1 or len(d) != n:
        raise ValueError('Dimensoes invalidas do sistema tridiagonal.')
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    pivot = b[0]
    if abs(pivot) < 1e-14:
        raise ZeroDivisionError('Pivo nulo na eliminacao de Thomas.')
    if n > 1:
        cp[0] = c[0] / pivot
    dp[0] = d[0] / pivot
    for i in range(1, n):
        pivot = b[i] - a[i - 1] * cp[i - 1]
        if abs(pivot) < 1e-14:
            raise ZeroDivisionError(f'Pivo nulo de Thomas na linha {i + 1}.')
        if i < n - 1:
            cp[i] = c[i] / pivot
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / pivot
    solution = np.empty(n)
    solution[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = dp[i] - cp[i] * solution[i + 1]
    return solution


def solve_a(rhs):
    return thomas(LOWER, DIAG, UPPER, rhs)


def temperaturas(q):
    q = np.asarray(q, dtype=float)
    return solve_a(q + C * TINF * np.ones(N))


def objetivo(q, lamb):
    r = temperaturas(q) - TD
    return 0.5 * (np.dot(r, r)/(DTREF**2) + lamb*np.dot(q, q)/(QREF**2))


def objetivo_gradiente(q, lamb):
    """Avalia J, gradiente e T usando dois sistemas Thomas."""
    t = temperaturas(q)
    r = t - TD
    j = 0.5 * (np.dot(r, r)/(DTREF**2) + lamb*np.dot(q, q)/(QREF**2))
    p = solve_a(r / (DTREF**2))
    grad = p + (lamb/(QREF**2))*q
    return j, grad, t


def verifica_gradiente(q, lamb, indices=(0, 9, 19), eps=1.0e-5):
    _, g, _ = objetivo_gradiente(q, lamb)
    verificacoes = []
    for i in indices:
        delta = np.zeros(N)
        delta[i] = eps
        g_num = (objetivo(q + delta, lamb) - objetivo(q - delta, lamb))/(2.0*eps)
        verificacoes.append((i + 1, g[i], g_num, abs(g[i] - g_num)))
    return verificacoes


def otimizar(lamb):
    q = np.zeros(N)
    hist_j = []
    hist_g = []
    hist_alpha = []
    convergiu = False
    for it in range(MAXITER + 1):
        j, g, t = objetivo_gradiente(q, lamb)
        normg = float(np.sqrt(np.dot(g, g)))
        hist_j.append(float(j))
        hist_g.append(normg)
        if normg < TOL:
            convergiu = True
            break
        if it == MAXITER:
            break
        z = solve_a(g)
        gg = np.dot(g, g)
        denom = np.dot(z, z)/(DTREF**2) + lamb*gg/(QREF**2)
        alpha = gg/denom
        hist_alpha.append(float(alpha))
        q = q - alpha*g
    return {
        'lambda': lamb,
        'q': q,
        't': t,
        'j0': hist_j[0],
        'j': hist_j[-1],
        'normg': hist_g[-1],
        'iterations': len(hist_j) - 1,
        'converged': convergiu,
        'rms': float(np.sqrt(np.mean((t - TD)**2))),
        'qnorm': float(np.sqrt(np.dot(q, q))),
        'negatives': int(np.count_nonzero(q < -1e-10)),
        'history_j': np.array(hist_j),
        'history_g': np.array(hist_g),
        'history_alpha': np.array(hist_alpha),
    }


def salvar_graficos(resultados, pasta):
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 145})
    fig, ax = plt.subplots(figsize=(7.3, 4.15))
    ax.plot(X*100, TD, color='black', linestyle='--', linewidth=2, label='Perfil desejado')
    for res in resultados:
        ax.plot(X*100, res['t'], marker='o', markersize=3, label=f"$\\lambda={res['lambda']:g}$")
    ax.set(xlabel='Posicao na barra (cm)', ylabel='Temperatura (graus C)', title='Perfil de temperaturas')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout(); fig.savefig(pasta/'perfil_temperaturas.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 4.15))
    for res in resultados:
        ax.plot(X*100, res['q'], marker='o', markersize=3, label=f"$\\lambda={res['lambda']:g}$")
    ax.axhline(0, color='black', linewidth=0.9)
    ax.set(xlabel='Posicao na barra (cm)', ylabel='Potencia por aquecedor (W)', title='Potencias otimizadas')
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(pasta/'potencias.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 4.15))
    for res in resultados:
        ax.semilogy(np.arange(len(res['history_j'])), res['history_j'], label=f"$\\lambda={res['lambda']:g}$")
    ax.set(xlabel='Iteracao', ylabel='Funcao objetivo J (escala log.)', title='Historicos de convergencia')
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(pasta/'historicos_J.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 4.15))
    for res in resultados:
        ax.semilogy(np.arange(len(res['history_g'])), res['history_g'], label=f"$\\lambda={res['lambda']:g}$")
    ax.axhline(TOL, color='black', linestyle='--', linewidth=1, label='Tolerancia')
    ax.set(xlabel='Iteracao', ylabel='Norma do gradiente', title='Criterio de convergencia')
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(pasta/'historicos_gradiente.png'); plt.close(fig)


def main():
    pasta = Path(__file__).resolve().parent
    q0 = np.zeros(N)
    teste = temperaturas(q0)
    erro_teste = float(np.max(np.abs(teste - TINF)))
    verificacoes = verifica_gradiente(np.full(N, 0.5), 0.02)
    resultados = [otimizar(lam) for lam in LAMBDAS]
    salvar_graficos(resultados, pasta)
    with (pasta / 'resultados.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda', 'iteracoes', 'convergiu', 'J_inicial', 'J_final', 'norma_gradiente', 'erro_rms_K', 'norma_potencia_W', 'potencias_negativas'])
        for r in resultados:
            writer.writerow([r[k] for k in ('lambda', 'iterations', 'converged', 'j0', 'j', 'normg', 'rms', 'qnorm', 'negatives')])
    with (pasta / 'perfis.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['volume', 'x_m', 'T_desejada_C'] + [f'T_lambda_{r["lambda"]}' for r in resultados] + [f'Q_lambda_{r["lambda"]}' for r in resultados])
        for i in range(N):
            writer.writerow([i+1, X[i], TD[i]] + [r['t'][i] for r in resultados] + [r['q'][i] for r in resultados])

    print(f'DX = {DX:.8f} m; G = {G:.8f} W/K; C = {C:.8f} W/K')
    print(f'Teste Q=0: max |T-25| = {erro_teste:.3e} K')
    print('Verificacao por diferencas finitas (Q_i=0.5 W, lambda=0.02):')
    for ix, ga, gn, ea in verificacoes:
        print(f'  i={ix:2d}: grad analitico={ga:+.11e}, diferencas={gn:+.11e}, erro={ea:.3e}')
    print('\nResultados:')
    for r in resultados:
        print(f" lambda={r['lambda']:.3f} | convergiu={r['converged']} | iter={r['iterations']:5d} | J0={r['j0']:.12g} | J*={r['j']:.12g} | ||g||={r['normg']:.3e} | ET={r['rms']:.7g} K | ||Q||={r['qnorm']:.7g} W | negativas={r['negatives']}")
        print('    Q=', np.array2string(r['q'], precision=6, suppress_small=False, max_line_width=120))
    # Verificacoes internas de consistencia: monotonicidade ate arredondamento
    assert erro_teste < 1e-9
    assert max(row[-1] for row in verificacoes) < 1e-6
    assert all(np.all(np.diff(r['history_j']) <= 1e-9) for r in resultados)
    # Todos os casos devem convergir com a tolerancia e o limite do enunciado.
    assert all(r['converged'] for r in resultados), 'Algum caso nao atingiu tolerancia.'


if __name__ == '__main__':
    main()
