"""ATIVIDADE 01 - BISSEÇÃO E FALSA POSIÇÃO (MANNING / CANAL RETANGULAR).
CONSULTA NA PROVA: alterar B, S, n_manning, Q, intervalo e tolerância.
MATEMÁTICA: resolver f(H)=Qprevista(H)-Q=0. H em metros, Q em m³/s.
A condição f(a)*f(b)<0 exige função contínua e garante raiz no intervalo.
Bisseção: H=(a+b)/2; falsa posição: H=(a*f(b)-b*f(a))/(f(b)-f(a)).
"""
import numpy as np
B, S, n_manning, Q = 20.0, 0.0002, 0.03, 5.0  # Dados do PDF.
H_INFERIOR, H_SUPERIOR = 0.5, 1.0
TOL = 1e-6  # Critério do enunciado: abs(f(H)) < 10^-6.
MAX_ITER = 500

def vazao_prevista(H):
    """Q = sqrt(S)/n * (B*H)^(5/3) / (B+2H)^(2/3)."""
    if H <= 0:  # Profundidade física não pode ser nula ou negativa.
        return 0.0
    return np.sqrt(S) / n_manning * (B*H)**(5/3) / (B+2*H)**(2/3)

def f(H):
    return vazao_prevista(H) - Q  # Resíduo de vazão (m³/s).

def resolver(metodo, a, b, tol=TOL, max_iter=MAX_ITER):
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError('Escolha outro intervalo: f(a) e f(b) devem ter sinais opostos.')
    for k in range(1, max_iter+1):
        # Guardamos os extremos ANTIGOS para imprimir a situação da iteração.
        a_antigo, b_antigo = a, b
        if metodo == 'bissecao':
            x = (a+b)/2            # Metade exata do intervalo.
        elif metodo == 'falsa_posicao':
            x = (a*fb-b*fa)/(fb-fa)  # Interseção da secante com eixo f=0.
        else:
            raise ValueError('Método desconhecido.')
        fx = f(x)
        print(f'{metodo:13s} {k:03d} Hl={a_antigo:.10f} Hu={b_antigo:.10f} '
              f'Hr={x:.10f} f(Hr)={fx:+.3e}')
        if abs(fx) < tol:  # MESMO critério pedido para ambos os métodos.
            return x, k
        # Mantém somente a metade/subintervalo que ainda tem mudança de sinal.
        if fa*fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
    raise RuntimeError(f'{metodo} não atingiu a tolerância após {max_iter} iterações.')

if __name__ == '__main__':
    resultados = {}
    for metodo in ('bissecao', 'falsa_posicao'):
        H, passos = resolver(metodo, H_INFERIOR, H_SUPERIOR)
        U = Q/(B*H)       # Continuidade: U=Q/Área, Área=B*H.
        resultados[metodo] = (H, passos)
        print(f'{metodo}: H={H:.9f} m; U={U:.9f} m/s; passos={passos}; erro_Q={f(H):+.3e}\n')
        assert abs(f(H)) < TOL
    print('Diferença entre profundidades:', abs(resultados['bissecao'][0]-resultados['falsa_posicao'][0]))
    print('Iterações:', {m:d[1] for m,d in resultados.items()})
    # Ambos são intervalares por preservarem o sinal oposto nos extremos.
