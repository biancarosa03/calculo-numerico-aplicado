"""ATIVIDADE 02 - NEWTON-RAPHSON E SECANTE (CONVECÇÃO + RADIAÇÃO).
Altere A, h, epsilon, sigma, Tinf e P para outro enunciado.
Balanço em regime permanente: P=h A(T-Tinf)+eps sigma A(T^4-Tinf^4).
A raiz f(T)=0 é a temperatura em K que equilibra a energia.
Newton usa derivada; secante NÃO usa derivada (ideal para função caixa-preta).
"""
A, h, epsilon, sigma, Tinf, P = .01, 10., .8, 5.67e-8, 300., 20.
TOL, MAX_ITER = 1e-6, 100  # Critério principal: diferença entre iteradas em K.

def f(T):
    return h*A*(T-Tinf) + epsilon*sigma*A*(T**4-Tinf**4) - P

def derivada(T):
    return h*A + 4*epsilon*sigma*A*T**3

def newton(T=350.0):
    for k in range(1, MAX_ITER+1):
        df = derivada(T)
        if abs(df) < 1e-14:
            raise ZeroDivisionError('Derivada nula: Newton não pode avançar.')
        Tnovo = T - f(T)/df  # Atualização T(k+1) = T(k) - f(Tk)/f’(Tk).
        erro = abs(Tnovo-T)
        print(f'Newton {k:02d}: T={Tnovo:.9f} K f(T)={f(Tnovo):+.4e} ΔT={erro:.3e}')
        if erro < TOL and abs(f(Tnovo)) < 1e-7:
            return Tnovo, k
        T = Tnovo
    raise RuntimeError('Newton não convergiu.')

def secante(T0=400., T1=425.):
    # Valores iniciais saem da tabela do PDF: f(400)<0, f(425)>0.
    for k in range(1, MAX_ITER+1):
        f0, f1 = f(T0), f(T1)
        if abs(f1-f0) < 1e-14:
            raise ZeroDivisionError('Secante com inclinação insuficiente.')
        Tnovo = T1 - f1*(T1-T0)/(f1-f0)  # Diferenças finitas substituem a derivada.
        erro = abs(Tnovo-T1)
        print(f'Secante {k:02d}: T={Tnovo:.9f} K f(T)={f(Tnovo):+.4e} ΔT={erro:.3e}')
        if erro < TOL and abs(f(Tnovo)) < 1e-7:
            return Tnovo, k
        T0, T1 = T1, Tnovo  # Descarta ponto mais antigo.
    raise RuntimeError('Secante não convergiu.')

if __name__ == '__main__':
    t1, k1 = newton()
    t2, k2 = secante()
    print(f'Newton: {t1:.8f} K em {k1} iterações; Secante: {t2:.8f} K em {k2}.')
    print('Diferença de soluções (K):', abs(t1-t2))
    print('Balanço energético em Newton: perda menos potência (W) =', f(t1))
    assert abs(t1-t2) < 1e-6
