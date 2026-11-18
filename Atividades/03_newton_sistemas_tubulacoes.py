"""ATIVIDADE 03 - NEWTON-RAPHSON PARA 2 TUBOS EM PARALELO.
Incógnitas Q1,Q2; equações F1=Q1+Q2-Q=0 e F2=hL1-hL2=0.
A Jacobiana é a matriz 2x2 de derivadas parciais dFi/dQj.
Como solicitado no PDF, resolvemos J*delta=-F com a REGRA DE CRAMER.
CONSULTA: atualizar os parâmetros Q,g,L1,D1,f1,K1,L2,D2,f2,K2,Kv.
"""
import numpy as np
Q,g = .050, 9.81
L1,D1,f1,K1 = 30.,.10,.020,1.5
L2,D2,f2,K2,Kv = 20.,.08,.022,1.,8.
tol, max_iter = 1e-10, 100

# U=4Q/(pi D²). Substituir em hL=(f L/D + K)*U²/(2g) dá hL=a*Q².
a1 = (f1*L1/D1 + K1)*8/(g*np.pi**2*D1**4)
a2 = (f2*L2/D2 + K2+Kv)*8/(g*np.pi**2*D2**4)

def F(q1,q2):
    return np.array([q1+q2-Q, a1*q1**2-a2*q2**2],float)

def jacobiana(q1,q2):
    # Linha 1: dF1/dQ1=1 e dF1/dQ2=1.
    # Linha 2: dF2/dQ1=2a1 Q1 e dF2/dQ2=-2a2 Q2.
    return np.array([[1.,1.],[2*a1*q1,-2*a2*q2]])

def cramer_2x2(J, b):
    # Determinantes de Cramer, sem recorrer a np.linalg.solve.
    determinante = J[0,0]*J[1,1] - J[0,1]*J[1,0]
    if abs(determinante) < 1e-14:
        raise ZeroDivisionError('Jacobiana singular: revise o chute inicial.')
    delta1 = (b[0]*J[1,1]-J[0,1]*b[1])/determinante
    delta2 = (J[0,0]*b[1]-b[0]*J[1,0])/determinante
    return np.array([delta1,delta2])

def resolver(q1_inicial=.025,q2_inicial=.025):
    x = np.array([q1_inicial,q2_inicial],float)
    for k in range(1,max_iter+1):
        jac = jacobiana(*x)
        # Linearização de Newton: J(xk)*delta = -F(xk).
        delta = cramer_2x2(jac,-F(*x))
        x = x+delta
        erro = max(abs(delta))
        residuo = max(abs(F(*x)))
        print(f'{k:02d} Q1={x[0]:.10f} Q2={x[1]:.10f} '
              f'delta={delta} erro={erro:.3e} ||F||inf={residuo:.3e}')
        if erro<tol and residuo<tol:
            return x,k
    raise RuntimeError('Não convergiu.')

if __name__ == '__main__':
    (q1,q2),iteracoes=resolver()
    U1,U2=4*q1/(np.pi*D1**2),4*q2/(np.pi*D2**2)
    print('Vazões (m³/s):',q1,q2,'total:',q1+q2)
    print('Velocidades (m/s):',U1,U2)
    print('Perdas de carga (m):',a1*q1*q1,a2*q2*q2)
    print('Conferência Newton versus biblioteca:', np.linalg.norm(F(q1,q2),ord=np.inf))
    assert min(q1,q2)>0 and abs(q1+q2-Q)<1e-9 and abs(a1*q1*q1-a2*q2*q2)<1e-9
