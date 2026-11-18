"""ATIVIDADE 08 - MÉTODO DE MÜLLER / ESCOAMENTO CRÍTICO TRAPEZOIDAL.
A(H)=bH+zH², T(H)=b+2zH, condição Fr²=Q² T/(g A³)=1.
f(H)=g A(H)³-Q²T(H)=0 (polinômio de grau 6).
Müller aproxima f por parábola nos últimos TRÊS pontos (pode dar complexos).
Somente H real e positivo tem interpretação física.
CONSULTA: alterar b,z,g,Q, três estimativas e tol.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
b,z,g=4.,1.5,9.81
TOL,MAX_ITER=1e-8,100

def area(H):return b*H+z*H**2

def largura_livre(H):return b+2*z*H

def funcao(H,Q):
    return g*area(H)**3-Q**2*largura_livre(H)

def coeficientes(Q):
    # Ordenação: [H^6,H^5,H^4,H^3,H^2,H^1,H^0].
    return np.array([g*z**3,3*g*b*z**2,3*g*b*b*z,g*b**3,0,-2*Q**2*z,-Q**2*b])

def muller(Q,x0=.5,x1=1.0,x2=1.5,exibir=True):
    x0,x1,x2=complex(x0),complex(x1),complex(x2)
    for it in range(1,MAX_ITER+1):
        h0=x1-x0;h1=x2-x1
        if abs(h0*h1*(h0+h1))<1e-20:raise ZeroDivisionError('Pontos coincidentes.')
        d0=(funcao(x1,Q)-funcao(x0,Q))/h0
        d1=(funcao(x2,Q)-funcao(x1,Q))/h1
        a=(d1-d0)/(h1+h0)  # Coeficiente quadrático da parábola.
        bm=a*h1+d1         # Coeficiente linear deslocado para x2.
        c=funcao(x2,Q)    # Valor de f no último ponto.
        D=np.lib.scimath.sqrt(bm*bm-4*a*c)
        E=bm+D if abs(bm+D)>abs(bm-D) else bm-D  # Evita cancelamento.
        if abs(E)<1e-18:raise ZeroDivisionError('Denominador de Müller nulo.')
        x3=x2-2*c/E
        erro=abs(x3-x2)
        if exibir:print(f'{it:02d} H0={x0.real:.8f} H1={x1.real:.8f} '
                        f'H2={x2.real:.8f} H3={x3.real:.10f} '
                        f'f(H3)={funcao(x3,Q).real:+.3e} erro={erro:.3e}')
        if erro<TOL and abs(funcao(x3,Q))<1e-5:
            if abs(x3.imag)>1e-7 or x3.real<=0:
                raise ValueError('Raiz complexa ou não positiva: não física.')
            return float(x3.real),it
        x0,x1,x2=x1,x2,x3
    raise RuntimeError('Müller não convergiu.')

def froude(H,Q):
    return np.sqrt(Q**2*largura_livre(H)/(g*area(H)**3))

if __name__=='__main__':
    print('Coeficientes para Q=20:',coeficientes(20.))
    print('Referência PDF: [33.10875,264.87,706.32,627.84,0,-1200,-1600]')
    H20,it=muller(20.)
    print(f'Q=20 -> Hc={H20:.10f} m; Fr={froude(H20,20.):.12f}')
    # Sensibilidade a diferentes três estimativas iniciais.
    for pontos in ((.2,.8,1.4),(.4,.9,1.6),(.6,1.1,1.7)):
        H,_=muller(20.,*pontos,exibir=False)
        print('Teste três chutes:',pontos,'Hc=',H)
    # Varredura COMPLETA solicitada, salva 2 colunas Q,Hc.
    vazoes=np.array([5.,10.,15.,20.,25.,30.,35.,40.])
    profundidades=[]
    for Q in vazoes:
        H,it=muller(Q,exibir=False)
        profundidades.append(H)
        print(f'Q={Q:5.1f} m³/s -> Hc={H:.9f} m; Fr={froude(H,Q):.10f}; iter={it}')
        assert abs(froude(H,Q)-1)<1e-7
    np.savetxt('08_tabela_vazao_profundidade.csv',np.column_stack([vazoes,profundidades]),
               delimiter=',',header='Q_m3_s,Hc_m',comments='')
    plt.plot(vazoes,profundidades,'o-');plt.grid();plt.xlabel('Vazão Q (m³/s)')
    plt.ylabel('Profundidade crítica Hc (m)');plt.tight_layout()
    plt.savefig('08_Hc_vs_Q.png',dpi=150)
    plt.close()
