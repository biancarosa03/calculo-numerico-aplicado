"""PPC3 - CONDUÇÃO TRANSIENTE 1D EM SÓLIDO COM GERAÇÃO INTERNA.
Diferenças finitas implícitas, solução tridiagonal pelo algoritmo de Thomas.
x=0: simetria/adiabático; x=L: convecção h(T-T∞).
NÃO há parâmetros materiais numéricos fixados no enunciado do PPC:
os valores abaixo são exemplo ilustrativo, NÃO dados fornecidos pelo professor.
Inclui validação por solução de série para qger=0, refinamentos, com qger e gráficos.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# PARÂMETROS EXEMPLIFICATIVOS; substituir por cenário da atividade/projeto.
L=.01              # Meia-espessura (m), centro em x=0 e face em x=L.
k=15.              # Condutividade (W/(m K)).
rho=7800.          # Densidade (kg/m³).
Cp=500.            # Calor específico (J/(kg K)).
h=120.             # Convecção (W/(m² K)).
Ti=350.            # Temperatura inicial uniforme (K).
Tinf=300.          # Temperatura ambiente (K).
qger=1e6           # Geração volumétrica demonstrativa (W/m³).
alpha=k/(rho*Cp)

def thomas(a,b,c,d):
    """Tridiagonal: a tem N-1 subdiagonais; b N diagonais; c N-1 superdiagonais."""
    a=np.asarray(a,float).copy();b=np.asarray(b,float).copy()
    c=np.asarray(c,float).copy();d=np.asarray(d,float).copy();n=len(b)
    if n<2:raise ValueError('Malha deve ter pelo menos dois nós.')
    for i in range(1,n):
        if abs(b[i-1])<1e-15:raise ZeroDivisionError('Pivô nulo no Thomas.')
        m=a[i-1]/b[i-1]
        b[i]-=m*c[i-1]
        d[i]-=m*d[i-1]
    if abs(b[-1])<1e-15:raise ZeroDivisionError('Pivô final nulo.')
    x=np.zeros(n);x[-1]=d[-1]/b[-1]
    for i in range(n-2,-1,-1):x[i]=(d[i]-c[i]*x[i+1])/b[i]
    return x

def simular(N=31,dt=.5,tempo_final=120.,geracao=0.):
    """Backward Euler: (I-Fo*Laplaciano)Tnovo=Tantigo+qdot dt/(rho Cp)."""
    dx=L/(N-1);Fo=alpha*dt/dx**2;Bi=h*dx/k
    n_passos=round(tempo_final/dt)
    if abs(n_passos*dt-tempo_final)>1e-10:raise ValueError('Tempo final deve ser múltiplo de dt.')
    T=np.full(N,Ti);historico=np.zeros((n_passos+1,N));historico[0]=T
    a=np.full(N-1,-Fo);b=np.full(N,1+2*Fo);c=np.full(N-1,-Fo)
    c[0]=-2*Fo      # x=0 adiabático -> nó fantasma T(-dx)=T(+dx).
    a[-1]=-2*Fo     # Na superfície: nó fantasma representa convecção externa.
    b[-1]=1+2*Fo+2*Fo*Bi
    for it in range(1,n_passos+1):
        rhs=T+geracao*dt/(rho*Cp)
        rhs[-1]+=2*Fo*Bi*Tinf
        T=thomas(a,b,c,rhs)
        historico[it]=T
    return np.linspace(0,L,N),np.linspace(0,tempo_final,n_passos+1),historico

def autovalor(indice,bi_global):
    """Autovalores da solução analítica: mu*tan(mu)=hL/k."""
    a=indice*np.pi+1e-12;b=indice*np.pi+np.pi/2-1e-12
    for _ in range(100):
        m=.5*(a+b)
        f=m*np.tan(m)-bi_global
        if f>0:b=m
        else:a=m
    return .5*(a+b)

def exata_sem_geracao(x,t,termos=75):
    """Série analítica para θ=(T-Tinf)/(Ti-Tinf), qger=0, T(x,0)=Ti."""
    BiG=h*L/k;Fo_t=alpha*t/L**2
    theta=np.zeros_like(x,dtype=float)
    for j in range(termos):
        mu=autovalor(j,BiG)
        Aj=4*np.sin(mu)/(2*mu+np.sin(2*mu))
        theta+=Aj*np.cos(mu*x/L)*np.exp(-mu*mu*Fo_t)
    return Tinf+(Ti-Tinf)*theta

def validar_thomas():
    # Checagem independente numérica com matriz tridiagonal 4x4 conhecida.
    A=np.array([[4.,-1,0,0],[-2,5,-2,0],[0,-1,3,-1],[0,0,-1,2]])
    rhs=np.array([3.,0.,7.,1.])
    x=thomas(np.diag(A,-1),np.diag(A),np.diag(A,1),rhs)
    print('Thomas x=',x,'; erro ||Ax-b||inf=',max(abs(A@x-rhs)))
    assert np.allclose(A@x,rhs,atol=1e-12)

def comparar(N,dt,tempo=120.):
    x,t,T=simular(N,dt,tempo_final=tempo,geracao=0.)
    Tanalitica=exata_sem_geracao(x,tempo)
    erro_max=np.max(abs(T[-1]-Tanalitica))
    print(f'N={N:4d}, dt={dt:7.4f} s, dx={x[1]-x[0]:.6g} m, '
          f'erro final máximo sem geração={erro_max:.7f} K')
    return erro_max

if __name__=='__main__':
    validar_thomas()
    # Convergência temporal e espacial: reduzir dt e dx DEVE reduzir erro numérico.
    for N in (11,21,41):
        for dt in (2.,1.,.5):comparar(N,dt)
    x,t,T0=simular(N=41,dt=.5,geracao=0.)
    _,_,Tg=simular(N=41,dt=.5,geracao=qger)
    for perfil,rotulo in ((T0,'sem_geracao'),(Tg,'com_geracao')):
        plt.figure()
        for instante in (0,30,60,120):
            it=int(round(instante/.5))
            plt.plot(x,perfil[it],label=f't={instante} s')
        plt.xlabel('Posição x (m)');plt.ylabel('Temperatura T (K)')
        plt.title(rotulo.replace('_',' '));plt.grid();plt.legend();plt.tight_layout()
        plt.savefig(f'12_PPC3_perfis_{rotulo}.png',dpi=145);plt.close()
        plt.figure();plt.imshow(perfil,origin='lower',aspect='auto',extent=[0,L,0,120])
        plt.colorbar(label='T (K)');plt.xlabel('x (m)');plt.ylabel('Tempo (s)')
        plt.title('Campo transiente - '+rotulo);plt.tight_layout()
        plt.savefig(f'12_PPC3_campo_{rotulo}.png',dpi=145);plt.close()
    print('Erro final sem geração:',np.max(abs(T0[-1]-exata_sem_geracao(x,120.))))
    print('Temperatura no centro/superfície final COM geração:',Tg[-1,0],Tg[-1,-1])
