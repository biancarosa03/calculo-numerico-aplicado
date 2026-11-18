"""PPC1 - RK4 manual: sedimentação de esfera adimensional.
St dv/dt = 1-v-(3/8)Re_s v², v(0)=0. Tempo/velocidade adimensionais.
Inclui Re=0, Re>0, comparação analítica, passo de tempo e gráficos.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rk4(f,t,v0):
    v=np.zeros(len(t));v[0]=v0
    for i in range(len(t)-1):
        h=t[i+1]-t[i]
        k1=f(t[i],v[i])
        k2=f(t[i]+h/2,v[i]+h*k1/2)
        k3=f(t[i]+h/2,v[i]+h*k2/2)
        k4=f(t[i]+h,v[i]+h*k3)
        v[i+1]=v[i]+h*(k1+2*k2+2*k3+k4)/6
    return v

def derivada(t,v,St,Re):
    return (1-v-3*Re*v*v/8)/St

def exata(t,St,Re):
    if abs(Re)<1e-14:return 1-np.exp(-t/St)
    a=3*Re/8
    raiz=np.sqrt(1+4*a)
    vp=2/(1+raiz)  # Raiz positiva da equação 1-v-av²=0, forma estável.
    vm=(-1-raiz)/(2*a)
    C=vp/vm
    E=np.exp(-raiz*t/St)
    return (vp-C*E*vm)/(1-C*E)

if __name__=='__main__':
    t=np.linspace(0,10,2001)
    plt.figure()
    for St in (.2,.5,1.,2.):
        f=lambda tempo,v:derivada(tempo,v,St,0.)
        v=rk4(f,t,0.)
        erro=np.max(abs(v-exata(t,St,0.)))
        print(f'St={St}, Re=0, erro máximo vs analítica: {erro:.3e}')
        plt.plot(t,v,label=f'St={St:g}')
        assert erro<1e-5
    plt.xlabel('t*');plt.ylabel('v*');plt.title('Re_s=0 - valores de St')
    plt.grid();plt.legend();plt.tight_layout();plt.savefig('10_PPC1_St.png',dpi=145);plt.close()

    St=1.
    plt.figure()
    for Re in (0.,.1,.3,.5,1.):
        f=lambda tempo,v:derivada(tempo,v,St,Re)
        v=rk4(f,t,0.)
        err=np.max(abs(v-exata(t,St,Re)))
        print(f'St={St}, Re_s={Re}, erro máximo vs Riccati exata: {err:.3e}')
        plt.plot(t,v,label=f'Re_s={Re:g}')
        assert err<2e-6
    plt.xlabel('t*');plt.ylabel('v*');plt.title('Influência de Re_s')
    plt.grid();plt.legend();plt.tight_layout();plt.savefig('10_PPC1_Re.png',dpi=145);plt.close()

    St,Re=1.,.5
    hs=np.array([.8,.4,.2,.1,.05,.025])
    erros=[]
    for h in hs:
        t=np.linspace(0,10,int(round(10/h))+1)
        v=rk4(lambda tt,x:derivada(tt,x,St,Re),t,0.)
        erro=np.max(abs(v-exata(t,St,Re)))
        erros.append(erro)
        print(f'refinamento h={h:.6f}; erro máximo={erro:.6e}')
    plt.loglog(hs,erros,'o-');plt.xlabel('passo Δt*');plt.ylabel('erro máximo')
    plt.title('Refinamento de passo no RK4');plt.grid();plt.tight_layout()
    plt.savefig('10_PPC1_refinamento.png',dpi=145);plt.close()
    print('Ordem aproximada de RK4:',np.polyfit(np.log(hs[-4:]),np.log(erros[-4:]),1)[0])
