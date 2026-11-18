"""ATIVIDADE 09 - MINIMIZAÇÃO ECONÔMICA DE ISOLANTE CILÍNDRICO.
OBJETIVO: menor custo total C(e)=CE(e)+CI(e) para e∈[0,0.12] m.
Rcond'=ln(r2/r1)/(2πk); Rconv'=1/(2πh r2); q'=(Ti-Tinf)/(Rcond'+Rconv').
A razão áurea elimina partes do intervalo e REUTILIZA avaliações.
A interpolação quadrática constrói uma parábola por três custos e procura seu vértice.
Sem derivadas analíticas para minimizar, conforme enunciado.
CONSULTA: alterar r1,L,Ti,Tinf,k,h,t,cI,PREÇOS,tol,limites.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r1,L,Ti,Tinf,k,h=.040,20.,180.,40.,.045,12.
t,cI=30000.,9000.
E_MIN,E_MAX=0.,.12
TOL,MAX_ITER=1e-6,200
PREÇOS=[.35,.70,1.05,1.40]

def custos(e,pE=.70):
    if e<0:raise ValueError('Espessura física não pode ser negativa.')
    r2=r1+e
    Rcond=np.log(r2/r1)/(2*np.pi*k)
    Rconv=1/(2*np.pi*h*r2)
    q_linha=(Ti-Tinf)/(Rcond+Rconv)  # Taxa em W/m.
    CE=pE*L*q_linha*t/1000  # W -> kW, multiplicar por horas e R$/kWh.
    CI=cI*np.pi*L*(r2*r2-r1*r1) # Custo do volume instalado.
    return CE+CI,CE,CI,q_linha

def razao_aurea(a=E_MIN,b=E_MAX,pE=.70,tol=TOL,max_iter=MAX_ITER):
    avaliacoes=0
    def objetivo(x):
        nonlocal avaliacoes
        avaliacoes+=1
        return custos(x,pE)[0]
    tau=(np.sqrt(5)-1)/2
    x1=b-tau*(b-a);x2=a+tau*(b-a)
    f1=objetivo(x1);f2=objetivo(x2) # Duas avaliações iniciais.
    for it in range(1,max_iter+1):
        if b-a<=tol:break
        if f1>f2:
            a=x1;x1=x2;f1=f2 # Reusa a avaliação antiga de x2.
            x2=a+tau*(b-a);f2=objetivo(x2)
        else:
            b=x2;x2=x1;f2=f1
            x1=b-tau*(b-a);f1=objetivo(x1)
    else:it=max_iter
    x=(a+b)/2
    return dict(e=x,custo=objetivo(x),iteracoes=it,avaliacoes=avaliacoes,
                largura=b-a,convergiu=(b-a<=tol),protecao=0)

def interpolacao_quadratica(x1=.04,x2=.07,x3=.11,pE=.70,tol=TOL,max_iter=MAX_ITER):
    """Três pontos ordenados, valor central mínimo; fallback áureo se degenerar."""
    avaliacoes=0;protecao=0
    def objetivo(x):
        nonlocal avaliacoes
        avaliacoes+=1
        return custos(x,pE)[0]
    f1,f2,f3=objetivo(x1),objetivo(x2),objetivo(x3)
    # Pontos sugeridos pelo professor podem NÃO cercar o mínimo para todos os preços.
    if not (x1<x2<x3 and f2<f1 and f2<f3):
        # Busca uma tripla que cerque o mínimo usando amostras do intervalo físico.
        candidatos=np.linspace(E_MIN,E_MAX,61)
        valores=[objetivo(x) for x in candidatos]
        i=int(np.argmin(valores))
        if not (0<i<len(candidatos)-1):
            # Mínimo na borda: comparação física com extremos.
            aurea=razao_aurea(E_MIN,E_MAX,pE,tol,max_iter)
            extremos=[(objetivo(E_MIN),E_MIN),(objetivo(E_MAX),E_MAX),(aurea['custo'],aurea['e'])]
            c,x=min(extremos)
            return dict(e=x,custo=c,iteracoes=0,avaliacoes=avaliacoes+aurea['avaliacoes'],
                        largura=0.,convergiu=True,protecao=1)
        x1,x2,x3=candidatos[i-1:i+2]; f1,f2,f3=valores[i-1:i+2]
        protecao+=1
    for it in range(1,max_iter+1):
        if x3-x1<=tol:break
        # Diferenças divididas da parábola: p(x)=Ax²+Bx+D.
        d1=(f2-f1)/(x2-x1)
        A=((f3-f2)/(x3-x2)-d1)/(x3-x1)
        B=d1-A*(x1+x2)
        novo=-B/(2*A) if A>1e-12 else float('nan')
        separacao=min(abs(novo-x) for x in (x1,x2,x3))
        if not np.isfinite(novo) or not (x1<novo<x3) or separacao<1e-12:
            # Passo protegido de razão áurea preserva tripla com mínimo central.
            tau=(np.sqrt(5)-1)/2
            novo=x2+(1-tau)*(x3-x2) if x3-x2>x2-x1 else x2-(1-tau)*(x2-x1)
            protecao+=1
        fn=objetivo(novo)
        if novo<x2:
            if fn<f2:x3,f3=x2,f2; x2,f2=novo,fn
            else:x1,f1=novo,fn
        else:
            if fn<f2:x1,f1=x2,f2;x2,f2=novo,fn
            else:x3,f3=novo,fn
        assert x1<x2<x3 and f2<=min(f1,f3)+1e-7
    else:it=max_iter
    return dict(e=x2,custo=f2,iteracoes=it,avaliacoes=avaliacoes,
                largura=x3-x1,convergiu=(x3-x1<=tol),protecao=protecao)

if __name__=='__main__':
    for e in np.linspace(0,.12,7):
        C,CE,CI,q=custos(e)
        print(f'e={e:.2f} m, CE={CE:.2f} R$, CI={CI:.2f} R$, C={C:.2f} R$, q={q:.3f} W/m')
    print('\nCOMPARAÇÃO PARA pE=0.70:')
    for func in (razao_aurea,interpolacao_quadratica):
        saida=func(pE=.70)
        print(func.__name__,saida)
        assert saida['convergiu']
    precos=[];espessuras=[]
    print('\nESTUDO DO PREÇO DA ENERGIA:')
    for pE in PREÇOS:
        r=razao_aurea(E_MIN,E_MAX,pE)
        e=r['e']; C,CE,CI,q=custos(e,pE)
        precos.append(pE);espessuras.append(e)
        print(f'pE={pE:.2f} R$/kWh -> e={e:.8f} m, CE={CE:.2f}, CI={CI:.2f}, C={C:.2f}')
    rcritico=k/h
    print(f'Raio crítico rc=k/h={rcritico:.7f} m; raio do tubo r1={r1:.7f} m')
    print('Como r1 > rc, a inclusão inicial de isolante REDUZ a perda térmica.')
    xs=np.linspace(E_MIN,E_MAX,300)
    plt.figure();plt.plot(xs,[custos(x)[0] for x in xs]);plt.grid()
    plt.xlabel('Espessura e (m)');plt.ylabel('Custo total (R$)');plt.tight_layout()
    plt.savefig('09_custo_vs_espessura.png',dpi=150);plt.close()
    plt.figure();plt.plot(precos,espessuras,'o-');plt.grid()
    plt.xlabel('Preço da energia (R$/kWh)');plt.ylabel('Espessura ótima e (m)');plt.tight_layout()
    plt.savefig('09_espessura_vs_preco.png',dpi=150);plt.close()
