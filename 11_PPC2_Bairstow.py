"""PPC2 - MÉTODO DE BAIRSTOW PARA POLINÔMIOS REAIS DE GRAU ARBITRÁRIO.
Raízes reais e pares complexos, polinômio sétimo de teste e polinômio físico
para sistema massa-amortecedor-mola de 2 graus de liberdade (EXEMPLO ESCOLHIDO).
Varreduras (r0,s0), iterações e fractal de bacias de atração.
ALTERAR M,C,K para o seu sistema da APC2; o PDF não fornece seus valores.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.seterr(over='ignore',invalid='ignore',divide='ignore')

def b_sintetico(coef_desc,r,s):
    # Coeficientes em ordem ascendente a0...an dentro desta função.
    a=np.asarray(coef_desc,float)[::-1]
    n=len(a)-1
    b=np.zeros(n+1);b[n]=a[n];b[n-1]=a[n-1]+r*b[n]
    for i in range(n-2,-1,-1):b[i]=a[i]+r*b[i+1]+s*b[i+2]
    return b

def par_bairstow(coef,r0,s0,tol=1e-9,max_iter=120):
    # Busca fator x²-rx-s; b0,b1 são o resto, devem zerar.
    r,s=float(r0),float(s0)
    escala=max(1.,np.max(abs(coef)))
    for it in range(1,max_iter+1):
        b=b_sintetico(coef,r,s)
        F=np.array([b[1],b[0]])
        norma=np.max(abs(F))
        if np.isfinite(norma) and norma<tol*escala:return r,s,it
        if not np.isfinite(norma):break
        # Derivadas do resto com diferenças finitas centrais; Newton 2x2 manual.
        hr=1e-6*max(1.,abs(r));hs=1e-6*max(1.,abs(s))
        rp=b_sintetico(coef,r+hr,s);rm=b_sintetico(coef,r-hr,s)
        sp=b_sintetico(coef,r,s+hs);sm=b_sintetico(coef,r,s-hs)
        J=np.column_stack((np.array([(rp[1]-rm[1]),(rp[0]-rm[0])])/(2*hr),
                           np.array([(sp[1]-sm[1]),(sp[0]-sm[0])])/(2*hs)))
        det=J[0,0]*J[1,1]-J[0,1]*J[1,0]
        if not np.isfinite(det) or abs(det)<1e-18:break
        dr=(-F[0]*J[1,1]+J[0,1]*F[1])/det
        ds=(-J[0,0]*F[1]+F[0]*J[1,0])/det
        # Backtracking evita passos que aumentem os resíduos do resto quadrático.
        melhorou=False
        for j in range(25):
            alpha=2.**(-j)
            rn,sn=r+alpha*dr,s+alpha*ds
            bn=b_sintetico(coef,rn,sn)
            if np.all(np.isfinite(bn)) and max(abs(bn[0]),abs(bn[1]))<norma:
                r,s=rn,sn;melhorou=True;break
        if not melhorou:break
    raise RuntimeError(f'Bairstow falhou com r0={r0},s0={s0}')

def raizes_bairstow(coef):
    p=np.array(coef,float);p=p/max(abs(p));res=[];detalhes=[]
    sementes=[(r,s) for r in np.linspace(-5,5,11) for s in np.linspace(-8,8,11)]
    while len(p)>3:
        achou=False
        for r0,s0 in sementes:
            try:
                r,s,it=par_bairstow(p,r0,s0)
                quoc,resto=np.polydiv(p,[1,-r,-s])  # SÓ divisão, NÃO acha raízes.
                if np.max(abs(resto))>1e-6*max(1.,max(abs(p))):continue
                disc=r*r+4*s
                par=(np.array([r+np.sqrt(disc),r-np.sqrt(disc)])/2
                     if disc>=0 else np.array([complex(r,np.sqrt(-disc)),complex(r,-np.sqrt(-disc))])/2)
                res.extend(par);p=quoc.real;detalhes.append((r,s,it));achou=True;break
            except (RuntimeError,ValueError,FloatingPointError,OverflowError,ZeroDivisionError):pass
        if not achou:raise RuntimeError('Fatoração não convergiu com as sementes escolhidas.')
    # Grau 1 ou 2: fórmula fechada (não é biblioteca de cálculo de raízes).
    if len(p)==3:
        a,b,c=p;d=complex(b*b-4*a*c)**.5
        res.extend([(-b+d)/(2*a),(-b-d)/(2*a)])
    elif len(p)==2:res.append(-p[1]/p[0])
    return np.asarray(res,dtype=complex),detalhes

def polinomio_caracteristico(M,C,K):
    # Para dois graus de liberdade: det(lambda² M + lambda C + K)=0.
    # Utiliza multiplicação de polinômios, NÃO função pronta de autovalores.
    p11=np.array([K[0,0],C[0,0],M[0,0]])
    p12=np.array([K[0,1],C[0,1],M[0,1]])
    p21=np.array([K[1,0],C[1,0],M[1,0]])
    p22=np.array([K[1,1],C[1,1],M[1,1]])
    produto1=np.polynomial.polynomial.polymul(p11,p22)
    produto2=np.polynomial.polynomial.polymul(p12,p21)
    diferenca=np.zeros(max(len(produto1),len(produto2)))
    diferenca[:len(produto1)]+=produto1
    diferenca[:len(produto2)]-=produto2
    return diferenca[::-1]  # Coeficientes da maior potência à constante.

def classificar_lambdas(raizes):
    for raiz in raizes:
        estado='estável' if raiz.real < -1e-7 else ('instável' if raiz.real>1e-7 else 'marginal')
        frequencia=abs(raiz.imag)/(2*np.pi)  # Frequência amortecida, Hz.
        print(f'lambda={raiz.real:+.8f}{raiz.imag:+.8f}j; {estado}; f={frequencia:.6f} Hz')

def mapa_bairstow(coef,grid=np.linspace(-4,4,45)):
    # Uma bacia corresponde a um fator convergido (r,s), não apenas à iteração.
    mapa=np.full((len(grid),len(grid)),-1,dtype=int)
    tempo=np.full_like(mapa,np.nan,dtype=float)
    centros=[]
    for i,s0 in enumerate(grid):
        for j,r0 in enumerate(grid):
            try:
                r,s,it=par_bairstow(coef,r0,s0,max_iter=70)
                c=np.array([r,s])
                indice=next((k for k,v in enumerate(centros) if np.linalg.norm(c-v)<1e-3),None)
                if indice is None:centros.append(c);indice=len(centros)-1
                mapa[i,j]=indice;tempo[i,j]=it
            except (RuntimeError,ValueError,OverflowError,FloatingPointError,ZeroDivisionError):pass
    plt.figure();plt.imshow(mapa,origin='lower',extent=[grid[0],grid[-1],grid[0],grid[-1]],
                             interpolation='nearest',aspect='auto')
    plt.colorbar(label='Bacia de convergência (-1 = falha)')
    plt.xlabel('r0');plt.ylabel('s0');plt.title('Bacias de Bairstow')
    plt.tight_layout();plt.savefig('11_PPC2_fractal_bacias.png',dpi=145);plt.close()
    plt.figure();plt.imshow(tempo,origin='lower',extent=[grid[0],grid[-1],grid[0],grid[-1]],aspect='auto')
    plt.colorbar(label='Iterações');plt.xlabel('r0');plt.ylabel('s0')
    plt.title('Velocidade de convergência');plt.tight_layout()
    plt.savefig('11_PPC2_iteracoes.png',dpi=145);plt.close()
    return centros,mapa

if __name__=='__main__':
    raizes_referencia=np.array([-3.,-1.,.5,2.,4.,1.+2j,1.-2j])
    pol7=np.poly(raizes_referencia).real
    r,det=raizes_bairstow(pol7)
    err=max(min(abs(x-y) for y in raizes_referencia) for x in r)
    print('Polinômio grau 7:',pol7,'\nRaízes por Bairstow:',r,'\nErro máximo:',err,'\nFatores:',det)
    assert len(r)==7 and err<1e-5

    # Valores EXEMPLO: definir com seu modelo massa-mola-amortecedor APC2.
    M=np.diag([2.,1.5])
    C=np.array([[2.0,-.5],[-.5,1.5]])
    K=np.array([[180.,-80.],[-80.,140.]])
    pol=polinomio_caracteristico(M,C,K)
    lambdas,fatores=raizes_bairstow(pol)
    print('Polinômio característico exemplo:',pol,'\nAutovalores:',lambdas)
    classificar_lambdas(lambdas)
    print('Checagem independente NumPy:',np.roots(pol))  # VALIDAÇÃO, não algoritmo.
    assert max(abs(np.polyval(pol,x)) for x in lambdas)<1e-4
    print('Variação dos chutes iniciais no polinômio de grau 7:')
    for r0,s0 in [(-3.,-3.),(-1.,1.),(0.,0.),(2.,2.),(4.,-2.),(6.,6.)]:
        try:
            r,s,it=par_bairstow(pol7,r0,s0)
            print(f'r0={r0}, s0={s0} -> r={r:.6f}, s={s:.6f}, it={it}')
        except RuntimeError:print(f'r0={r0}, s0={s0} -> não convergiu')
    centros,mapa=mapa_bairstow(pol7)
    print('Quantidade de fatores distintos encontrados na varredura:',len(centros))
