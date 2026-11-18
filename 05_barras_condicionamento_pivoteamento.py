"""ATIVIDADE 05 - BARRAS, CONDICIONAMENTO, GAUSS INGÊNUO E PIVOTADO.
Matriz A: equações de equilíbrio dos nós; força positiva=tração.
Parte 1: determinante e sensibilidade para três pares de ângulos.
Parte 2: sistema 5x5, cenários A e C e esforço Fref conhecido.
Parte 3: eliminação SEM e COM pivoteamento; trocas, resíduos e FLOPs.
Parte 4: perturbação apenas de b[1] (+0,01 N), e precisão reduzida.
CONSULTA: alterar ângulos, phi, Fref, carga e critério de pivô.
"""
import numpy as np
np.set_printoptions(precision=9,suppress=True)
FREF=np.array([100.,-80.,20.,35.,15.])  # Referência APENAS para verificar.
ANGULOS={'A':(15.,65.),'B':(10.,10.1),'C':(.01,.02)}
PHI=np.deg2rad(45.)

def matriz_duas(t1,t2):
    t1,t2=np.deg2rad([t1,t2])  # Python calcula seno/cosseno em RADIANOS.
    return np.array([[np.sin(t1),np.sin(t2)],
                     [np.cos(t1),np.cos(t2)]])

def matriz_cinco(t1,t2):
    t1,t2=np.deg2rad([t1,t2]);p=PHI
    # As cinco linhas são as equações de equilíbrio descritas no PDF.
    return np.array([[np.sin(t1),np.sin(t2),1,0,0],
                     [np.cos(t1),np.cos(t2),0,0,0],
                     [0,0,-1,np.cos(p),0],
                     [0,0,0,-np.sin(p),-1],
                     [0,0,0,0,1]],float)

def gauss(A,b,pivotear=False,imprimir=False,sig=None):
    """Eliminação densa; FLOPs excluem comparações/trocas; sig=4 arredonda CADA operação."""
    M=np.asarray(A,float).copy();y=np.asarray(b,float).copy();n=len(y)
    escala=max(1.,np.max(abs(M)))
    limite=1e-14*escala  # Limiar relativo à escala da matriz.
    if sig is not None:
        def arredondar(x):
            if x==0:return 0.
            return float(f'{x:.{sig}g}')
        M=np.vectorize(arredondar)(M);y=np.vectorize(arredondar)(y)
    else:
        def arredondar(x):return x
    trocas=[];mult_div=0;soma_sub=0
    for k in range(n-1):
        p=k+np.argmax(abs(M[k:,k])) if pivotear else k
        if abs(M[p,k])<=limite:
            raise ZeroDivisionError(f'Pivô insuficiente na coluna {k+1}.')
        if p!=k:
            M[[k,p],:]=M[[p,k],:];y[[k,p]]=y[[p,k]]
            trocas.append((k+1,p+1)) # Índices humanos começam em 1.
        pivo=M[k,k]
        for i in range(k+1,n):
            fator=arredondar(M[i,k]/pivo);mult_div+=1
            for j in range(k+1,n):
                produto=arredondar(fator*M[k,j]);mult_div+=1
                M[i,j]=arredondar(M[i,j]-produto);soma_sub+=1
            produto=arredondar(fator*y[k]);mult_div+=1
            y[i]=arredondar(y[i]-produto);soma_sub+=1
            M[i,k]=0.
        if imprimir:print(f'ETAPA {k+1}: linha pivô original na submatriz={p+1}, pivô={pivo:.9g}\n{M}\n')
    if abs(M[-1,-1])<=limite:raise ZeroDivisionError('Último pivô insuficiente.')
    x=np.zeros(n)
    for i in range(n-1,-1,-1):
        soma=y[i]
        for j in range(i+1,n):
            produto=arredondar(M[i,j]*x[j]);mult_div+=1
            soma=arredondar(soma-produto);soma_sub+=1
        x[i]=arredondar(soma/M[i,i]);mult_div+=1
    return x,trocas,(mult_div,soma_sub)

if __name__=='__main__':
    print('PARTE 1: SISTEMAS DE DUAS BARRAS')
    for nome,(t1,t2) in ANGULOS.items():
        A2=matriz_duas(t1,t2)
        cargas=-A2@FREF[:2]  # A*F=-p, então p=-A*Fref.
        Fb,_,_=gauss(A2,-cargas,pivotear=True)
        p_pert=cargas.copy();p_pert[1]+=.01
        Fp,_,_=gauss(A2,-p_pert,pivotear=True)
        print(nome,'|det A|=',abs(np.linalg.det(A2)),
              '|sin(t1-t2)|=',abs(np.sin(np.deg2rad(t1-t2))),
              'cond2=',np.linalg.cond(A2), 'F=',Fb,'DeltaF=',Fp-Fb)
    print('\nPARTE 2/3/4: SISTEMA 5x5')
    for nome in ('A','C'):
        A=matriz_cinco(*ANGULOS[nome]);b=A@FREF
        print('\n=== CENÁRIO',nome,'===')
        for pivotear in (False,True):
            rotulo='PIVOTADO' if pivotear else 'INGÊNUO'
            try:
                x,trocas,flops=gauss(A,b,pivotear,imprimir=pivotear)
                print(rotulo,'F=',x,'trocas=',trocas,'erro_ref=',np.max(abs(x-FREF)),
                      'resíduo_inf=',np.linalg.norm(A@x-b,np.inf),
                      'FLOPs (mult/div,soma/sub)=',flops,'total=',sum(flops))
                assert np.max(abs(x-FREF))<1e-7
            except ZeroDivisionError as e: print(rotulo,'FALHOU:',str(e))
        bnovo=b.copy();bnovo[1]+=.01
        f0,_,_=gauss(A,b,True);f1,_,_=gauss(A,bnovo,True)
        print('PERTURBAÇÃO: ||Delta b||inf=',np.max(abs(bnovo-b)),
              '||Delta F||inf=',np.max(abs(f1-f0)),
              'cond_inf(A)=',np.linalg.cond(A,np.inf))
        if nome=='C':
            for piv in (False,True):
                try:
                    xr,_,_=gauss(A,b,piv,sig=4)
                    print('4 ALGARISMOS SIGNIFICATIVOS',piv,
                          'F=',xr,'erro_ref=',np.max(abs(xr-FREF)),
                          'resíduo verdadeiro=',np.linalg.norm(A@xr-b,np.inf))
                except ZeroDivisionError as e:print('PRECISÃO 4 DÍGITOS FALHOU:',e)
    # Condicionamento mede SENSIBILIDADE física. Pivoteamento melhora estabilidade
    # da solução computada, mas NÃO modifica o condicionamento da matriz original.
