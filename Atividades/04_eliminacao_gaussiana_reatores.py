"""ATIVIDADE 04 - ELIMINAÇÃO GAUSSIANA INGÊNUA / CINCO REATORES.
Montamos A*c=b com balanços de massa; c=[c1,c2,c3,c4,c5].
Não usar pivoteamento NESTA atividade: o objetivo é aprender a eliminação ingênua.
A[i,k] <- A[i,k]-m*A[k,k]; b[i] <- b[i]-m*b[k].
CONSULTA: para novo sistema, alterar SOMENTE os balanços (A,b).
"""
import numpy as np
# Vazões do fluxograma: Qij: do reator i para o reator j.
Q01,c01=5.,10.
Q03,c03=8.,20.
Q12,Q15,Q23,Q24,Q25=3.,3.,1.,1.,1.
Q31,Q34,Q54,Q55,Q44=1.,8.,2.,2.,11.
A=np.array([[Q12+Q15,0,-Q31,0,0],
            [-Q12,Q23+Q24+Q25,0,0,0],
            [0,-Q23,Q31+Q34,0,0],
            [0,-Q24,-Q34,Q44,-Q54],
            [-Q15,-Q25,0,0,Q54+Q55]],float)
b=np.array([Q01*c01,0,Q03*c03,0,0],float)

def gauss_ingenuo(matriz,fonte,exibir=True):
    U=np.array(matriz,float).copy()
    d=np.array(fonte,float).copy()
    n=len(d);mult_div=0; soma_sub=0
    for k in range(n-1):  # k: coluna do pivô.
        if abs(U[k,k])<1e-14: raise ZeroDivisionError('Pivô nulo: ingênuo não funciona.')
        for i in range(k+1,n): # i: linhas abaixo do pivô.
            m=U[i,k]/U[k,k]; mult_div+=1  # Multiplicador para zerar a coluna k.
            for j in range(k+1,n):
                U[i,j]=U[i,j]-m*U[k,j]; mult_div+=1;soma_sub+=1
            d[i]=d[i]-m*d[k];mult_div+=1;soma_sub+=1
            U[i,k]=0.0
        if exibir: print(f'Após pivô k={k+1}:\n{U}\nvetor b modificado={d}\n')
    x=np.zeros(n)
    for i in range(n-1,-1,-1): # Substituição regressiva, de baixo para cima.
        s=d[i]
        for j in range(i+1,n):
            s-=U[i,j]*x[j];mult_div+=1;soma_sub+=1
        x[i]=s/U[i,i];mult_div+=1
    return x,U,(mult_div,soma_sub)

if __name__ == '__main__':
    print('Matriz e fontes originais:\n',A,'\n',b)
    c,U,flops=gauss_ingenuo(A,b)
    for i,valor in enumerate(c,1):print(f'c{i}={valor:.9f}')
    print('Matriz triangular:',U)
    print('FLOPs mult/div e soma/sub:',flops,'total:',sum(flops))
    print('Resíduo ||Ac-b||inf:',np.linalg.norm(A@c-b,np.inf))
    print('Conferência independente np.linalg.solve:',np.linalg.solve(A,b))
    assert np.allclose(A@c,b,atol=1e-10)
