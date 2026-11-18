"""ATIVIDADE 06 - INVERSÃO POR FATORAÇÃO LU / FLEXIBILIDADE DE MOLAS.
F=K u; u=C F; C=K^-1. K em N/m; C em m/N.
LU SEM pivoteamento conforme enunciado; UMA fatoração, várias substituições.
Cada coluna j da inversa satisfaz K*c_j=e_j (força unitária no nó j).
CONSULTA: mude vetor de rigidezes, as forças aplicadas e quantidade de nós.
"""
import numpy as np
np.set_printoptions(precision=10,suppress=True)

def rigidez(molas):
    n=len(molas)-1  # Ex.: 6 molas, 5 nós móveis.
    K=np.zeros((n,n))
    for i in range(n):
        K[i,i]=molas[i]+molas[i+1]  # Molas incidentes no nó i.
        if i<n-1:
            K[i,i+1]=K[i+1,i]=-molas[i+1]  # Acoplamento entre nós adjacentes.
    return K

def decomposicao_lu(A):
    # Doolittle: L tem diagonal 1 e U é triangular superior.
    U=np.array(A,float).copy();n=len(U);L=np.eye(n)
    for k in range(n-1):
        if abs(U[k,k])<1e-14:raise ZeroDivisionError('Pivô nulo na LU sem pivoteamento.')
        for i in range(k+1,n):
            m=U[i,k]/U[k,k];L[i,k]=m  # Armazena multiplicador em L.
            for j in range(k+1,n):U[i,j]-=m*U[k,j]
            U[i,k]=0.
    if abs(U[-1,-1])<1e-14:raise ZeroDivisionError('Último pivô nulo.')
    return L,U

def resolver_lu(L,U,b):
    n=len(b);y=np.zeros(n);x=np.zeros(n)
    for i in range(n):
        # Etapa 1: Ly=b, resolver de cima para baixo.
        y[i]=(b[i]-sum(L[i,j]*y[j] for j in range(i)))/L[i,i]
    for i in range(n-1,-1,-1):
        # Etapa 2: Ux=y, resolver de baixo para cima.
        x[i]=(y[i]-sum(U[i,j]*x[j] for j in range(i+1,n)))/U[i,i]
    return x

def analisar(molas,F):
    K=rigidez(molas);L,U=decomposicao_lu(K)
    n=len(K);C=np.zeros_like(K)
    for j in range(n):
        e=np.zeros(n);e[j]=1.  # Coluna j da identidade.
        C[:,j]=resolver_lu(L,U,e)  # JÁ fatorada: não decompor K novamente.
    u=C@F
    print('\nK (N/m):\n',K,'\nL:\n',L,'\nU:\n',U,'\nC (m/N):\n',C)
    print('L@U-K máximo:',np.max(abs(L@U-K)))
    print('K@C-I máximo:',np.max(abs(K@C-np.eye(n))))
    print('Simetria C-C.T máximo:',np.max(abs(C-C.T)))
    print('Carregamento F=',F,'N -> deslocamentos u=',u,'m')
    assert np.allclose(L@U,K,atol=1e-10)
    assert np.allclose(K@C,np.eye(n),atol=1e-10)
    assert np.allclose(C,C.T,atol=1e-10)
    assert np.allclose(K@u,F,atol=1e-9)
    return K,C,u

if __name__=='__main__':
    K2,C2,u2=analisar([1000.,500.,1000.],np.array([1.,0.]))
    # C[:,0] é deslocamento sob F=[1,0]; C[:,1] sob F=[0,1].
    print('Colunas de C2: esforços unitários:',C2[:,0],C2[:,1])
    K5,C5,u5=analisar([1200.,800.,1500.,1000.,1800.,900.],
                       np.array([0.,100.,0.,-50.,25.]))
