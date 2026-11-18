"""ATIVIDADE 07 - GAUSS-SEIDEL PARA PLACA QUADRADA 2D EM REGIME PERMANENTE.
Laplace: d²T/dx²+d²T/dy²=0. Malha dx=dy -> T_ij=média de quatro vizinhos.
Gauss-Seidel escreve na mesma matriz: novos valores são aproveitados na varredura.
Jacobi, diferentemente, usa cópia inteira da iteração anterior.
Bordas NÃO atualizadas: temperaturas são condições de contorno prescritas.
CONSULTA: alterar n, temperatura das bordas, tol, max_iter e L.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Salvar gráficos sem depender de janela gráfica.
import matplotlib.pyplot as plt
L,tol,max_iter=1.0,1e-6,100000

def simular(n,exibir=False):
    # n nós internos por direção -> n+2 contando bordas.
    T=np.full((n+2,n+2),20.0)
    T[0,1:-1]=100.  # Borda inferior quente; cantos ficam a 20°C.
    for it in range(1,max_iter+1):
        erro=0.
        for j in range(1,n+1):
            for i in range(1,n+1):
                antigo=T[j,i]
                T[j,i]=.25*(T[j-1,i]+T[j+1,i]+T[j,i-1]+T[j,i+1])
                erro=max(erro,abs(T[j,i]-antigo))
        if erro<tol:break
    else:raise RuntimeError('Gauss-Seidel atingiu máximo de iterações.')
    # RESÍDUO do sistema: não confundir com erro de atualização entre iterações.
    residuos=[]
    for j in range(1,n+1):
        for i in range(1,n+1):
            r=4*T[j,i]-(T[j-1,i]+T[j+1,i]+T[j,i-1]+T[j,i+1])
            residuos.append(abs(r))
    residuo=max(residuos)
    if exibir:print('Matriz incluindo bordas:\n',T)
    print(f'n={n:2d}; iterações={it:5d}; mudança_max={erro:.4e} °C; '
          f'resíduo_max={residuo:.4e} °C; centro={T[(n+1)//2,(n+1)//2]:.7f} °C')
    return T,it,erro,residuo

if __name__=='__main__':
    resultados={}
    for n in (5,10,20): # Refinar malha geralmente aumenta número de iterações.
        resultados[n]=simular(n,exibir=n==5)
        T= resultados[n][0]
        plt.figure();plt.imshow(T,origin='lower',extent=(0,L,0,L),cmap='inferno')
        plt.colorbar(label='Temperatura (°C)');plt.xlabel('x (m)');plt.ylabel('y (m)')
        plt.title(f'Gauss–Seidel: n={n} nós internos');plt.tight_layout()
        plt.savefig(f'07_placa_n{n}.png',dpi=140);plt.close()
    print('Comparação de iterações:', {n:resultados[n][1] for n in resultados})
    # Respostas rápidas: bordas fixas -> temperatura imposta; Jacobi usa somente
    # estados anteriores; malhas finas propagam correções em mais varreduras.
    assert all(res[3]<1e-4 for res in resultados.values())
