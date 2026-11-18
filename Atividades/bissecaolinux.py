"""Oficina I - Método da bisseção (Python 3, sem bibliotecas externas).

Problema: encontrar a raiz de f(x) = x² - 2 no intervalo [1, 2].
Para o desafio 8B, troque apenas a função f conforme indicado abaixo.
Executar no Linux: python3 bissecao.py
"""


def f(x):
    """Função cujo zero queremos aproximar."""
    return x * x - 2.0

    # Desafio 8B: substitua a linha acima por:
    # return x * x * x - x - 2.0


def bissecao(a, b, tol, max_iter=100):
    """Aproxima uma raiz por bisseção e devolve (raiz, número de iterações).

    A raiz deve estar inicialmente delimitada por valores de sinais opostos
    (ou coincidir com um dos extremos). O erro absoluto da aproximação
    final é limitado pela metade do comprimento do intervalo final.
    """
    if a >= b:
        raise ValueError("O limite inferior deve ser menor que o superior.")
    if tol <= 0:
        raise ValueError("A tolerância precisa ser positiva.")

    fa = f(a)
    fb = f(b)

    # Uma raiz pode coincidir com a extremidade do intervalo.
    if fa == 0.0:
        return a, 0
    if fb == 0.0:
        return b, 0
    if fa * fb > 0.0:
        raise ValueError("A função deve mudar de sinal no intervalo escolhido.")

    print("\n it |          a |          b |          m |         f(m)")
    print("-" * 69)

    k = 0
    # A metade do intervalo fornece um limite de erro para o ponto médio.
    while (b - a) / 2.0 > tol and k < max_iter:
        m = (a + b) / 2.0
        fm = f(m)
        k += 1
        print(f"{k:3d} | {a:10.6f} | {b:10.6f} | {m:10.6f} | {fm:12.5e}")

        if fm == 0.0:
            # Encontramos uma raiz exata na aritmética da máquina.
            return m, k
        elif fa * fm < 0.0:
            b = m  # Raiz permanece na metade esquerda.
        else:
            a = m  # Raiz permanece na metade direita.
            fa = fm

    # Como o roteiro pede, devolvemos o ponto médio do intervalo atualizado.
    m = (a + b) / 2.0
    if (b - a) / 2.0 > tol:
        print("Aviso: foi atingido o limite de iterações antes da tolerância.")
    return m, k


def main():
    print("Método da bisseção: f(x) = x² - 2")
    try:
        a = float(input("Limite inferior: "))
        b = float(input("Limite superior: "))
        tol = float(input("Tolerância: "))
        raiz, iteracoes = bissecao(a, b, tol)
    except ValueError as erro:
        print("Entrada inválida:", erro)
        return

    print("\nRESULTADO")
    print(f"Raiz aproximada: {raiz:.10f}")
    print(f"f(raiz): {f(raiz):.6e}")
    print(f"Iterações: {iteracoes}")


if __name__ == "__main__":
    main()
