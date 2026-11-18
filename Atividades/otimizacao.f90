! Oficina II - Interpolacao quadratica sucessiva, em Fortran 90.
! Funcao objetivo: f(x) = x**4 - 3*x + 2.
! No Linux: gfortran -Wall -Wextra -fcheck=all otimizacao.f90 -o otimizacao
!           ./otimizacao
program otimizacao
    implicit none

    ! real*8 e aceito pelo gfortran conforme o roteiro da oficina.
    real*8 :: x1, x2, x3, f1, f2, f3
    real*8 :: num, den, xnovo, fnovo, xanterior, erro, tol
    integer :: k, maxit
    logical :: falha

    ! Os tres pontos iniciais delimitam o minimo: f2 < f1 e f2 < f3.
    x1 = 0.0d0
    x2 = 1.0d0
    x3 = 2.0d0
    f1 = funcao(x1)
    f2 = funcao(x2)
    f3 = funcao(x3)

    tol = 1.0d-6
    maxit = 50
    k = 0
    erro = 1.0d0
    xanterior = x2
    falha = .false.

    print *, 'Pontos iniciais (x, f(x)):'
    print *, x1, f1
    print *, x2, f2
    print *, x3, f3
    print *, ' iteracao        xnovo              f(xnovo)            erro'

    do while (erro > tol .and. k < maxit)
        k = k + 1

        ! Vertice da parabola que interpola os tres pontos.
        num = f1*(x2**2-x3**2) + &
              f2*(x3**2-x1**2) + &
              f3*(x1**2-x2**2)
        den = 2.0d0*(f1*(x2-x3) + &
                     f2*(x3-x1) + &
                     f3*(x1-x2))

        ! Nunca dividir por denominador numericamente degenerado.
        if (abs(den) < 1.0d-12) then
            print *, 'Interpolacao degenerada.'
            falha = .true.
            exit
        end if

        xnovo = num / den
        fnovo = funcao(xnovo)

        ! Sem intervalo valido ou deslocamento, interrompemos o algoritmo.
        if (xnovo <= x1 .or. xnovo >= x3) then
            print *, 'Vertice fora do intervalo.'
            falha = .true.
            exit
        end if
        if (xnovo == x2) then
            print *, 'Sem progresso numerico.'
            falha = .true.
            exit
        end if

        ! Atualizar o trio preservando x1 < x2 < x3 e f2 <= f1, f3.
        if (xnovo < x2) then
            if (fnovo < f2) then
                x3 = x2
                f3 = f2
                x2 = xnovo
                f2 = fnovo
            else
                x1 = xnovo
                f1 = fnovo
            end if
        else
            if (fnovo < f2) then
                x1 = x2
                f1 = f2
                x2 = xnovo
                f2 = fnovo
            else
                x3 = xnovo
                f3 = fnovo
            end if
        end if

        ! Erro relativo aproximado definido no roteiro.
        erro = abs(xnovo-xanterior)/max(1.0d0, abs(xnovo))
        print '(I5,3(2X,ES18.10))', k, xnovo, fnovo, erro
        xanterior = xnovo
    end do

    print *
    print *, 'RESUMO'
    print *, 'Melhor x armazenado: ', x2
    print *, 'f(x) correspondente: ', f2
    print *, 'Iteracoes realizadas: ', k
    print *, 'Erro final: ', erro

    if (falha) then
        print *, 'Terminou por falha numerica; nao declarar convergencia.'
    else if (k >= maxit .and. erro > tol) then
        print *, 'Limite de iteracoes atingido.'
    else
        print *, 'Criterio de parada por tolerancia atendido.'
    end if

contains

    real*8 function funcao(x)
        implicit none
        real*8, intent(in) :: x
        funcao = x**4 - 3.0d0*x + 2.0d0
    end function funcao
end program otimizacao
