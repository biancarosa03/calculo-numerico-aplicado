
# Cálculo Numérico Aplicado

### Universidade de Brasília (UnB)
**Curso:** Engenharia Mecânica  
**Disciplina:** Cálculo Numérico Aplicado  
**Autora:** Bianca de Moura Rosa  

---

## Sobre o repositório

Este repositório reúne os códigos desenvolvidos em Python durante a disciplina de Cálculo Numérico Aplicado, do curso de Engenharia Mecânica da Universidade de Brasília (UnB).

As implementações têm como objetivo aplicar métodos numéricos na resolução de problemas matemáticos e de engenharia, explorando algoritmos para determinação de raízes, resolução de sistemas de equações, aproximações numéricas e otimização.

Os programas foram organizados entre atividades práticas e Programas para Casa (PPCs), permitindo acompanhar os diferentes métodos estudados ao longo da disciplina.

## Tecnologias utilizadas

- Python 3
- NumPy
- Matplotlib
- SciPy (quando aplicável)

## Atividades práticas

| Nº | Atividade | Conteúdo |
|---|---|---|
| 01 | Bisseção e Falsa Posição | Determinação de raízes de equações não lineares |
| 02 | Newton-Raphson e Secante | Métodos iterativos para determinação de raízes |
| 03 | Newton para Sistemas | Resolução de sistemas não lineares aplicados a tubulações |
| 04 | Eliminação Gaussiana | Resolução de sistemas lineares aplicados a reatores |
| 05 | Condicionamento e Pivoteamento | Análise numérica de uma estrutura de barras |
| 06 | Decomposição LU | Aplicação em problemas de flexibilidade |
| 07 | Gauss-Seidel | Distribuição de temperatura em uma placa bidimensional |
| 08 | Método de Müller | Determinação da profundidade crítica em canal trapezoidal |
| 09 | Otimização Numérica | Determinação da espessura ótima de isolamento térmico |
| 10 | Bisseção | Oficina de utilização do sistema linux |
| 09 | Otimização em Fortran | Oficina de utilização do sistema linux |

## Programas para Casa (PPCs)

| Programa | Conteúdo |
|---|---|
| PPC 01 | Método de Runge-Kutta |
| PPC 02 | Método de Bairstow |
| PPC 03 | Método de Thomas e diferenças finitas |
| PPC 04 | Otimização de aquecedores e análise térmica |

## Organização do repositório

```text
calculo-numerico-aplicado/
|
|-- Atividades/
|   |-- 01_bissecao_falsa_posicao.py
|   |-- 02_newton_secante.py
|   |-- 03_newton_sistemas_tubulacoes.py
|   |-- 04_eliminacao_gaussiana_reatores.py
|   |-- 05_barras_condicionamento_pivoteamento.py
|   |-- 06_LU_flexibilidade_molas.py
|   |-- 07_gauss_seidel_placa_2d.py
|   |-- 08_muller_canal_critico.py
|   |-- 09_otimizacao_isolamento_termico.py
|
|-- PPCs/
|   |-- 10_PPC1_Runge_Kutta.py
|   |-- 11_PPC2_Bairstow.py
|   |-- 12_PPC3_Thomas_diferencas_finitas.py
|   |-- 13_PPC4_otimizacao_aquecedores.py
|
|-- README.md
```

## Como executar os códigos

Para executar os programas, é necessário ter o Python instalado no computador.

As bibliotecas podem ser instaladas pelo terminal:

```bash
pip install numpy matplotlib scipy
```

Após baixar o repositório, acesse a pasta do programa desejado e execute o arquivo Python.

Exemplo:

```bash
python 01_bissecao_falsa_posicao.py
```

Os programas apresentam os resultados no terminal e, quando previsto, gráficos para análise do comportamento numérico.

## Objetivos de aprendizagem

- Compreender os fundamentos dos métodos numéricos.
- Implementar algoritmos utilizando Python.
- Analisar erros, precisão e critérios de convergência.
- Resolver problemas de engenharia por meio de métodos computacionais.
- Desenvolver habilidades de programação e análise numérica.

## Autoria

**Bianca de Moura Rosa**  
Graduanda em Engenharia Mecânica  
Universidade de Brasília (UnB)

---

*Repositório desenvolvido para fins acadêmicos e de estudo.*
