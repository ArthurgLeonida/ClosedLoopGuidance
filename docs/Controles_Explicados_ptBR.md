# Os controles usados no CFG-Ctrl e neste repositório

Guia em português para quem tem formação em controle e quer mapear cada peça do
repositório para o conceito clássico correspondente, usando a notação do artigo

> **CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance**
> Wang, Liu, Chi, Liu, Xue, Duan. CVPR 2026. arXiv:2603.03281

Cada seção diz **o que é**, **como funciona**, **se o artigo usa** e **onde está
no código**. Documentos relacionados: [diagnóstico e melhorias
priorizadas](Chattering_Fixes.md) e [protocolo dos benchmarks](Benchmark_Protocol.md).

---

## 1. Notação

| símbolo | significado | onde aparece |
|---|---|---|
| $x_t$ | estado latente no instante $t$ do fluxo | Eq. 8 |
| $v_\theta(x_t,t,c)$ | campo de velocidade **condicional** (com o prompt $c$) | Eq. 1 |
| $v_\theta(x_t,t,\varnothing)$ | campo de velocidade **incondicional** | Eq. 1 |
| $e(t)$ | **erro semântico**, $v_\theta(x_t,t,c)-v_\theta(x_t,t,\varnothing)$ | Eq. 6 |
| $w$ | escala de guidance (o ganho proporcional) | Eq. 1 |
| $\hat v$ | velocidade efetivamente aplicada ao integrador | Alg. 1 |
| $u_t$ | entrada de controle, $u_t=K_t\Pi_t\big(e(t)\big)$ | Eq. 10 |
| $K_t$ | agenda de ganho (escalar ou matriz) | Eq. 10 |
| $\Pi_t$ | operador de direção | Eq. 10 |
| $s(t)$ | **superfície deslizante**, $\dot e(t)+\lambda e(t)$ | Eq. 19 |
| $\lambda$ | parâmetro de forma da superfície | Eq. 19 |
| $\Delta e(t)$ | correção chaveada, $-K\operatorname{sign}(s(t))$, $K=k\mathbf I$ | Eq. 25 |
| $V(\mathbf s)$ | função de Lyapunov, $\tfrac12\lVert \mathbf s\rVert^2$ | Eq. 20 |
| $\Phi,\ \Gamma,\ \delta,\ \rho$ | deriva intrínseca, ganho efetivo e seus limitantes | Eqs. 36–39 |

Valores do artigo (suplemento §7.3): $\lambda=6$ nos três modelos; $k=0{,}1$
para SD3.5 e Qwen-Image, $k=0{,}7$ para Flux-dev.

No código discreto usamos índice $n$ para o passo do amostrador, $m_n$ para a
**memória armazenada** e $\delta_n$ para a correção aplicada.

---

## 2. A planta

Modelos de *flow matching* (SD3.5, Flux, Qwen-Image) aprendem um campo de
velocidade tal que a EDO

$$\frac{dx}{d\sigma}=v_\theta(x,\sigma,c),\qquad x_{\sigma=1}\sim\mathcal N(0,\mathbf I)$$

integrada de $\sigma=1$ (ruído puro) até $\sigma=0$ produz uma amostra de
$p(x\mid c)$. Na prática usam-se 20–50 passos de Euler.

> **Revisão: planta, estado, entrada, medição.** A *planta* é o que você quer
> influenciar; o *estado* é o que basta saber para prever o futuro; a *entrada*
> é o que você altera; a *medição* é o que você observa. Aqui a planta é o
> integrador da EDO em torno da rede neural; o estado é o latente $x_\sigma$; a
> entrada é uma velocidade aditiva $u_t$; e a medição é $e(t)$, que a rede
> fornece de graça a cada passo, porque ela já é avaliada duas vezes.
>
> A peculiaridade: **a rede é simultaneamente a dinâmica da planta e o sensor.**

O artigo escreve isso como o sistema afim no controle $\dot x=v_\theta(x,t)+G\,u_t$
com $G=\mathbf I$ (Eqs. 8–9): você pode somar qualquer velocidade.

**Existe realimentação.** A correção altera $x$, e $x$ altera a medição
seguinte. Uma versão anterior deste repositório afirmava que o ganho de malha
era desprezível a ponto de o sistema ser efetivamente em malha aberta; essa
afirmação era forte demais e foi retirada. O que se pode dizer é que a
transferência da prova contínua para a recorrência discreta precisa de
justificativa adicional (§7.4 abaixo).

---

## 3. Controle proporcional (P) — o próprio CFG

**No artigo: sim.** É a releitura central do artigo.

O *classifier-free guidance* aplica

$$\hat v=v_\theta(x_t,t,\varnothing)+w\,\big(v_\theta(x_t,t,c)-v_\theta(x_t,t,\varnothing)\big)
       =v_\varnothing+w\,e,\qquad w\ge1 .$$

Na forma $u_t=K_t\Pi_t(e(t))$, o CFG é $K_t=w$ e $\Pi_t=\mathbf I$: um
**controlador proporcional de ganho fixo** atuando sobre $e$ (Eqs. 11–13).

> **Revisão: controle P e o custo do ganho alto.** Um controlador proporcional
> aplica $u=Ke$. Aumentar $K$ dá resposta mais rápida e erro de regime menor,
> mas cobra dois preços: (i) tudo que estiver no sinal de erro é amplificado por
> $K$, inclusive ruído de medição e erro de modelo; (ii) com qualquer atraso na
> malha, ganho alto gera sobressinal e depois oscilação. Em imagens o sobressinal
> aparece como **supersaturação**: cores além do plausível, texturas
> excessivamente realçadas, estrutura deformada. A Fig. 1 (esquerda) do artigo é
> exatamente o retrato de fase de uma malha P subamortecida.

No código: `presets.cfg_baseline()`, que é `SMCConfig(k=0.0)`. Com $k=0$ a
correção é identicamente nula e o resultado é o CFG puro, bit a bit. Isso é
proposital: **a linha de base é um caso particular do método**, então qualquer
diferença medida é atribuível à correção e a mais nada.

---

## 4. O arcabouço unificador do artigo

O artigo propõe escrever qualquer variante de guidance como

$$u_t=K_t\,\Pi_t\big(e(t)\big),$$

separando **quanto** empurrar ($K_t$, a agenda de ganho) de **em que direção**
empurrar ($\Pi_t$, o operador de direção). A Tabela 1 do artigo classifica os
métodos existentes nesse molde. As três famílias abaixo **são citadas pelo
artigo, mas não são a contribuição dele**, e **não estão implementadas neste
repositório**.

### 4.1 Escalonamento de ganho (*gain scheduling*)

Trocar $w$ constante por $w(t)$ crescente ao longo da remoção de ruído.
Corresponde a $K_t=w(t)$, $\Pi_t=\mathbf I$ (Eq. 14).

> **Revisão: escalonamento de ganho.** Escolher o ganho como função conhecida de
> uma variável de operação — aqui o nível de ruído $t$. É adaptação em **malha
> aberta**: a agenda não olha para o desempenho da malha, apenas para onde o
> sistema está operando. Motivação aqui: no início da amostragem o estado é
> dominado por ruído e um ganho grande amplifica ruído em vez de semântica.

### 4.2 Realimentação por projeção

**APG** decompõe a direção de guidance em componentes paralela e ortogonal a
$v_\theta(x_t,t,c)$ e atenua a paralela (Eqs. 15–17). **CFG-Zero\*** projeta
sobre $v_\theta(x_t,t,\varnothing)$ e reescala (Eqs. 29–31). Em linguagem de
controle: $\Pi_t$ deixa de ser a identidade e passa a ser uma projeção
ortogonal, isto é, o controlador escolhe *como* a correção se distribui entre
direções, não só a intensidade total.

### 4.3 Controle preditivo (MPC)

**Rectified-CFG++** usa o erro avaliado num estado **previsto** meio passo à
frente (Eqs. 32–35).

> **Revisão: MPC.** O controle preditivo prevê a trajetória num horizonte curto
> sob uma entrada candidata, otimiza essa entrada, aplica só o primeiro trecho e
> repete. O Rectified-CFG++ é o caso degenerado — previsão de um passo, sem
> otimização — mas a ideia é a mesma: **antecipar em vez de reagir**.

---

## 5. Controle por modo deslizante (SMC) — a contribuição do artigo

**No artigo: sim.** É o método proposto, o SMC-CFG.

### 5.1 A superfície deslizante

$$s(t)=\dot e(t)+\lambda\,e(t)\qquad\text{(Eq. 19)}$$

Se $s\equiv0$, então $\dot e=-\lambda e$, ou seja, o erro semântico decai
exponencialmente com constante de tempo $1/\lambda$. Essa é a Eq. 18, o
**modelo de referência** que o artigo adota depois de observar (§3.2) que $e$
naturalmente encolhe ao longo da remoção de ruído.

> **Revisão: modelo de referência.** Em vez de exigir "erro zero", você
> especifica *como* o erro deve ir a zero. Escolher $\lambda$ é escolher a
> rapidez exigida. Numa amostragem que dura uma unidade de tempo de fluxo,
> $\lambda=6$ pede queda de $e^{-6}\approx0{,}25\%$ ao longo da corrida.

### 5.2 A lei de chaveamento

$$\Delta e(t)=-K\operatorname{sign}\big(s(t)\big),\quad K=k\mathbf I,\qquad
\hat v=v_\varnothing+w\,\big(e+\Delta e\big)\qquad\text{(Eqs. 25, Alg. 1)}$$

Repare na peculiaridade: a "entrada" é aplicada **ao próprio sinal de erro**,
que depois é multiplicado por $w$. A velocidade de fato somada ao fluxo é
$w\,\Delta e=-wk\operatorname{sign}(s)$.

> **Revisão: modo deslizante em um parágrafo.** Escolha uma função $s$ do estado
> cujo conjunto zero codifique o comportamento desejado. Projete uma entrada
> **descontínua** que sempre empurre $s$ para zero: $u=-k\operatorname{sign}(s)$.
> Seguem duas fases. Na **fase de alcance**, $s$ vai a zero em tempo finito. Na
> **fase de deslizamento**, o estado permanece em $s=0$ e evolui segundo a
> dinâmica reduzida que você projetou — aqui $\dot e=-\lambda e$ — a despeito de
> perturbações limitadas que entrem pelo mesmo canal da entrada (perturbações
> *casadas*). Essa insensibilidade é o motivo de o SMC ser chamado robusto.
> O preço é o **chattering**: em qualquer implementação amostrada o estado não
> consegue permanecer sobre a superfície, cruza-a repetidamente e a entrada
> chaveia na taxa de amostragem.

### 5.3 Lyapunov e convergência em tempo finito

O artigo toma $V(\mathbf s)=\tfrac12\lVert\mathbf s\rVert^2$ e, sob as
Hipóteses 1 e 2 ($\lVert\Phi\rVert\le\delta$ e
$\Gamma=w\mathbf I+\Delta\Gamma$ com $\lVert\Delta\Gamma\rVert\le\rho$),
obtém $\dot V\le-\eta\lVert\mathbf s\rVert$ e portanto
$\lVert\mathbf s(t)\rVert\le\lVert\mathbf s(0)\rVert-\eta t$ (Eqs. 26–28, 40–44).

> **Revisão: Lyapunov.** Uma "energia" $V\ge0$, nula exatamente no alvo, que
> você mostra decrescer ao longo de toda trajetória. Se $\dot V\le-cV$ há
> convergência exponencial (nunca exatamente zero); se $\dot V\le-\eta\sqrt V$,
> como aqui, há convergência em **tempo finito**. É o $\operatorname{sign}$ que
> compra essa taxa: sua magnitude não diminui quando $s\to0$ — e é exatamente
> por isso que ele treme.
>
> Toda prova de SMC precisa de duas coisas: (a) perturbação **limitada**, para
> que um $k$ finito a domine; (b) **autoridade da entrada sobre $\dot s$ com
> sinal conhecido**, isto é $s^\top\Gamma\operatorname{sign}(s)>0$.

### 5.4 Chattering

> **Revisão: chattering e faixa quase-deslizante.** No tempo contínuo um relé
> ideal chaveia infinitamente rápido sobre a superfície. Qualquer implementação
> amostrada chaveia no máximo uma vez por passo, então o estado ultrapassa a
> superfície e volta: um ziguezague de amplitude proporcional a $k$ (Gao, Wang
> & Homaifa, 1995, chamam a faixa resultante de *quasi-sliding-mode band*). O
> chattering é inofensivo quando a planta filtra passa-baixas (a inércia de um
> motor) e nocivo quando não filtra.

---

## 6. Do contínuo ao discreto: o que o código realmente executa

Esta é a seção que mais importa para ler os resultados, porque **a recorrência
implementada não é a equação contínua**.

O `Algorithm 1` do artigo, e o código dos autores, executam:

~~~text
m = e                        # inicialização, apenas no primeiro passo
s = (e - m) + lam * m
delta = -k * sign(s)
v_hat = v_uncond + w * (e + delta)
m = e + delta                # memória do erro CORRIGIDO
~~~

Expandindo a segunda linha:

$$\boxed{\,s_n=e_n+(\lambda-1)\,m_{n-1}\,}$$

Ou seja, a "derivada" $\dot e$ virou uma diferença finita **não dividida pelo
passo de tempo**, e a superfície discreta é uma combinação do erro atual com a
memória anterior. Em `cfgctrl/controllers.py` isso é literalmente
`s = (e_meas - self._e_prev) + cfg.lam * self._e_prev`.

Duas consequências que os diagnósticos medem:

**(a) Com memória corrigida ($m_{n-1}=e_{n-1}+\delta_{n-1}$), a superfície
contém a correção anterior:**

$$s_n=e_n+(\lambda-1)e_{n-1}+(\lambda-1)\delta_{n-1}.$$

Quando os dois termos de erro medido são pequenos frente ao termo de correção,
$s_n\approx(\lambda-1)\delta_{n-1}$, cuja norma RMS é $(\lambda-1)k=0{,}5$ com
$\lambda=6$ e $k=0{,}1$. O sinal inverte a correção seguinte,
$\delta_n\approx-\delta_{n-1}$.

Isso é uma **órbita alternada**, **não** um ponto fixo do estado completo. Não
exige começar de erro/memória exatamente nulos, e $\lvert e\rvert<k$ sozinho não
é condição suficiente para toda trajetória. A redação anterior deste repositório
dizia "ponto fixo" e era forte demais.

**(b) A interpretação contínua não se transfere de graça.** Para memória medida
e $\lambda=6$, impor $s_n=0$ em passos consecutivos exigiria $e_n=-5e_{n-1}$ —
crescimento alternado, não decaimento exponencial. Portanto a superfície em
diferenças finitas **não herda** automaticamente o sentido de
$\dot e+\lambda e=0$. O próprio suplemento do artigo (§6.3.4) separa o argumento
de projeto contínuo do chattering discreto e de um "corredor de estabilidade"
heurístico para $k$.

---

## 7. Refinamentos deste repositório

**Nenhum dos quatro está no artigo.** São opções independentes em
`SMCConfig`; com todas desligadas você tem exatamente a lei publicada.

| opção | padrão | no artigo? |
|---|---|---|
| `switching="sign"` / `"sat"` | `sign` | `sign` é do artigo; `sat` não |
| `store_corrected` | `True` | `True` é do artigo; `False` não |
| `excess_only` | `False` | não |
| `relative_gain` | `False` | não |

### 7.1 Camada limite (`switching="sat"`)

$$\Delta e=-k\operatorname{sat}\!\left(\frac{s}{\phi}\right),\qquad
\operatorname{sat}(z)=\max(-1,\min(1,z)),\qquad \phi=k\lambda .$$

> **Revisão: camada limite.** É o remédio clássico para chattering (Slotine &
> Li, *Applied Nonlinear Control*, §7.1). Dentro de uma faixa de meia-largura
> $\phi$ em torno da superfície, o relé vira um termo proporcional de ganho
> $k/\phi$: a entrada fica contínua e o estado acomoda **dentro** da faixa em
> vez de cruzá-la.

Com $\phi=k\lambda$ e **se** $s=\lambda e$, a lei vira exatamente o
*soft-threshold* $\operatorname{sign}(e)\max(\lvert e\rvert-k,0)$. Essa
igualdade vale na inicialização e quando os erros consecutivos coincidem;
**com memória, em geral $s\ne\lambda e$**, então trata-se de aproximação, não
identidade. Além disso, um valor RMS de $s$ abaixo de $\phi$ não implica que
toda componente esteja dentro da faixa.

### 7.2 Memória do erro medido (`store_corrected=False`)

Guardar $m_n=e_n$ em vez de $m_n=e_n+\delta_n$. Remove o termo
$(\lambda-1)\delta_{n-1}$ e, com ele, o mecanismo de alternância de 7.1(a).

Mas **não** remove a medição anterior: a superfície continua sendo
$s_n=e_n+(\lambda-1)e_{n-1}$, com coeficiente 5 sobre $e_{n-1}$. Portanto
$\operatorname{rms}(s_n)\ne\lambda\operatorname{rms}(e_n)$ em geral, e nada
garante superfície nula na última avaliação do denoiser.

### 7.3 Correção apenas da extrapolação (`excess_only=True`)

O CFG pode ser escrito de duas formas equivalentes:

$$\hat v=v_\varnothing+w\,e \;=\; v_c+(w-1)\,e .$$

A segunda separa a **predição condicional** $v_c$ (a melhor estimativa que o
modelo tem) da **extrapolação** $(w-1)e$ (onde mora tudo o que dá errado com
guidance alto). Esta opção corrige só a extrapolação:

$$\hat v=v_c+(w-1)\big(e+\Delta e\big)
       =v_\varnothing+w\left(e+\frac{w-1}{w}\Delta e\right).$$

Em $w=1$ o resultado é exatamente $v_c$, para qualquer $k$ — ou seja, CFG puro.
A lei do artigo, nesse ponto, entrega $v_c+\Delta e$, que já não amostra a lei
condicional. Em contrapartida, para $w>1$ a correção aplicada é apenas
$(w-1)/w$ da original, então **um resultado favorável a $k$ fixo pode ser
apenas intervenção mais fraca**; a comparação justa exige também uma linha de
base do artigo com $k$ reajustado.

### 7.4 Ganho relativo (`relative_gain=True`)

$$k_n=\alpha\operatorname{rms}(e_n),\qquad \phi_n=\beta\operatorname{rms}(e_n).$$

Motivação: o artigo precisa de $k=0{,}1$ em dois modelos e $0{,}7$ no terceiro,
o que indica que $k$ está em unidades de velocidade que mudam por modelo. Uma
fração do erro corrente é adimensional. **Ganho e largura da camada precisam ser
escalados juntos**, senão a lei deixa de ser equivariante a um reescalonamento
positivo constante da sequência de erro. Unidades consistentes não provam
transferência entre modelos treinados.

---

## 8. O que os diagnósticos medem

A tabela histórica apresentada pelo usuário usa as colunas abaixo.
No pipeline atual, `python -m cfgctrl.benchmark diagnostics --out results/paper`
gera `diagnostics.csv`; os registros JSON por imagem guardam as séries completas.
Os nomes resumidos são `e_last`, `s_last`, `delta_late`, `chatter_late`,
`switch_late` e `velocity_correction_late`. As colunas `predicted`, `deriv` e `±`
abaixo pertencem à tabela histórica, não ao novo CSV.

| coluna | definição | cuidado na leitura |
|---|---|---|
| `rms(e)` | $\operatorname{rms}(e_n)$, primeiro passo → último | a medição, o "sensor" |
| `rms(s) last` | $\operatorname{rms}(s_n)$ na **última avaliação do denoiser** | é antes da atualização final do escalonador; não é medida sobre a imagem decodificada |
| `predicted` | $(\lambda-1)k$, incluindo o fator $(w-1)/w$ quando aplicável | só vale no regime alternado de pequeno erro, com chaveamento por sinal e ganho fixo |
| `chatter` | fração de componentes de $s$ que trocaram de sinal | é estatística de **sinal**; correções suaves podem cruzar zero com amplitude ínfima |
| `switch` | $\operatorname{rms}(\delta_n-\delta_{n-1})$ normalizado por **duas vezes o ganho aplicado** | por amostra/passo vale $\sqrt{\text{chatter}}$ para lei de sinal sem zeros; a igualdade não sobrevive à média |
| `deriv` | desacordo estrito de sinal entre $s_n$ e a **memória** $m_{n-1}$ | **não** compara com o erro medido atual, logo valor baixo **não** prova $s_n\approx\lambda e_n$ |
| `±` | desvio padrão amostral entre pares prompt–semente | não é intervalo de confiança da média |

A tabela adicional de verificação da memória medida usa a desigualdade
triangular sobre normas:

$$\big\lvert(\lambda-1)R_{n-1}-R_n\big\rvert\le R_s\le(\lambda-1)R_{n-1}+R_n,
\qquad R_j=\operatorname{rms}(e_j).$$

São **limitantes de norma**: logs escalares de RMS não reconstroem direção.

---

## 9. Lendo a corrida COCO

Números medidos em SD3.5-large, 1000 prompts do MS-COCO, 5 escalas, 30 passos.

**Braço `paper` ($\lambda=6$, $k=0{,}1$, sinal, memória corrigida).**
$\operatorname{rms}(s)$ final entre 0,5088 e 0,5154 nas cinco escalas, com
desvio de $\pm0{,}002$ entre 1000 prompts, enquanto
$\operatorname{rms}(e)$ cai de 0,1851 para ~0,010. Razão 1,02–1,03 contra o
valor limite $(\lambda-1)k=0{,}5$. `chatter` 0,993–0,997 e `switch` 0,996–0,998.

Leitura: **consistente com a órbita alternada induzida pela memória** descrita
em §6(a). Quase toda componente inverte sua correção de amplitude fixa a cada
passo, e a superfície não decresce junto com o erro. A quase independência de
$w$ também é esperada: o que se armazena é uma correção de magnitude $k$,
**antes** da multiplicação por $w$.

**Braço `excess` (saturação + memória medida + só extrapolação).**
$\operatorname{rms}(s)$ final 0,099–0,114, com `chatter` ~0,26 e `switch`
~0,086–0,090 (normalizado pelo ganho **aplicado**, já com $(w-1)/w$).

Leitura: o termo de correção anterior sumiu, e a superfície caiu ~90% em
relação ao valor inicial $\lambda\operatorname{rms}(e_0)=6\times0{,}1851=1{,}11$.
Mas ela **continua contendo** $(\lambda-1)e_{n-1}$, e por isso fica em ~0,1 e
não perto de zero. Um valor final de 0,1 sozinho não mostra que a superfície
parou de cair. O `chatter` residual de ~0,26 vem em boa parte do próprio modelo
mudando de sinal entre passos.

**O que isso estabelece.** A recorrência discreta implementada não atinge
superfície nula na última avaliação, e a alternância do braço `paper` é
compatível com o mecanismo de memória previsto. **O que não estabelece:** nada
sobre qualidade de imagem. O artigo reporta ganhos de FID/CLIP e nada aqui os
contradiz. Para qualidade, use `python -m cfgctrl.benchmark evaluate --out results/paper`
e `python -m cfgctrl.benchmark report --out results/paper`, após a geração.
Compare fidelidade em níveis semelhantes de alinhamento, usando uma varredura
de escalas e os mesmos prompts/sementes. O novo relatório apresenta as métricas;
não declara automaticamente um vencedor.

---

## 10. Tabela-resumo

| controle | no artigo | neste repo | onde |
|---|---|---|---|
| Proporcional (CFG, ganho $w$) | sim, é a releitura central | sim | `presets.cfg_baseline()` |
| Escalonamento de ganho $w(t)$ | citado (Tab. 1) | não | — |
| Projeção (APG, CFG-Zero\*) | citado (Tab. 1) | não | — |
| Preditivo (Rectified-CFG++) | citado (Tab. 1) | não | — |
| **Modo deslizante, $\operatorname{sign}(s)$** | **sim, é a contribuição** | sim | `presets.paper()` |
| Camada limite $\operatorname{sat}(s/\phi)$ | não | sim | `switching="sat"` |
| Memória do erro medido | não | sim | `store_corrected=False` |
| Correção só da extrapolação | não | sim | `excess_only=True` |
| Ganho relativo | não (é *future work* deles) | sim | `relative_gain=True` |
| Ação integral / anti-windup | não | não | não há integrador aqui |

---

## 11. Glossário

**Planta** — o sistema a controlar. Aqui, o integrador da EDO com a rede.

**Realimentação** — usar a medição para decidir a entrada. Existe aqui: a
correção muda $x$, que muda a medição seguinte.

**Ganho de malha** — produto dos ganhos ao longo da malha; quanto de uma
mudança na entrada retorna como mudança na medição uma volta depois.

**Modelo de referência** — trajetória desejada para o erro, aqui
$\dot e=-\lambda e$.

**Superfície deslizante** — função $s$ do estado cujo zero codifica o
comportamento desejado.

**Fase de alcance / fase de deslizamento** — chegar à superfície em tempo
finito / permanecer nela com dinâmica reduzida.

**Condição de alcance** — $s^\top\dot s<0$, ou a versão forte
$s^\top\dot s\le-\eta\lVert s\rVert$ que dá tempo finito. Exige autoridade da
entrada sobre $\dot s$ com sinal conhecido.

**Perturbação casada / não casada** — entra pelo mesmo canal da entrada e pode
ser cancelada pelo chaveamento / não entra e não pode.

**Chattering** — chaveamento de frequência finita em torno da superfície, com
amplitude proporcional ao ganho e ao passo.

**Camada limite** — trocar $\operatorname{sign}$ por saturação de largura
$\phi$; entrada contínua, estado acomoda dentro da faixa.

**Grau relativo** — quantas derivadas da saída são necessárias até a entrada
aparecer. O SMC clássico supõe grau relativo um em $s$.

**Função de Lyapunov** — "energia" positiva definida que decresce ao longo das
trajetórias; a taxa de decrescimento dá o tipo de convergência.

**Soft-threshold** — $\operatorname{sign}(e)\max(\lvert e\rvert-k,0)$; zera
componentes pequenas e encolhe as grandes em $k$.

**Órbita alternada** — ciclo em que a variável troca de sinal a cada passo com
amplitude aproximadamente constante. **Não** é ponto fixo do estado completo.
