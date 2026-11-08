# DOTPG Draft

Source PDF: `thesis_reference/DOTPG-draft.pdf`

Converted with `pdftotext -layout -enc UTF-8 -eol unix`; PDF page breaks are represented as Markdown horizontal rules.

     B RIDGING THE R EALITY G AP : D UAL O PTIMAL T RANSPORT
      P OLICY G RADIENT FOR R EAL -W ORLD R EINFORCEMENT
                L EARNING FROM D EMONSTRATIONS


                                Hongjun Ma1 , Weichang Li1,2 , Zhihua Yang2 Lincong Wu2
                                               1
                                                South China University of Technology
                                        2
                                            Guangzhou Wanwei Vision Technology Co., Ltd.
    mahongjun@scut.edu.cn, aumax@mail.scut.edu.cn, yangzhihua@tringai.com, wulincong@tringai.com




                                                       A BSTRACT
         A fundamental obstacle to deploying reinforcement learning (RL) in real-world settings is the
         pervasive reality gap—a systematic discrepancy that causes policies trained in simulation to fail when
         deployed on physical systems. Although learning from demonstrations (LfD) provides a promising
         alternative to manual reward engineering, current approaches often depend on imperfect predefined
         reward functions or encounter issues with distribution shift, which limits their practical applicability.
         To address these challenges, this paper introduces the Dual Optimal Transport Policy Gradient
         (DOT-PG) framework, which uses optimal transport duality to learn robust policies directly from
         expert trajectories. By minimizing the Wasserstein distance between the state-action distributions of
         the agent and the expert, DOT-PG automatically extracts both a stable policy gradient and a robust
         value function from the optimal transport dual variables—entirely removing the need for handcrafted
         reward design. The proposed method offers theoretical guarantees for policy improvement and is
         designed to learn policies that remain invariant under sim-to-real dynamics mismatches. Extensive
         experiments show that DOT-PG substantially outperforms state-of-the-art imitation learning and
         RLfD baselines in zero-shot sim-to-real transfer success rate, sample efficiency, and asymptotic
         performance, establishing a new paradigm for training real-world policies from demonstration data.


Keywords Sim-to-Real Transfer, Optimal Transport, Policy Gradient, Reinforcement Learning from Demonstrations,
Robust Reinforcement Learning


1     Introduction

The paradigm of reinforcement learning (RL) has demonstrated remarkable unprecedented success across diverse
domains, from mastering complex games to enabling sophisticated robotic behaviors. However, the transition of RL
methodologies from controlled simulated environments to real-world physical systems remains fraught with fundamental
persistent challenges. Central among these is the “reality gap”—the systematic inherent discrepancy between simulation
and reality that manifests through unmodeled dynamics, sensor inaccuracies, actuator delays, and environmental
uncertainties. This gap represents a critical fundamental bottleneck that prevents the widespread deployment of RL in
safety-critical applications such as autonomous driving, medical robotics, and industrial automation.
The challenge of the reality gap is particularly acute in domains requiring fine-grained precise physical interactions.
While sophisticated high-fidelity simulators like MuJoCo, Drake and Issac Sim have advanced significantly, they
inevitably incorporate certain deliberate simplifications and approximations that diverge from physical reality. These
discrepancies, though often imperceptible during training, can lead to sudden catastrophic failures during deployment.
A policy that appears optimal in simulation may fail abruptly and completely when faced with subtle real-world friction,
sensor noise, or unmodeled dynamics. This pressing challenge has spurred urgent significant research interest in
developing algorithms that can learn robust policies capable of bridging the simulation-to-reality divide.


---

Learning from demonstrations (LfD) has emerged as a promising approach to address this challenge. By leveraging
expert trajectories, LfD methods seek to bypass the need for meticulous reward engineering—a process that is
particularly challenging in real-world scenarios where reward signals may be sparse, poorly shaped, or difficult to
specify. However, current LfD methodologies face their own fundamental limitations, such as compounding causal
distribution shift, inherent sensitivity to noisy demonstrations, over-reliance on prohibitively expensive large datasets,
and poor out-of-distribution generalization capabilities, when applied to real-world settings.
Behavioral cloning (BC), while conceptually simple and straightforward to implement, suffers from the well-documented
problem of compounding errors. The supervised learning framework of BC fails to account for the sequential nature
of decision-making, leading to cascading failures when the agent encounters states outside the expert’s distribution.
Techniques like Dataset Aggregation (DAgger) partially mitigate this issue through interactive expert querying, but this
approach requires substantial human involvement and may be impractical for many real-world applications.
Inverse reinforcement learning (IRL) and its modern variants offer a more principled alternative by seeking to recover
the underlying reward function that explains expert behavior. However, IRL methods typically involve solving a nested
optimization problem that requires repeated policy evaluation in an inner loop, making them computationally prohibitive
for complex tasks. Moreover, the reward functions recovered by IRL may not generalize well across the reality gap, as
they are often tied to features that are specific to the simulation environment.
The advent of adversarial imitation learning, particularly Generative Adversarial Imitation Learning (GAIL), represented
a significant advancement by framing imitation as a distribution matching problem. While GAIL and its variants have
achieved impressive results, they inherit the training instabilities characteristic of generative adversarial networks.
These methods are often sensitive to hyperparameter choices, prone to mode collapse, and may produce policies that
exploit simulator-specific artifacts rather than learning robust behaviors.
The fundamental limitation shared by these approaches is their reliance on metrics that may not adequately capture
the geometric structure of the policy space. f-divergences, which underlie many adversarial methods, can provide
misleading signals when distributions have limited support overlap—a common scenario in real-world applications
where data is scarce. This limitation becomes particularly problematic when policies must generalize across the reality
gap, where the training and deployment distributions may have significant divergence.
On the other hand, optimal transport (OT) theory offers a principled rigorous framework that directly addresses these
core limitations. The Wasserstein distance, unlike typical f-divergences, provides a smooth and meaningful geometric
measure of distribution discrepancy even when distributions have completely disjoint supports. This crucial property
makes OT particularly well-suited for imitation learning scenarios where limited expert demonstrations may cover only
a small portion of the relevant state-action space. More importantly, the dual formulation of optimal transport provides
a natural and direct connection to value function and generalized advantage learning in reinforcement learning.
The Kantorovich-Rubinstein duality establishes that the Wasserstein distance can be expressed as a supremum over
Lipschitz functions. This dual perspective reveals that the optimal critic in the OT formulation serves simultaneously as
a measure of distribution discrepancy and as a potential-based reward signal for policy improvement. This insight forms
the theoretical foundation of our approach, bridging the gap between distribution matching and policy optimization.
In this paper, we introduce the novel Dual Optimal Transport Policy Gradient (DOT-PG) framework, which system-
atically leverages optimal transport duality to enable provably robust reinforcement learning directly from expert
demonstrations trajectories. Our principled approach is grounded in several fundamental key insights:
First, by formulating imitation learning as a Wasserstein distance minimization problem, we obtain a smooth and
well-behaved optimization landscape that facilitates stable training and better convergence properties. The Lipschitz
continuity requirements inherent in the OT dual formulation naturally enforce regularization on the learned value
functions, preventing the explosive gradients that often plague adversarial methods.
Second, the optimal dual variables in our framework serve multiple purposes, that is, they not only provide a dense
learning signal for policy improvement, but also act as a robust value function estimate, and at the same time naturally
yield an advantage function for policy gradients. This multi-functional role eliminates the need for separate reward
learning and policy optimization phases, leading to greater algorithmic simplicity and efficiency.
Third, the geometric properties of the Wasserstein distance make our approach particularly suitable for real-world
applications. By focusing on the underlying metric structure of the state-action space rather than the probability
density ratios, the proposed DOT-PG algorithm learns policies that are more robust to distribution shifts and dynamics
mismatches, which usually lead to the challenges in dealing with the sim-to-reality gap.
The contributions of this work are fourfold, encompassing both theoretical foundations and practical implementations:




                                                            2


---

First, we propose DOT-PG, a novel framework that for the first time unifies optimal transport duality with policy
gradient methods, establishing a new paradigm for reinforcement learning directly from demonstrations.
Second, we establish comprehensive theoretical guarantees demonstrating that our approach ensures monotonic policy
improvement through bounded policy updates with provable convergence. The derived policy gradients exhibit inherent
robustness properties fundamentally due to the Lipschitz continuity enforced by the OT dual formulation, providing
rigorous mathematical foundations for stable and convergent learning under well-behaved state distributions.
Third, we develop practical, efficient, and scalable algorithmic implementations that maintain theoretically crucial
Lipschitz boundedness through differentiable normalization and learnable adaptive constraint mechanisms. This design
ensures certified convergence properties while enabling robustness against dynamics mismatch, observation noise,
unmodeled effects and environmental variability, effectively avoiding complex density ratio estimation.
Fourth, we conduct extensive experimental validation demonstrating DOT-PG’s performance robustness and policy
convergence in challenging sim-to-real transfer. In general, our method outperforms state-of-the-art baselines while
maintaining persistent stable performance and consistent convergence behavior across diverse continuous control tasks.
The DOT-PG framework represents a significant step toward making reinforcement learning practically viable for
real-world systems. By leveraging the mathematical foundations of optimal transport theory, we provide a principled
solution to the challenge of learning robust policies from demonstrations that can successfully bridge the reality gap.


2     Related Work

2.1   Imitation Learning Paradigms

Imitation learning has evolved through several distinct methodological approaches, each with unique strengths and
limitations in addressing the reality gap challenge. Behavioral Cloning (BC) represents the most straightforward
approach, framing imitation as a supervised learning problem over expert state-action pairs. While computationally
efficient, BC suffers from fundamental limitations including compounding errors and covariate shift, where small errors
accumulate during execution leading to progressively worsening performance. Pioneering work by Ross et al. (2011)
introduced Dataset Aggregation (DAgger) to address these issues through iterative data collection and expert relabeling,
though this approach requires ongoing expert involvement that may be impractical for real-world deployment.
Inverse Reinforcement Learning (IRL) approaches, initiated by Ng and Russell (2000), take a fundamentally different
perspective by attempting to recover the underlying reward function that explains expert behavior. The Maximum
Entropy IRL framework (Ziebart et al., 2008) advanced the field by providing a probabilistic formulation that resolves
reward ambiguity, while later extensions like Guided Cost Learning (Finn et al., 2016) and Adversarial Inverse
Reinforcement Learning (AIRL) (Fu et al., 2018) improved scalability through deep learning integration. However,
these methods typically require solving a nested reinforcement learning problem in an inner loop, making them
computationally intensive and challenging to tune for complex tasks.

2.2   Adversarial Imitation Learning

The introduction of Generative Adversarial Imitation Learning (GAIL) (Ho & Ermon, 2016) marked a significant
paradigm shift by framing imitation as a distribution matching problem using adversarial training. GAIL employs a
discriminator to distinguish between expert and agent trajectories while the policy learns to generate behavior that
fools this discriminator. Subsequent improvements include InfoGAIL (Li et al., 2017), which incorporates latent
codes for skill discovery, and VAIL (Peng et al., 2019), which adds information-theoretic constraints for improved
sample efficiency. Despite their empirical success, these methods inherit the training instabilities characteristic of
generative adversarial networks, including mode collapse, sensitivity to hyperparameters, and difficulty in achieving
convergence—issues that become particularly problematic in real-world applications where training stability is crucial.

2.3   Optimal Transport in Machine Learning

Optimal transport theory has emerged as a powerful framework for comparing probability distributions in machine
learning. The Wasserstein GAN (Arjovsky et al., 2017) demonstrated the practical benefits of using Wasserstein
distances for generative modeling, providing improved training stability and meaningful loss metrics. In domain
adaptation, Wasserstein Distance Guided Representation Learning (Shen et al., 2018) showed that OT metrics can
effectively align distributions across domains. More recently, optimal transport has been applied to reinforcement
learning through methods like Wasserstein Adversarial Imitation Learning (WAIL) (Xiao et al., 2019) and Wasserstein


                                                           3


---

Inverse Reinforcement Learning (WIRL) (Ni et al., 2020), which replace f-divergences with Wasserstein distances but
often maintain the adversarial training framework with its associated instability.

2.4   Policy Gradient Methods

Policy gradient methods form the foundation of modern deep reinforcement learning in continuous action spaces.
The Deep Deterministic Policy Gradient (DDPG) (Lillicrap et al., 2015) extended the deterministic policy gradient
theorem to deep neural networks, while Twin Delayed DDPG (TD3) (Fujimoto et al., 2018) addressed overestimation
bias through clipped double Q-learning and delayed policy updates. The Soft Actor-Critic (SAC) (Haarnoja et al.,
2018) framework incorporated maximum entropy reinforcement learning for improved exploration and robustness.
While these methods have achieved state-of-the-art performance in simulated environments, they typically require
carefully engineered reward functions and struggle with the sim-to-real transfer problem due to their sensitivity to
reward specification and dynamics mismatch.

2.5   Bridging the Reality Gap

Several approaches have been developed specifically to address the simulation-to-reality transfer challenge. Domain
Randomization (Tobin et al., 2017) exposes policies to varied simulated conditions to encourage robustness, while
System Identification (Yu et al., 2017) attempts to precisely match simulation parameters to real-world dynamics.
Meta-learning approaches (Finn et al., 2017) aim to learn policies that can quickly adapt to new environments, and
Robust Reinforcement Learning methods (Morimoto & Doya, 2005; Pinto et al., 2017) explicitly optimize for worst-case
performance under dynamics uncertainty. However, these approaches often require extensive real-world data collection
or make conservative assumptions that limit asymptotic performance.

2.6   Integration with Imitation Learning

Recent work has begun to explore the intersection of imitation learning and sim-to-real transfer. SimOpt (Ramos et al.,
2019) combines domain randomization with demonstration-guided policy search, while DART (James et al., 2019)
uses adversarial discriminators to align simulation and real-world data distributions. However, these methods typically
maintain the adversarial training framework with its associated instabilities and high sample complexity, or require
complex multi-stage optimization procedures that are sensitive to hyperparameter tuning.

2.7   Highlights of DOT-PG Algorithm

Compared with these results, our proposed Dual Optimal Transport Policy Gradient (DOT-PG) framework distinguishes
itself from prior work by directly leveraging optimal transport duality to connect distribution matching with policy
optimization in a single coherent framework. Unlike adversarial methods, DOT-PG provides theoretical guarantees
of policy improvement and training stability through its Lipschitz-constrained optimization. Unlike traditional IRL
approaches, it avoids the computationally expensive inner reinforcement learning loop. And unlike prior OT-based
imitation learning methods, it fully exploits the connection between OT dual variables and value functions to enable
direct policy gradient computation without intermediate reward learning. This integrated approach allows DOT-PG
to simultaneously address the challenges of sample efficiency, training stability, and sim-to-real robustness that have
limited previous methods, providing a principled foundation for real-world reinforcement learning from demonstrations.


3     Theoretical Foundations

3.1   Optimal Transport Theory

Optimal transport theory provides a powerful geometric framework for comparing probability distributions by con-
sidering the minimal cost required to transform one distribution into another. The theory has evolved through two
fundamental formulations that provide complementary perspectives on distribution matching.
Monge Formulation: The classical Monge formulation seeks a deterministic transport map that minimizes the
transportation cost. Precisely, given two probability measures µ and ν defined on metric spaces X and Y respectively,
and a cost function c : X × Y → R+ , the Monge problem aims to find a transport map T : X → Y that minimizes
                                                     Z
                                               inf      c(x, T (x))dµ(x)                                         (1)
                                             T ♯µ=ν   X


                                                          4


---

where T ♯µ denotes the pushforward measure of µ under T , requiring that T transports the mass from µ to ν. While
conceptually elegant, this formulation is often too restrictive as it requires the existence of a deterministic transport map
and may not have a solution when µ contains atoms but ν does not.
Kantorovich Relaxation: The Kantorovich formulation generalizes the Monge problem by considering probabilistic
couplings rather than deterministic maps. This relaxation admits solutions under much more general conditions, i.e.,
                                                           Z
                                     Wc (µ, ν) = inf             c(x, y)dγ(x, y)                                  (2)
                                                      γ∈Π(µ,ν)        X ×Y

where Π(µ, ν) denotes the set of all joint distributions (couplings) with marginals µ and ν. The coupling γ specifies
how mass is transported from µ to ν, allowing for splitting mass from a single source point to multiple target points.
Wasserstein Distance: For p ≥ 1, the p-Wasserstein distance provides a family of metrics on probability distributions
                                                     Z                       1/p
                                                                    p
                              Wp (µ, ν) =       inf          d(x, y) dγ(x, y)                                      (3)
                                                   γ∈Π(µ,ν)       X ×Y
                        +
where d : X × Y → R is a metric on the underlying space. It should be emphasized that the case p = 1 is particularly
important due to its connection to the Kantorovich-Rubinstein duality which will be discussed in Theorem 1.
Connectioin to Our Result: We consider the state-action distributions ρπ (agent) and ρE (expert) defined on the joint
state-action space S × A. The key advantage of Wasserstein distance over f-divergences (such as KL-divergence or
JS-divergence) is its ability to provide meaningful gradients even when distributions have non-overlapping support.
This property makes it particularly suitable for imitation learning, where the agent’s initial distribution may be far from
the expert distribution, and ensures smooth and stable training dynamics throughout the learning process.
What’s more, the geometric nature of optimal transport allows it to capture the underlying structure of the state-action
space, making it robust to variations in policy parameterization and environmental dynamics, as it enables policies to
maintain expert-like behavior even under distribution shifts between simulation and real-world deployment.
Essentially, the optimal Transport theory provides a geometric framework for comparing probability distributions by
finding the minimal cost required to transform one distribution into another. Given two probability distributions ρπ
(agent) and ρE (expert) defined on a metric space X , and for any p ≥ 1, the p-Wasserstein distance is defined as below
                                                           Z                        1/p
                              Wp (ρπ , ρE ) =      inf             d(x, y)p dγ(x, y)
                                                   γ∈Π(ρπ ,ρE )       X ×X
where Π(ρπ , ρE ) denotes the set of all joint distributions (couplings) with marginals ρπ and ρE , d(x, y) is a metric on
X , typically the Euclidean distance, γ represents the transport plan that specifies how mass is moved from ρπ to ρE .

3.2     Kantorovich Duality and Entropic Regularization

The Kantorovich duality and entropic regularization provide the mathematical foundation that connects primal optimal
transport problems with their dual formulations, and enable efficient computation and theoretical analysis.

3.2.1    Fundamental Duality Theorems
The fundamental theorem of Kantorovich duality establishes the dual formulation for the 1-Wasserstein distance.
Theorem 1 (Kantorovich-Rubinstein Duality). For the 1-Wasserstein distance with metric cost d(x, y), we have
                                                Z                  Z             
                             W1 (µ, ν) = sup          f (x)dµ(x) −     f (y)dν(y)                               (4)
                                                f ∈Lip1       X              Y

where Lip1 is the set of 1-Lipschitz functions |f (x1 ) − f (x2 )| ≤ d(x2 , x2 ), d(x1 , x2 ) is the distance between x1 , x2 .

Proof. Based on the Fenchel-Rockafellar duality framework, let us consider the following two convex functionals
                                    Z          Z                 
                                                                   0      if f ∈ Lip1
                            Φ(f ) = f dµ − f dν, Ψ(f ) =                                                        (5)
                                                                   +∞ otherwise
Then, we give the conjugate functions of Φ(f ) and Ψ(f ), which is calculated by the supremums of integral functions
                              Z                     Z                                    Z
                  ∗                                                             ∗
                Φ (g) = sup       f g − Φ(f ) = sup        f (g + ν − µ) , Ψ (g) = sup          f g.              (6)
                             f                            f                                     f ∈Lip1



                                                                  5


---

Note that Φ∗ (g) is finite only when gR = µ − ν in the distributional sense, otherwise it diverges to +∞. Obviously, this
is because the supremum over f of f (g + ν − µ) is unbounded unless the equation g + ν − µ = 0 is imposed.
Next, by the Fenchel-Rockafellar duality theorem, it can be directly derived that the following equation holds
                                        sup [Φ(f ) − Ψ(f )] = inf [Ψ∗ (g) + Φ∗ (−g)]                                     (7)
                                         f                               g

It can be seen that the left-hand side of this equation is exactly our primal problem of Fenchel-Rockafellar duality
                                                       Z          Z      
                                                  sup       f dµ − f dν                                              (8)
                                                 f ∈Lip1

For the right-hand side of this equation, substituting our conjugate functions gives the following expression
                                                              "       Z                #
                                 inf [Ψ∗ (g) + Φ∗ (−g)] = inf sup        f g + Φ∗ (−g)                                   (9)
                                 g                                   g       f ∈Lip1

Since Φ∗ (−g) implicitly imposed the constraints −g = ν − µ (or equivalently g = µ − ν), thus it implies
                                              Z                    Z
                                  inf sup        f g = sup f ∈ Lip1 f d(µ − ν)                                       (10)
                                     g=µ−ν f ∈Lip1

This confirms the Fenchel-Rockafellar duality and yields the desired 1-Wasserstein distance formulation, that is
                                                         Z          Z     
                                     W1 (µ, ν) = sup          f dµ − f dν .
                                                           f ∈Lip1


The conclusion can be extended for general cost functions, i.e., we have the following more general Kantorovich duality.
Theorem 2 (General Kantorovich Duality). For any lower semi-continuous cost function c(x, y), we have the equation
                                            Z                    Z          Z     
                                     inf       cdγ = sup             f dµ + gdν                                   (11)
                                      γ∈Π(µ,ν)                (f,g)∈Φc

where Φc is defined by the set of (f, g), i.e., Φc = {(f, g) ∈ L1 (µ) × L1 (ν) : f (x) + g(y) ≤ c(x, y) for all x, y}.

All these duality results hold particular significance for our proposed DOT-PG framework, as they fundamentally trans-
form the computationally challenging optimization over transport couplings into a more tractable functional optimization
problem. This reformulation enables the optimal dual variables f ∗ and g ∗ to be naturally interpreted as meaningful
quantities that can be effectively approximated using neural networks. Moreover, in the 1-Wasserstein formulation, the
inherent Lipschitz constraint serves as a powerful regularizer for the learned value functions, simultaneously ensuring
training stability while enhancing the generalization capabilities of our policy gradient method—precisely the properties
needed to bridge the simulation-to-reality gap in practical reinforcement learning applications.

3.2.2   Entropic Regularization and Sinkhorn Algoritm
While the Kantorovich duality provides theoretical elegance, the computational complexity of solving the optimal
transport problem scales poorly with dimension. We use entropic regularization to transform the original problem into
                                                      Z                            
                                Wc,ϵ (µ, ν) = inf         cdγ + ϵ · KL(γ∥µ ⊗ ν)                                  (12)
                                                     γ∈Π(µ,ν)

where KL denotes the Kullback-Leibler divergence and ϵ > 0 is the regularization parameter. The entropic regularization
convexifies the primal problem and enables efficient computation through the celebrated Sinkhorn algorithm.
The entropic regularization can be understood as introducing a “logistical flexibility” term: rather than insisting on the
absolute minimum cost regardless of planning complexity, we allow for slightly suboptimal routes in exchange for a
more manageable and robust distribution plan. The KL divergence term KL(γ∥µ ⊗ ν) penalizes plans that deviate too
drastically from simply independently sampling source and destination pairs, effectively smoothing the transport map.
By the entropic regularization, the dual formulation of this regularized problem becomes into the following form
                                     Z          Z           Z                                    
                                                                   f (x)+g(y)−c(x,y)
                  Wc,ϵ (µ, ν) = sup       f dµ + gdν − ϵ         e         ϵ         − 1 dµ(x)dν(y)              (13)
                                  f,g


                                                                     6


---

The optimal solutions satisfy the elegant scaling relations, which actually form the principle of the Sinkhorn algorithm
                                      Z                                       Z
                                           g(y)−c(x,y)                            f (x)−c(x,y)
                     f (x) = −ϵ log e           ϵ      dν(y), g(y) = −ϵ log e           ϵ      dµ(x)                 (14)

The mathematical structure of Kantorovich duality and entropic regularization provides both computational tractability
and interpretability: the potentials f and g correspond to value functions that assess state and action qualities, at the
same time, the entropic regularization ensures stable learning and prevents overfitting to the demonstration data. In our
DOT-PG framework, we integrate the Sinkhorn algorithm as an initialization method for the Critic network, which
approximates the OT dual variable and significantly enhances learning efficiency and stability.

3.3     Reinforcement Learning Foundations

3.3.1    Markov Decision Process
The theoretical foundation of our approach is built upon the Markov Decision Process (MDP) framework, which
provides a mathematical formalism for sequential decision-making problems. An MDP is formally defined by the
tuple (S, A, P, r, γ, ρ0 ), where S is the State space representing all possible configurations of the environment; A
is the Action space containing all possible decisions the agent can make; P(s′ |s, a) is the State transition dynamics
specifying the probability of transitioning to state s′ from state s after taking action a; r(s, a) is the Reward function,
providing immediate feedback for taking action a in state s; γ ∈ [0, 1) is the Discount factor, balancing immediate
versus future rewards; ρ0 (s) is the Initial state distribution. The agent’s behavior is governed by a policy π(a|s), which
defines a probability distribution over actions given states. The objective in standard RL is to find an optimal policy π ∗
that maximizes the expected discounted return, which is usually written in the following mathematical form
                                                               "∞                  #
                                                                 X
                                               J(π) = Eτ ∼π         γ t r(st , at )                                    (15)
                                                               t=0

where τ = (s0 , a0 , s1 , a1 , . . . ) denotes a trajectory sampled by following policy π which is actually a dynamic behavior.
In our proposed framework, we maintain the MDP structure but eliminate the dependence on manually designed reward
functions by leveraging the optimal transport duality to derive intrinsic learning signals from expert demonstrations.

3.3.2    State-Action Value Functions
The state-action value function (Q-function) plays a central role in value-based reinforcement learning methods. For a
given policy π, the Q-function Qπ (s, a) represents the expected discounted return when starting from state s, taking
action a, and thereafter following policy π, which is formally written by the following equation
                                                 "∞                                    #
                                                  X
                                                        k
                               Qπ (s, a) = Eτ ∼π       γ r(st+k , at+k ) st = s, at = a .                         (16)
                                                     k=0

This formulation captures the long-term value of taking a specific action in a given state while committing to the policy
π for all future decisions. According to classical reinforcement learning theory, the optimal Q-function Q∗ (s, a), which
corresponds to the best achievable performance from any policy, satisfies the Bellman optimality equation, i.e.,
                                                                      h                 i
                                 Q∗ (s, a) = r(s, a) + γEs′ ∼P(·|s,a) max ′
                                                                            Q ∗ ′ ′
                                                                               (s  , a )  .                          (17)
                                                                           a

This recursive relationship forms the foundation of many RL algorithms, expressing that the optimal value equals the
immediate reward plus the discounted value of the best possible future state. The fixed-point nature of this equation
enables iterative solution methods like Q-learning. Also, an advantage function Aπ (s, a) can be defined to provide a
relative measure of action quality by comparing each action’s value against the average performance of the policy
                                              Aπ (s, a) = Qπ (s, a) − V π (s)                                             (18)
          π                    π
where V (s) = Ea∼π(·|s) [Q (s, a)] is the state-value function representing the expected return from state s when
following policy π. The advantage function has several important properties: 1. zero expectation: Ea∼π(·|s) [Aπ (s, a)] =
0 for all states s; 2. policy gradient: ∇θ J(θ) = Es∼dπ ,a∼πθ [∇θ log πθ (a|s)Aπ (s, a)]; 3. intuitive interpretability:
positive values indicate better-than-average actions, while negative values indicate worse-than-average actions.
In DOT-PG, these concepts connect elegantly with optimal transport theory, where the advantage function naturally
emerges from the dual formulation and inherits the Lipschitz continuity properties essential for robust policy learning,
providing a dense learning signal that guides policy improvement without requiring explicit reward specification.


                                                              7


---

4     Problem Formulation and DOT-PG Algorithm
4.1     Problem Formulation

4.1.1    Markov Decision Process and Distribution Matching
We consider the standard reinforcement learning framework formalized as a Markov Decision Process (MDP), defined
by the tuple (S, A, P, r, γ, ρ0 ). However, unlike the traditional RL settings, we operate under the learning from
demonstrations (LfD) paradigm where the reward function r(s, a) is unknown or difficult to specify. Instead, we are
provided with expert demonstrations DE = (si , ai )i = 1N sampled from the expert’s state-action distribution ρE (s, a).
The fundamental objective is to find a policy πθ that minimizes the discrepancy between its induced state-action
distribution ρπθ and the expert distribution ρE . We formulate this as a Wasserstein distance minimization problem
                                                         Z
                   min W (ρπθ , ρE ) = min        inf       c((s, a), (sE , aE ))dγ((s, a), (sE , aE ))           (19)
                      θ                     θ    γ∈Π(ρπθ ,ρE )

where Π(ρπθ , ρE ) denotes the set of all couplings between ρπθ and ρE , and c((s, a), (sE , aE )) is a cost function
measuring the distance between state-action pairs. Then, the discounted state-action distribution ρπ is defined as follows
                                                           ∞
                                                           X
                                   ρπ (s, a) = (1 − γ)           γ t P (st = s, at = a|π, P)                           (20)
                                                           t=0


4.1.2    Kantorovich Duality-Based Reformulation
Now, by applying the Kantorovich-Rubinstein duality, we can transform the primal optimal transport problem into a
more tractable min-max optimization problem which is rigorously given by the following formal expression
                                                                                 
                              min max E(s,a)∼ρE [f (s, a)] − E(s,a)∼ρπθ [f (s, a)]                           (21)
                                   θ   f ∈Lip1

This dual reformulation offers significant computational advantages that facilitate practical implementation. Most
importantly, the dual variable f (s, a) can be efficiently approximated using neural networks, enabling the capture of
complex value functions in high-dimensional state-action spaces. Furthermore, this parameterization allows for gradient-
based optimization, making the framework compatible with stochastic gradient methods that scale to large datasets.
Critically, the inherent Lipschitz constraint on f (s, a) serves as a powerful regularizer that stabilizes the training
process by preventing overfitting and ensuring smooth optimization landscapes, thereby simultaneously combining the
representational capacity of deep learning with the theoretical robustness of optimal transport.

4.1.3    Connecting Optimal Transport with Policy Gradients
Traditionally, the foundation for policy-based reinforcement learning methods is established by a certain of policy
gradient. That is, for a differentiable policy πθ parameterized by θ, the gradient of the expected return is given by
                                  ∇θ J(πθ ) = E(s,a)∼ρπθ [∇θ log πθ (a|s)Qπθ (s, a)] .                                 (22)
The basic idea herein begins with the expression for the expected return which is written as the following form
                                            "∞                 # Z
                                              X
                                                   t
                           J(πθ ) = Eτ ∼πθ        γ r(st , at ) = ρπθ (s, a)r(s, a)dsda.                               (23)
                                                  t=0

Taking the gradient and applying the log-derivative trick immediately gives the policy gradient as below
                             Z                              Z
                 ∇θ J(πθ ) = ∇θ ρπθ (s, a)r(s, a)dsda = ρπθ (s, a)∇θ log ρπθ (s, a)r(s, a)dsda                         (24)

Using the identity ∇θ log ρπθ (s, a) = ∇θ log πθ (a|s)+∇θ log dπθ (s), where dπθ (s) is the discounted state distribution,
and noting that the term involving ∇θ log dπθ (s) vanishes in expectation, we obtain the desired expression. Furthermore,
to reduce variance in the procedure of gradient estimation, the classical RL method is revised to subtract a state-dependent
baseline V πθ (s) from the action-value function, defining the advantage function Aπθ (s, a) = Qπθ (s, a) − V πθ (s).
This strategy is then developed to preserve the unbiasedness of the gradient estimate while reducing variance
                                  ∇θ J(πθ ) = E(s,a)∼ρπθ [∇θ log πθ (a|s)Aπθ (s, a)] .                                 (25)


                                                                 8


---

Our key theoretical contribution establishes a fundamental connection between distribution matching and policy
optimization by extending the standard policy gradient to the optimal transport setting. Specifically, for deterministic
policies, we demonstrate that the gradient of the 1-Wasserstein distance admits a policy gradient decomposition
                                                          h                              i
                             ∇θ W (ρπθ , ρE ) = −Es∼dπθ ∇θ πθ (s)∇a f ∗ (s, a) a=π (s)                             (26)
                                                                                               θ

             ∗                                                                             ∗
where f is the optimal Kantorovich dual variable. This result reveals that ∇a f (s, a) plays a role analogous to the
advantage function Aπθ (s, a) in standard policy gradient methods, providing directional guidance for policy updates
to minimize distribution discrepancy. The negative sign reflects that we are minimizing the Wasserstein distance,
in contrast to maximizing expected return in conventional RL. This theoretical insight enables reward-free policy
optimization by directly matching the agent’s state-action distribution to expert demonstrations, bypassing the need for
manual reward engineering while maintaining the benefits of policy gradient methods for stable optimization.
Theorem 3 (OT-Based Policy Gradient for Deterministic Policies). Let πθ be a deterministic policy, dπθ the discounted
state visitation distribution, and ρE the expert state-action distribution. Under appropriate regularity conditions, the
gradient of the 1-Wasserstein distance with respect to policy parameters is given by the following equation
                                                            h                            i
                               ∇θ W (ρπθ , ρE ) = −Es∼dπθ ∇θ πθ (s)∇a f ∗ (s, a) a=π (s)
                                                                                               θ

             ∗                                                                 πθ
where f (s, a) is the optimal Kantorovich dual function, ρπθ (s, a) = d (s) · δ(a − πθ (s)) for deterministic policies.

Proof. We startes from the following formulation of the discussed Kantorovich duality and entropic regularization
                                                     h                             i
                               W (ρπθ , ρE ) = sup EρE [f (s, a)] − Eρπθ [f (s, a)] .
                                                    f ∈Lip1

        ∗
Let f denote the optimal dual function achieving this supremum. Under the envelope theorem and assuming sufficient
regularity conditions for differentiation under the supremum, the gradient with respect to θ satisfies that
                                            ∇θ W (ρπθ , ρE ) = −∇θ Eρπθ [f ∗ (s, a)]
where we note that the optimal f ∗ depends on θ only through the constraint of the supremum, and this dependence
vanishes at optimality by the envelope theorem. For a deterministic policy πθ , the state-action distribution factorizes as
ρπθ (s, a) = dπθ (s) · δ(a − πθ (s)), where dπθ is the discounted state visitation distribution. Therefore, we have
                                           Eρπθ [f ∗ (s, a)] = Es∼dπθ [f ∗ (s, πθ (s))].
Applying the chain rule, ∇θ Es∼dπθ [f ∗ (s, πθ (s))] = Es∼dπθ [∇θ f ∗ (s, πθ (s))], where we have exchanged expec-
tation and gradient under appropriate regularity conditions. Now, applying the chain rule to ∇θ f ∗ (s, πθ (s)),
∇θ f ∗ (s, πθ (s)) = ∇θ πθ (s) · ∇a f ∗ (s, a) a=π (s) . Note that the term involving ∇θ f ∗ (s, a) is zero by the envelope
                                                  θ
theorem, as f ∗ is optimal with respect to the dual objective. Combining these results yields the following equation
                                                            h                                i
                             ∇θ W (ρπθ , ρE ) = −Es∼dπθ ∇θ πθ (s) · ∇a f ∗ (s, a) a=π (s) .
                                                                                               θ



4.2     DOT-PG Solution Framework

The DOT-PG algorithm is built on three core principles: distribution matching as its foundation, alternating optimization
as its methodology, and the integration of theory with practice. This design stems from understanding imitation
learning’s essence - systematically guiding the agent’s state-action distribution to progressively converge toward the
expert distribution. The algorithm employs a three-stage alternating optimization framework where each phase has clear
theoretical foundations and serves distinct functions, ensuring both theoretical soundness and practical effectiveness.

4.2.1       Three-Component Architecture
The proposed DOT-PG framework employs a composite architecture which is composed of the following networks
1. Policy Network (Actor) πθ (a|s):

            • Architecture: Multi-layer perceptron with residual connections
            • Output: Parameters of diagonal Gaussian distribution N (µθ (s), σθ (s))
            • Normalization: Layer normalization and tanh activations for output bounds


                                                                9


---

2. Dual Network (Critic) fϕ (s, a):

         • Input: Concatenated state-action vectors
         • Architecture: Spectral normalization for Lipschitz enforcement
         • Constraint: Gradient penalty with λ > 0
         • Role: Approximates optimal transport dual variable

3. Q-Network Qψ (s, a):
         • Purpose: Long-horizon value estimation
         • Training: Temporal difference learning with target networks
         • Usage: Stabilizes policy updates through advantage estimation

4.2.2    Alternating Optimization Strategy
Stage 1 - Critic Update: Given fixed policy, maximize the Kantorovich dual objective
                               max EρE [fϕ ] − Eρπ[fϕ ] − λEx̂[(|∇fϕ (x̂)|2 − 1)2 ]
                                                                                  
                                                                                                                     (27)
                                  ϕ

In the first stage of distribution measurement, based on the Kantorovich duality theory, we achieve accurate estimation
of the Wasserstein distance through a discriminator network with gradient penalty. This design not only provides a
smooth optimization landscape but also delivers meaningful gradient signals even when the distribution supports do not
overlap, fundamentally resolving the training instability issues prevalent in traditional adversarial imitation learning.
Stage 2 - Value Learning: Update Q-function using modified Bellman equation
                                  Qψ (s, a) ← fϕ (s, a) + γEs′ ∼P [Qψtarget (s′ , πθ (s′ ))]                         (28)
The second stage of long-horizon value estimation extends the distribution matching signal obtained in the previous step
to the temporal decision-making level. By using the Kantorovich dual variable as an intrinsic reward signal, we construct
a modified Bellman equation that enables the Q-function to learn the value of state-action pairs in long-term distribution
matching. This design connects instantaneous distribution matching with long-term policy optimization, while ensuring
the stability of value estimation through techniques such as target networks and clipped double Q-learning.
Stage 3 - Policy Improvement: Update policy to minimize Wasserstein distance
                                               min Es ∼ ρπ [Qψ (s, πθ (s))]                                          (29)
                                                θ

The third stage of policy optimization translates the OT policy gradient theorem into practice. Policy updates based on
the reparameterization trick not only provide low-variance gradient estimates but also maintain necessary exploration
capability through entropy regularization. This stage transforms the achievements of the first two stages into substantive
policy improvements, directly optimizing the policy to minimize the Wasserstein distance to the expert distribution.
These three stages form an organic whole through sophisticated coordination mechanisms. Distribution measurement
provides the signal foundation for value learning, value estimation offers long-term guidance for policy optimization,
and policy improvements in turn influence subsequent distribution measurements, creating a virtuous optimization cycle.
In practical implementation, we ensure training stability through differentiated update frequencies—the discriminator
updates most frequently to guarantee the accuracy of distribution measurement, value function updates follow, policy
updates are relatively cautious, while target networks track the changes of the main network at the slowest pace.

4.3     DOT-PG Algorithm and Design Rationale

A structured training procedure is implemented by integrating optimal transport theory with deep reinforcement learning,
that operates through five coordinated phases to ensure stable and efficient learning from expert demonstrations. The
DOT-PG algorithm is built around several key components that work in concert to enable effective imitation learning:
1. Dual Network (Critic): The network fϕ serves as the optimal transport dual variable, providing a measure of
distribution discrepancy between agent and expert behaviors. The gradient penalty term ensures the Lipschitz continuity
required by the Kantorovich-Rubinstein duality, enabling stable estimation of the Wasserstein distance. This approach
provides superior stability compared to weight clipping methods commonly used in Wasserstein-based algorithms.
2. Policy and Q-Networks: The policy network πθ and Q-network Qψ work together to optimize long-term behavior.
The Q-function learns to estimate cumulative future “rewards” as defined by the dual network output, while the policy


                                                             10


---

network adjusts to maximize these estimated returns. This modular design maintains theoretical consistency with
maximum entropy reinforcement learning frameworks.
3. Alternating Optimization: The algorithm is to employ a carefully sequenced update procedure where the dual network
is optimized to accurately measure distribution distance, followed by policy improvements based on these measurements.
This alternating approach prevents training instability while ensuring consistent progress. This structured approach is to
prevent training instability while ensuring consistent progress toward policy optimization.
4. Target Networks: The use of target networks for Q-value estimation (Qψtarget ) reduces the risk of divergent training
behavior that can occur when using rapidly changing value estimates in temporal difference learning. A kind of soft
update mechanism (τ -weighted averaging) is needed to ensure smooth target value transitions.

Algorithm 1 Dual Optimal Transport Policy Gradient (DOT-PG)
Require: Expert dataset DE , training steps T , batch size B
Ensure: Trained policy πθ
  Initialize: Policy network πθ , dual network fϕ , Q-network Qψ , target network Qψtarget , replay buffer B
  for t = 1 to T do
     Phase 1: Data Collection
     Execute current policy πθ in environment
     Store collected transitions (s, a, s′ ) in replay buffer B
     Phase 2: Dual Network Update
     Sample expert batch BE ∼ DE and policy batch Bπ ∼ B
     Compute interpolated points: ŝ = ϵsE + (1 − ϵ)sπ , â = ϵaE + (1 − ϵ)aπ
     Calculate gradient penalty: LGP = E[(∥∇fϕ (ŝ, â)∥2 − 1)2 ]
     Update dual network: ϕ ← ϕ − αϕ ∇ϕ (E[fϕ (s, a)] − E[fϕ (sE , aE )] + λLGP )
     Phase 3: Q-Network Update
     Compute target values: y = fϕ (s, a) + γE[Qψtarget (s′ , πθ (s′ ))]
     Update Q-network: ψ ← ψ − αψ ∇ψ E[(Qψ (s, a) − y)2 ]
     Phase 4: Policy Update
     Update policy parameters: θ ← θ + αθ ∇θ E[Qψ (s, πθ (s))]
     Phase 5: Target Network Update
     Update target network: ψtarget ← τ ψ + (1 − τ )ψtarget
  end for

To summarize, our proposed DOT-PG’s five-phase optimization delivers distinctive practical advantages through several
key characteristics. First, the algorithm ensures remarkable stability via its gradient penalty mechanism, which maintains
Lipschitz continuity without compromising gradient flow, outperforming traditional weight clipping methods. Second,
strategic batch reuse across phases maximizes data efficiency from both expert demonstrations and agent interactions,
while the modular architecture enables flexible integration of diverse network architectures within the core optimal
transport framework. Last and crucially, the implementation maintains rigorous theoretical consistency, with each
component directly corresponding to established optimal transport and reinforcement learning principles, ensuring
mathematical faithfulness while enabling effective imitation learning in continuous control domains.

4.4   Theoretical Guarantees for DOT-PG Solution

Theorem 4 (Optimal Transport Dual Convergence). Let fϕ be the dual network parameterized by ϕ, and ρπ , ρE the
agent and expert state-action distributions respectively. Under gradient penalty regularization with coefficient λ, the
sequence {ϕk } converges to the optimal Kantorovich dual variable, which is precisely expressed by
                               lim fϕk = f ∗ = arg max [EρE [f (s, a)] − Eρπ [f (s, a)]]                             (30)
                              k→∞                     f ∈Lip1

with the gradient penalty ensuring Lf -Lipschitz continuity and superior stability compared to weight clipping methods.

Proof. The regularized dual optimization problem is given by the following form under gradient penalty regularization.
                             max EρE [f (s, a)] − Eρπ [f (s, a)] − λEρ̂ [(∥∇f ∥2 − 1)2 ]
                                                                                        
                                                                                                                 (31)
                              f ∈F

where F denotes the neural network function class with sufficient expressive power (deep networks can approximate
continuous functions), ρ̂ represents the interpolation distribution, defined as uniform sampling along straight lines
between ρπ and ρE . Specifically, for pairs (s, a) ∼ ρπ and (sE , aE ) ∼ ρE , interpolation points are xϵ = ϵ(s, a) +


                                                            11


---

(1 − ϵ)(sE , aE ) with ϵ ∼ U [0, 1]. It can be clearly seen from previous discussion that this formulation combines the
Kantorovich dual objective with a gradient penalty term that enforces approximate 1-Lipschitz continuity.
Next, let us analyze the gradient penalty properties. The term λEρ̂ [(∥∇f ∥2 − 1)2 ] ensures Lipschitz continuity by
           Eρ̂ [(∥∇f ∥2 − 1)2 ] = 0 ⇒ ∥∇f ∥2 = 1 almost everywhere ⇒ |f (x) − f (y)| ≤ ∥x − y∥2 , ∀x, y.          (32)
In practice, minimizing this penalty encourages ∥∇f ∥2 ≈ 1, achieving approximate 1-Lipschitz continuity. Because
the neural networks in F are smooth functions, and the gradient penalty covers critical regions between distributions,
thus the interpolation distribution ρ̂ ensures comprehensive coverage. Compared to weight clipping, gradient penalty
preserves network capacity while providing more stable optimization by avoiding gradient vanishing/explosion issues.
Now, consider the convergence analysis for the stochastic gradient descent update for dual network parameters, i.e.,
                         ϕk+1 = ϕk + αϕ,k ∇ϕ EρE [fϕ ] − Eρπ [fϕ ] − λEρ̂ [(∥∇fϕ ∥2 − 1)2 ] .
                                                                                              
                                                               P∞                  P∞       2
The learning rates αϕ,k satisfy Robbins-Monro conditions k=1 αϕ,k = ∞,                k=1 αϕ,k < ∞. These conditions
ensure that learning rates are sufficiently small to suppress noise but large enough to guarantee convergence.
Note that the optimization objective is concave because the expectation terms are linear in f and the negative gradient
penalty term is concave (as it’s the negative of a convex function). Given sufficient capacity in F (neural networks
can represent optimal f ∗ ), stochastic approximation convergence theorems (Kushner & Yin, 2003) guarantee that
{ϕk } converges almost surely to the stationary point corresponding to f ∗ . Formally, define L(ϕ) = EρE [fϕ ] −
Eρπ [fϕ ] − λEρ̂ [(∥∇fϕ ∥2 − 1)2 ]. Since L(ϕ) is concave and Lipschitz continuous, the stochastic gradient descent
updates converge to arg max L(ϕ), which equals the Kantorovich dual solution as the gradient penalty term approaches
zero at optimum. Under gradient penalty regularization, the dual network sequence {ϕk } converges almost surely to the
optimal Kantorovich dual variable f ∗ , with gradient penalty ensuring stable and efficient optimization.
Theorem 5 (Wasserstein Distance Estimation Stability). The stable estimation of the 1-Wasserstein distance is provided
                                   W (ρπ , ρE ) = EρE [f ∗ ] − Eρπ [f ∗ ] + O(λ−1 )                               (33)
where the estimation error decays exponentially with the gradient penalty strength λ in regularized dual optimization.

Proof. The Wasserstein distance is given by the Kantorovich-Rubinstein duality theorem with 1-Wasserstein distance
                                W (ρπ , ρE ) = sup [EρE [f (s, a)] − Eρπ [f (s, a)]]                           (34)
                                                 f ∈Lip1

can be optimized over 1-Lipschitz functions, where the supremum is achieved by the optimal Kantorovich potential f ∗ .
According to the Kantorovich regularized estimation, we solve a regularized version of this optimization problem, i.e.,
                                                Ŵ = EρE [fλ∗ ] − Eρπ [fλ∗ ]                                         (35)
where fλ∗ is the solution to the gradient-penalized optimization problem which is mathematically expressed by
                         fλ∗ = arg max EρE [f (s, a)] − Eρπ [f (s, a)] − λEρ̂ [(∥∇f ∥2 − 1)2 ] .
                                                                                             
                                   f ∈F
Next, we will show that this regularization ensures practical computability while maintaining theoretical guarantees. In
fact, the estimation error |W − Ŵ | can be bounded by applying the following relaxation techniques of norm computation
          |W − Ŵ | = |(EρE [f ∗ ] − Eρπ [f ∗ ]) − (EρE [fλ∗ ] − Eρπ [fλ∗ ])| = |EρE [f ∗ − fλ∗ ] − Eρπ [f ∗ − fλ∗ ]| (36)
                               ∗     ∗             ∗     ∗         ∗       ∗         ∗     ∗             ∗     ∗
                     ≤ |EρE [f − fλ ]| + |Eρπ [f − fλ ]| ≤ ∥f − fλ ∥∞ + ∥f − fλ ∥∞ = 2∥f − fλ ∥∞                      (37)
because the expectations are taken over probability measures, the worst-case difference bounds the expected difference.
For the regularization error bound, using regularization theory for constrained optimization problems, it implies that
                                                ∥f ∗ − fλ∗ ∥∞ ≤ Cλ−1                                               (38)
where the constant C > 0 depends on the smoothness properties of the distributions ρπ and ρE , the geometric structure
of the state-action space and the capacity and architecture of the neural network function class F. This bound of
the approximation error follows from the fact that the gradient penalty term acts as a Tikhonov regularizer, and the
regularization error typically decays polynomially with the regularization parameter for well-posed problems.
Movtived by Kushner & Yin, 2003, the exponential decay ∥f ∗ − fλ∗ ∥∞ ≤ C exp(−cλ) is achieved when: the neural
network has sufficient capacity to represent the optimal Kantorovich potential; or the distributions ρπ and ρE have
bounded support and satisfy certain smoothness conditions; the optimization algorithm effectively minimizes the
gradient penalty term; the interpolation distribution ρ̂ adequately covers the region between ρπ and ρE . Combining
these results, we obtain the final error bound W (ρπ , ρE ) = EρE [fλ∗ ] − Eρπ [fλ∗ ] + O(λ−1 ) with the potential for
exponential convergence O(exp(−cλ)) under sufficient representational capacity.


                                                           12


---

Theorem 6 (Maximum Entropy Policy Optimization). The policy network πθ and Q-network Qψ maintain theoretical
consistency with maximum entropy RL. To be precise, the optimal policy satisfies the following equation
                                                          exp(Q∗ (s, a) − V ∗ (s))
                                        π ∗ (a|s) = R                                                              (39)
                                                         exp(Q∗ (s, a′ ) − V ∗ (s))da′
where Q∗ (s, a) represents the cumulative expected output of the dual network f ∗ , i.e., the immediate reward.

Proof. Considering the maximum entropy objective formulation incorporating an entropy bonus in the standard RL
                                          "∞                                  #
                                           X
                                                 t
                              J(π) = Eπ        γ (f (st , at ) + αH(π(·|st ))) ,
                                                   t=0
                                                                    R
where f (st , at ) is the immediate objective/reward, H(π(·|s)) = − π(a|s) log π(a|s)da is the policy entropy, α > 0
is the temperature parameter that control the entropy regularization and γ ∈ [0, 1] is the discount factor.
We consider the following soft Bellman equation of maximum entropy RL, defining the soft Q-function recursively, i.e.,
                                        Q(s, a) = f (s, a) + γEs′ ∼P (·|s,a) [V (s′ )],
where the soft value function V (s) represents the expected maximum entropy return from state s, which is given by
                                                       Z               
                                                                Q(s, a)
                                         V (s) = α log exp                da
                                                                  α
where α is the temperature parameter is introduced to control the exploration-exploitation tradeoff: if α → 0, then the
solution approaches a deterministic policy, and if α → ∞, the solution approaches a total uniform random policy.
To analyze the optimal policy characterization of the policy, note that it maximizes the expected return plus entropy
                                                                     Z
             ∗
            π = arg max Eπ [Q(s, a) − α log π(a|s)] = arg max π(a|s) (Q(s, a) − α log π(a|s)) da,
                        π                                            π

treating the policy optimization as a regularized expected reward maximization with the entropy term as a regularizer.
                                                                                              R
Essentially, we solve this variational optimization problem under the normalization constraint π(a|s)da = 1, that is,
                         Z                         Z                              Z            
                    L = π(a|s)Q(s, a)da − α π(a|s) log π(a|s)da + λ 1 − π(a|s)da .

Then the Euler-Lagrange equation is used from calculus of variations and take the functional derivative w.r.t. π(a|s)
                                      ∂L
                                            = Q(s, a) − α(log π(a|s) + 1) − λ = 0.
                                    ∂π(a|s)

To find the optimal policy solution, it suffices to solve the above equation Q(s, a) − α(log π(a|s) + 1) − λ = 0 with
                                                                            
                                                             Q(s, a) − λ − α
                                          π ∗ (a|s) = exp                      .
                                                                    α
To satisfy the normalization constraint, we compute the partition function by solving for the normalization constant
                                                                              Z                   
                                       Q(s, a) − λ − α
            Z                Z
                ∗                                                        λ+α                   Q(s, a)
               π (a|s)da = exp                            da = 1, exp             = exp                   da.
                                               α                           α                      α
                                                                           
Finally, recognizing that the soft value function V (s) = α log exp Q(s,a)
                                                               R
                                                                         α     da gives directly the following equation

                                             exp(Q(s, a)/α)        exp(Q(s, a) − V (s))
                            π ∗ (a|s) = R             ′      ′
                                                               =R                           .
                                            exp(Q(s, a )/α)da     exp(Q(s, a′ ) − V (s))da′
Theorem 7 (Long-term Behavior Optimization). Under the modular policy-Q network architecture, the learned policy
optimizes long-term distribution matching, which is precisely described by the following equation
                                ∇θ J(πθ ) = Es∼ρπ [∇θ DKL (πθ (·|s)∥π ∗ (·|s))] + O(ϵQ )                           (40)
where the term ϵQ represents the Q-function approximation error defined by ϵQ = maxs,a |Qψ (s, a) − Q∗ (s, a)|.


                                                               13


---

Proof. Consider the policy gradient in maximum entropy RL with ρπ as the state visitation distribution under policy πθ
                                    J(πθ ) = Es∼ρπ ,a∼πθ [Qψ (s, a) − α log πθ (a|s)]
where the policy gradient is given by ∇θ J(πθ ) = Es∼ρπ ,a∼πθ [∇θ log πθ (a|s) (Qψ (s, a) − α log πθ (a|s))].
For the optimal policy reference, from the Maximum Entropy Policy Optimization theorem, it directly implies that
                                      exp(Q∗ (s, a)/α)
                       π ∗ (a|s) = R                       , Q∗ (s, a) = α log π ∗ (a|s) + V ∗ (s)
                                     exp(Q∗ (s, a′ )/α)da′
in which Q∗ (s, a) is the optimal Q-function and V ∗ (s) = α log exp(Q∗ (s, a′ )/α)da′ is the optimal value function.
                                                                 R

Considering that Qψ (s, a) = Q∗ (s, a) + ϵQ (s, a) where ϵQ (s, a) represents the Q-function approximation error, then
                   ∇θ J(πθ ) = Es∼ρπ ,a∼πθ [∇θ log πθ (a|s) (Q∗ (s, a) + ϵQ (s, a) − α log πθ (a|s))]

Recall the definition of KL divergence is given by the equation DKL (πθ ∥π ∗ ) =REa∼πθ [log πθ (a|s) − log π ∗ (a|s)] .
Taking the gradient with respect to θ and note that Ea∼πθ [∇θ log πθ (a|s)] = ∇θ πθ (a|s)da = ∇θ 1 = 0, we have
              ∇θ DKL (πθ ∥π ∗ ) = Ea∼πθ [∇θ log πθ (a|s)(log πθ (a|s) − log π ∗ (a|s)) + ∇θ log πθ (a|s)]
                                = Ea∼πθ [∇θ log πθ (a|s)(log πθ (a|s) − log π ∗ (a|s))] .
By connecting policy gradient and KL divergence and substituting Q∗ (s, a) = α log π ∗ (a|s) + V ∗ (s), it implies that
                        ∇θ J(πθ ) = αEs∼ρπ ,a∼πθ [∇θ log πθ (a|s)(log π ∗ (a|s) − log πθ (a|s))]
                                    + Es∼ρπ ,a∼πθ [∇θ log πθ (a|s)(V ∗ (s) + ϵQ (s, a))] .
Note that αEs∼ρπ [∇θ DKL (πθ (·|s)∥π ∗ (·|s))], Ea∼πθ [∇θ log πθ (a|s)V ∗ (s)] = V ∗ (s)Ea∼πθ [∇θ log πθ (a|s)] = 0.
Therefore, we’re left with ∇θ J(πθ ) = αEs∼ρπ [∇θ DKL (πθ (·|s)∥π ∗ (·|s))] + Es∼ρπ ,a∼πθ [∇θ log πθ (a|s)ϵQ (s, a)].
In the above equation, the error term O(ϵQ ) = Es∼ρπ ,a∼πθ [∇θ log πθ (a|s)ϵQ (s, a)] represents the impact of Q-function
approximation error on the policy gradient. If ϵQ is small (accurate Q-function approximation), this term becomes
negligible. Assuming the policy gradient and Q-error are bounded, we can express this as O(ϵQ ), where ϵQ represents
the maximum Q-function approximation error, that is, ϵQ = maxs,a |Qψ (s, a) − Q∗ (s, a)|. Absorbing the constant α
into the definition (or assuming α = 1 for simplicity), it implies that the final conclusion is achieved
                               ∇θ J(πθ ) = Es∼ρπ [∇θ DKL (πθ (·|s)∥π ∗ (·|s))] + O(ϵQ ).
Theorem 8 (Alternating Optimization Stability). The carefully sequenced update procedure—dual network optimization
followed by policy improvement—ensures training stability, which is rigorously described by the following inequality
                                E[W (ρπk+1 , ρE ) − W (ρπk , ρE )] ≤ −α∥∇W ∥2 + βδk                                   (41)
where δk is the dual network approximation error at iteration k, and α, β > 0 are two positive constants.

Proof. The alternating optimization sequence is composed of two updates driven by the gradients of L(ϕ, θ), fϕk+1
                       ϕk+1 = ϕk + αϕ ∇ϕ L(ϕk , θk ), θk+1 = θk − αθ ∇θ W (ρπθk , ρE ; fϕk+1 ),
where L(ϕ, θ) serves the dual objective for estimating W (ρπ , ρE ), the dual network fϕk+1 serves the policy optimization.
By performance difference decomposition, we analyze the expected change in true Wasserstein distance, that is,
             E[W (ρπk+1 , ρE ) − W (ρπk , ρE )] = E[W (ρπk+1 , ρE ) − W (ρπk+1 , ρE ; fϕk+1 )]
                                                  |                   {z                     }
                                                           Term A: Dual approximation error for πk+1

              + E[W (ρπk+1 , ρE ; fϕk+1 ) − W (ρπk , ρE ; fϕk+1 )] + E[W (ρπk , ρE ; fϕk+1 ) − W (ρπk , ρE )] .
                |                      {z                        } |                   {z                   }
                              Term B: Policy improvement                          Term C: Dual improvement for πk

For Term A: by the dual formulation of Wasserstein distance and the approximation properties, it can be seen that
           E[W (ρπk+1 , ρE ) − W (ρπk+1 , ρE ; fϕk+1 )] ≤ E[∥f ∗ − fϕk+1 ∥∞ ] = δk , δk = ∥fϕk+1 − fθ∗k ∥∞ .
This term represents how well our dual network approximates the true optimal transport cost for the new policy.
For Term B: using the policy gradient theorem and Taylor expansion, this term is analyzed by the following equation
W (ρπk+1 , ρE ; fϕk+1 ) − W (ρπk , ρE ; fϕk+1 ) = ⟨∇θ W (ρπk , ρE ; fϕk+1 ), θk+1 − θk ⟩ + O(∥θk+1 − θk ∥2 )
= −αθ ∥∇θ W (ρπk , ρE ; fϕk+1 )∥2 + O(αθ2 ) ≤ −α∥∇W1 ∥2 + O(αθ2 ), α = αθ , ∥∇W ∥ = ∥∇θ W (ρπk , ρE ; fϕk+1 )∥.


                                                              14


---

For Term C: this term captures how the dual network update affects the Wasserstein estimate for the previous policy
                           E[W (ρπk , ρE ; fϕk+1 ) − W (ρπk , ρE )] ≤ −γE[∥fϕk+1 − fϕk ∥2 ] ≤ 0.
Then, by summing up all terms together, the following inequality holds which estimates E[W (ρπk+1 , ρE )−W (ρπk , ρE )]
          E[W (ρπk+1 , ρE ) − W (ρπk , ρE )] ≤ δk + (−α∥∇W ∥2 + O(αθ2 )) ≤ −α∥∇W ∥2 + βδk + O(αθ2 ).

For error propagation analysis, by using the contraction property of the dual update, we analyze the term δk+1 as below
                    δk+1 = ∥fϕk+2 − fθ∗k+1 ∥∞ ≤ ∥fϕk+2 − fθ∗k ∥∞ + ∥fθ∗k − fθ∗k+1 ∥∞
                       ≤ (1 − cαϕ )δk + L∥θk+1 − θk ∥ + O(αϕ2 ) ≤ (1 − cαϕ )δk + O(αθ ) + O(αϕ2 )
where c > 0 is the contraction rate, L is the Lipschitz constant of the optimal dual function with policy parameters.
What’s more, under appropriate learning rates satisfying: αθ is small enough to make O(αθ2 ) negligible, αϕ is chosen
such that (1 − cαϕ ) < 1 and the sequence {δk } converges to 0, we obtain the following monotonic improvement
                                       lim E[W1 (ρπk+1 , ρE ) − W1 (ρπk , ρE )] ≤ 0.
                                      k→∞

Theorem 9 (Two-Time-Scale Convergence). Under Robbins-Monro conditions for learning rates αϕ ≫ αθ , the
alternating optimization converges, limk→∞ ∥∇θ W (ρπθk , ρE )∥ = 0 almost surely, where the critic learning system
is designed to be faster than the actor system to ensure stable policy updates.

Proof. We consider the coupled stochastic approximation system which is composed of updated parameters θk and ϕk
                                       θk+1 = θk − αθ,k ĝk , ϕk+1 = ϕk + αϕ,k ĥk
where ĝk = ∇θ W (ρπθk , ρE ; ϕk ) + Mθ,k , ĥk = ∇ϕ L(ϕk , θk ) + Mϕ,k , Mθ,k , Mϕ,k are martingale difference sequences.
For learning rate conditions and time-scale separation, consider the Robbins-Monro conditions on learning rate, i.e.,
                 ∞                  ∞                 ∞                 ∞
                 X                  X
                                           2
                                                      X                 X
                                                                               2                αθ,k
                       αθ,k = ∞,          αθ,k < ∞,         αϕ,k = ∞,         αϕ,k < ∞, lim            =0
                                                                                          k→∞ αϕ,k
                 k=0                k=0               k=0               k=0

is imposed. Here, this time-scale separation means that the critic (ϕ) evolves on a faster time-scale than the actor (θ).
Fast Subsystem Analysis (Critic Convergence) For fixed θ, consider the critic ODE, i.e., dϕ  dt = ∇ϕ L(ϕ, θ). Assume
                                                                                     ∗
that for each fixed θ, L(ϕ, θ) is strongly concave in ϕ with unique maximizer ϕ (θ) and the stochastic gradient ĥk
satisfies: E[ĥk |Fk ] = ∇ϕ L(ϕk , θk ), E[∥ĥk ∥2 |Fk ] ≤ C(1 + ∥ϕk ∥2 + ∥θk ∥2 ). By Theorem 2.2 of Borkar (2008) on
two-time-scale stochastic approximation, for fixed θ, the fast subsystem converges: limk→∞ ∥ϕk − ϕ∗ (θ)∥ = 0 a.s..
Slow Subsystem Analysis (Actor Dynamics) On the slow time-scale, the actor update can be written as follows
                                θk+1 = θk − αθ,k [∇θ W1 (ρπθk , ρE ; ϕ∗ (θk )) + ϵk + Mθ,k ]
where ϵk = ∇θ W1 (ρπθk , ρE ; ϕk ) − ∇θ W (ρπθk , ρE ; ϕ∗ (θk )) is the critic approximation error. From the fast subsystem
convergence and the Lipschitz continuity of ∇θ W , then it can be immediately derived that limk→∞ ∥ϵk ∥ = 0 a.s..
                                                                                                                      ∗
Consider an associated ODE of the stability, the limiting ODE for the slow subsystem is dθ dt = −∇θ W (ρπθ , ρE ; ϕ (θ)).
Assume that the function V (θ) = W (ρπθ , ρE ) is continuously differentiable and serves as a Lyapunov function for the
ODE. Compute the time derivative: dV (θ(t))
                                        dt   = −⟨∇θ W (ρπθ , ρE ), ∇θ W (ρπθ , ρE ; ϕ∗ (θ))⟩. For the critic is sufficiently
accurate such that: ⟨∇θ W1 (ρπθ , ρE ), ∇θ W1 (ρπθ , ρE ; ϕ∗ (θ))⟩ ≥ c∥∇θ W1 (ρπθ , ρE )∥2 for some c > 0. This ensures:
                                        dV (θ(t))
                                                  ≤ −c∥∇θ W (ρπθ , ρE )∥2 ≤ 0.
                                           dt
Next, let us perform the martingale noise analysis, note that the martingale difference sequence Mθ,k satisfies:
E[Mθ,k |Fk ] = 0, E[∥Mθ,k ∥2 |Fk ] ≤ K(1+∥θk ∥2 ). By the martingale convergence theorem and the square-summability
of αθ,k , it can be seen from the above conclusions that the cumulative noise effect vanishes.
Fanally, we apply the Kushner-Clark lemma (Theorem 2.3.1 in Kushner & Yin, 2003) for two-time-scale systems, at
the same time, under the stated assumptions and learning rate conditions, the iterates (θk , ϕk ) converge almost surely
to the set: {(θ, ϕ) : ϕ = ϕ∗ (θ), ∇θ W (ρπθ , ρE ; ϕ∗ (θ)) = 0}. Moreover, by the accuracy of the critic approximation,
∇θ W (ρπθ , ρE ; ϕ∗ (θ)) = 0, which means that ∇θ W (ρπθ , ρE ) = 0. Therefore, we conclude that
                                       lim ∥∇θ W (ρπθk , ρE )∥ = 0 almost surely.
                                      k→∞


                                                             15


---

Theorem 10 (Target Network Stabilization). The use of target networks Qψtarget with soft update parameter τ ensures
bounded temporal difference error, the norm bound of which is generally estimated by the following inequality
                                                                    C
                                         E[∥Qψ − Q∗ ∥] ≤                     + O(τ −1 )                                  (42)
                                                               1 − γ(1 − τ )
where the constant C depends on the Lipschitz constants of the value functions and policy.

Proof. Let us define Q∗ as Optimal Q-function, Qk as current Q-function estimate at iteration k, Qktarget as target
Q-function at iteration k, τ as soft update parameter (0 < τ ≪ 1) and γ as discount factor as (0 ≤ γ < 1). Then,
                             Qk+1 = Qk + αδt ∇Qk (st , at ), Qk+1         k           k
                                                              target = τ Q + (1 − τ )Qtarget

are the update equations are receptively used for Q-function and target network. The TD error with target network is:
                    δt = f (st , at ) + γQktarget (st+1 , at+1 ) − Qk (st , at ) = (T ∗ Qktarget − Qk ) + ϵapprox
where T ∗ is the Bellman optimality operator and ϵapprox represents the related function approximation error. Define the
estimation errors: ek = Qk − Q∗ (current network error) and ektarget = Qktarget − Q∗ (target network error). The Q-update
is expressed as: Qk+1 = Qk + α(T ∗ Qktarget − Qk + ϵk ) where ϵk encompasses both approximation and sampling errors.

          ek+1 = (1 − α)ek + α(T ∗ Qktarget − Q∗ ) + αϵk = (1 − α)ek + α(T ∗ (Q∗ + ektarget ) − Q∗ ) + αϵk .
From the contraction property of T ∗ : ∥T ∗ Q − T ∗ Q′ ∥ ≤ γ∥Q − Q′ ∥, ∥ek+1 ∥ ≤ (1 − α)∥ek ∥ + αγ∥ektarget ∥ + α∥ϵk ∥.
The target network update gives ek+1          k             k       ∗        k            k
                                  target = τ Q + (1 − τ )Qtarget − Q = τ e + (1 − τ )etarget , which is considered by
                                 k+1  
                                                                  ∥ek ∥
                                                                                  
                                 ∥e       ∥     1−α       αγ                   α∥ϵk ∥
                                             ≤                              +           .
                                 ∥ek+1
                                    target ∥      τ      1 − τ ∥ektarget ∥       0
                       
           1−α      αγ
Let A =                   . The spectral radius ρ(A) determines stability, whose characteristic equation is given by
             τ    1−τ

                                     λ2 − (2 − α − τ )λ + (1 − α)(1 − τ ) − αγτ = 0.
The eigenvalues satisfy |λ| < 1 when (1 − α)(1 − τ ) − αγτ < 1 which holds for 0 < α, τ < 1 and γ < 1.
Assuming the approximation error is bounded (∥ϵk ∥ ≤ ϵ), the steady-state solution satisfies the following inequality
                                    
                            e∞          αϵ                       αϵ                       ϵ
                (I − A)             ≤      , e∞ ≤                              =                    .
                         etarget,∞      0            α(1 − γ) − αγτ + ατ          1 − γ + γτ + τ
                                              ϵ            −1
For small τ , we can approximate: e∞ ≤ 1−γ(1−τ    ) + O(τ     ). The constant C depends on the Lipschitz constant LQ
of the Q-function, the Lipschitz constant Lπ of the policy and the Maximum approximation error ϵ, i.e., C = LQ Lπ ϵ.
                                                     LQ Lπ ϵ                     α
                                E[∥Qψ − Q∗ ∥] ≤                   + O(τ −1 ) + O     .
                                                  1 − γ(1 − τ )                   τ
The O(τ −1 ) term arises when τ is very small, while the O(α/τ ) term is the coupling between learning rates.
Theorem 11 (Smooth Value Transition). The τ -weighted averaging mechanism provides smooth target value transitions:
                                            ∥Qk+1        k               k    k
                                              ψtarget − Qψtarget ∥ ≤ τ ∥Qψ − Qψtarget ∥                                  (43)
which is able to prevent the divergent training behavior from rapidly changing value estimates.

Proof. The target network update with soft update parameter τ is Qk+1          k            k
                                                                  ψtarget = τ Qψ + (1 − τ )Qψtarget , 0 < τ ≪ 1. Then,

                      ∥Qk+1        k               k            k          k               k    k
                        ψtarget − Qψtarget ∥ = ∥τ Qψ + (1 − τ )Qψtarget − Qψtarget ∥ = τ ∥Qψ − Qψtarget ∥.

Theorem 12 (Transfer Performance Bound). The performance degradation is estimated by the following expression
                                                                       2Lf Lπ
                              |W (ρreal           sim
                                   π , ρE ) − W (ρπ , ρE )| ≤                  · E[TV(Preal , Psim )]                    (44)
                                                                      (1 − γ)2
                                                     ′                ′
where TV(Preal , Psim )(s, a) = 21
                                     P
                                      s′ ∈S |Preal (s |s, a) − Psim (s |s, a)| represents the expected dynamics model error.


                                                                 16


---

Proof. By using the triangle inequality for Wasserstein distance: |W (ρreal           sim             real sim
                                                                       π , ρE ) − W (ρπ , ρE )| ≤ W (ρπ , ρπ ) and
                                                  ∞
                                                  X                                           ∞
                                                                                              X
                                ρreal
                                 π = (1 − γ)            γ t ρreal,t
                                                             π      ,    ρsim
                                                                          π = (1 − γ)                 γ t ρsim,t
                                                                                                           π
                                                  t=0                                         t=0

is the discounted state distributions where ρtπ is the state distribution at time t. By the convexity of Wasserstein distance
                                                                        ∞
                                                                        X
                                     W (ρreal sim
                                         π , ρπ ) ≤ (1 − γ)                   γ t W (ρreal,t
                                                                                      π      , ρsim,t
                                                                                                π     ).
                                                                        t=0

For any state s, it can be seen from the above facts that the next-state distribution difference is bounded by the inequality
                                W (Preal (·|s, π(s)), Psim (·|s, π(s))) ≤ Lπ · TV(Preal , Psim )
where Lπ is the Lipschitz constant of the policy. The distribution divergence evolves with the following upper-bound

                          W (ρreal,t+1
                              π        , ρsim,t+1
                                          π       ) ≤ Lπ · E[TV(Preal , Psim )] + γW (ρreal,t
                                                                                       π      , ρsim,t
                                                                                                 π     ).
                                                                                                  t
Solving the recurrence relation W (ρreal,t
                                    π      , ρsim,t
                                              π     ) ≤ Lπ · E[TV(Preal , Psim )] · 1−γ
                                                                                    1−γ . and substituting it gives the equation

                                                             ∞
                        Lπ                                  X                          Lπ γ
    W (ρreal sim
        π , ρπ ) ≤         · E[TV(Preal , Psim )] · (1 − γ)     γ t (1 − γ t ) =        2 (1 + γ)
                                                                                                  · E[TV(Preal , Psim ).
                       1−γ                                  t=0
                                                                                 (1 − γ)

By absorbing constants and incorporating the Lipschitz constant Lf of the cost function, it can be seen that
                                                                          2Lf Lπ
                             |W (ρreal           sim
                                  π , ρE ) − W (ρπ , ρE )| ≤                      · E[TV(Preal , Psim )].
                                                                         (1 − γ)2
Theorem 13 (Monotonic Policy Improvement). Each policy update ensures consistent performance enhancement:
                    W (ρπk , ρE ) − W (ρπk+1 , ρE ) ≥ E[AOT (s, πk+1 (s))] − 2Lf · W1 (ρπk , ρπk+1 )                       (45)
where AOT (s, a) = fk∗ (s, a) − Ea′ ∼πk (·|s) [fk∗ (s, a′ )] is the OT-based advantage function, avoiding oscillatory behavior.

Proof. We analyze the change in Wasserstein distance between consecutive policies, denoted by the following form
                                           ∆W = W (ρπk , ρE ) − W (ρπk+1 , ρE ).
By applying the Kantorovich duality for 1-Wasserstein distance W (ρπ , ρE ) = sup∥f ∥L ≤1 [EρE [f (s)] − Eρπ [f (s)]] .
Let fk∗ be the optimal Lipschitz function achieving this supremum for policy πk . By the optimality of fk+1
                                                                                                        ∗
                                                                                                            for πk+1 :

                                          W (ρπk+1 , ρE ) ≥ EρE [fk∗ ] − Eρπk+1 [fk∗ ].

Therefore, we obtain the key inequality: ∆W ≥ Eρπk+1 [fk∗ ] − Eρπk [fk∗ ]. Considering the Lipschitz property of fk∗ and
the definition of Wasserstein distance, it can be directly derived that |Eρπk+1 [fk∗ ] − Eρπk [fk∗ ]| ≤ Lf · W (ρπk , ρπk+1 ).
Consider that AOT also satisfies the Lipschitz property |AOT (s, πk+1 (s)) − AOT (s′ , πk+1 (s′ ))| ≤ Lf · ∥s − s′ ∥, then

                  Es∼ρπk [AOT (s, πk+1 (s))] − Es∼ρπk+1 [AOT (s, πk+1 (s))] ≤ Lf · W1 (ρπk , ρπk+1 ),

                  Eρπk+1 [fk∗ ] − Eρπk [fk∗ ] − Es∼ρπk [AOT (s, πk+1 (s))] ≤ 2Lf · W1 (ρπk , ρπk+1 ).

Combining all inequalities gives W1 (ρπk , ρE ) − W1 (ρπk+1 , ρE ) ≥ E[AOT (s, πk+1 (s))] − 2Lf W1 (ρπk , ρπk+1 ).

Theorem 14 (End-to-End Convergence). The complete DOT-PG framework with all five coordinated phases converges
to an ϵ-optimal policy:
                            lim sup W (ρπk , ρE ) ≤ ϵdual + ϵpolicy + ϵvalue + ϵtarget                   (46)
                                    k→∞

where each ϵ term represents the approximation error from the corresponding algorithmic component.


                                                                    17


---

Proof. Let us define Ŵ k as the estimated Wasserstein distance using current dual network fϕk , Ŵ k,∗ as the optimal
Wasserstein distance achievable with perfect dual for policy πk , and W ∗ as the global minimum Wasserstein distance
over all possible policies. Then, we analyze the the total error by decomposing it into several parts as follows
                     W (ρπk , ρE ) = W (ρπk , ρE ) − Ŵ k + Ŵ k − Ŵ k,∗ + Ŵ k,∗ − W ∗ + |{z}
                                                                                           W∗
                                     |        {z        } |      {z     } |       {z   }                .
                                               ϵdual             ϵpolicy          ϵvalue      ϵtarget

Note that ϵdual is the bound of the Dual Network Approximation Error which rises from imperfect estimation of the
optimal transport cost and is bounded by Theorem 1 (Maximum Entropy Policy Optimization): ϵdual ≤ ∥fϕk − f ∗ ∥∞ ,
and converges to zero due to alternating optimization (Theorem 5).
Recall that ϵpolicy is the bound of the Policy Optimization Error which measures suboptimality of current policy given
current dual estimate and is bounded by monotonic improvement guarantee (Theorem 10): ϵpolicy ≤ E[AOT (s, πk (s))]−
2Lf W (ρπk−1 , ρπk ), and vanishes due to two-time-scale convergence (Theorem 6).
Notice that ϵvalue is the bound of the Value Function Approximation Error which comes from imperfect Q-function
                                                                                      C            −1
estimation and is controlled by target network stabilization (Theorem 7): ϵvalue ≤ 1−γ(1−τ ) + O(τ    ).
Recall that ϵtarget is the bound of the irreducible Approximation Error which represents fundamental limitations of
policy class and function approximation, and cannot be eliminated but can be minimized through architecture design.
In the proposed coordinated phases, the convergence mechanism is introduced by: 1. Dual Network Phase: ϵdual → 0
through consistent estimation; 2. Policy Improvement Phase: ϵpolicy → 0 via monotonic updates; 3. Value Learning
Phase: ϵvalue → 0 with temporal difference learning; 4. Target Network Phase: Stabilizes all components; 5. Transfer
Learning Phase: Maintains robustness to distribution shift.
By the triangle inequality and individual convergence, lim supk→∞ W (ρπk , ρE ) ≤ ϵdual + ϵpolicy + ϵvalue + ϵtarget . Each
term can be made arbitrarily small with sufficient network capacity, training iterations, and suitable hyperparameters.
Theorem 15 (Sample Complexity). For ϵ > 0, DOT-PG requires at most:
                                                                    !
                                                  L2f L2π         1
                                     N (ϵ) = O              · log                                                      (47)
                                                (1 − γ)4 ϵ2       δ

samples to achieve W1 (ρπ , ρE ) ≤ ϵ with probability at least 1 − δ.

Proof. The sample complexity bound is established through systematic error analysis. The proof considers four primary
error sources, each with distinct characteristics and theoretical foundations supported by established literature Bayraktar,
E., Eckstein, S., & Zhang, X. (2025), and Fournier, N., & Guillin, A. (2015).
                                                                                                              ∗
Error Source Analysis: dual√ network estimation error when estimates the optimal transport cost function f scales as
E[∥fϕ −f ∗ ∥∞ ] ≤ O(Lf / ndual ) which are mainly based on the statistical learning theory for Lipschitz function classes
in Wainwright, M. J. (2019). During the optimization, the policy gradient estimator has the variance which is bounded
by the inequality Var[k ] ≤ O(L2π /(1−γ)2 npolicy ) due to cumulative return variance over horizon 1/(1−γ) Agarwal,
                                                                                                                √      A.,
                                                                                                    ∗
Kakade, S. M., Lee, J. D., & Mahajan, G. (2021). Q-function approximation
                                                                        √     error obeys E[∥Q ψ −Q   ∥] ≤ O(1/   n value ),
while temporal difference amplification yields ϵTD ≤ O(1/(1 − γ) nTD ) from bootstrapping in value estimation,
according to Sutton, R. S., & Barto, A. G. (1998).
Lipschitz and Discount Dependencies: The product Lf Lπ emerges from error propagation: policy changes affect state
distributions (∥ρπ − ρπ′ ∥ ≤ Lπ ∥π − π ′ ∥) which propagate through transport costs (|f (ρ) − f (ρ′ )| ≤ Lf ∥ρ − ρ′ ∥),
yielding |W1 (ρπ , ρE ) − W1 (ρπ′ , ρE )| ≤ Lf Lπ · E[∥π − π ′ ∥] via the performance difference lemma Kakade, S., &
Langford, J. (2002). The (1 − γ)−4 dependence aggregates four factors: policy gradient variance O((1 − γ)−2 ) from
horizon summation, TD error O((1 − γ)−1 ) from Bellman contraction, distribution shift accumulation O((1 − γ)−1 )
                          Pt−1
from W1 (ρtπ , ρtπ′ ) ≤ Lπ k=0 γ k ∥π − π ′ ∥, and their worst-case composition.
                                                                                                              S4
High-Probability Guarantee: Applying Hoeffding’s inequality and union bound over error events Ei : P( i=1 Ei ) ≤
                                                                                   2 2
                                                                                   CLf Lπ
4 exp −c · n · ϵ2 /(L2f L2π (1 − γ)−4 ) . Setting this ≤ δ and solving yields n ≥ (1−γ)4 ϵ2 ·log(4/δ). Policy optimization

dominates complexity    due to nested loops, on-policy sampling, and high-dimensional optimization, giving the final
                         L2f L2π
                                       
                                       1
bound N (ϵ, δ) = O     (1−γ)4 ϵ2 · log δ   .
The bound justifies Lipschitz regularization for sample efficiency and highlights exponential horizon dependence. For
stochastic environments, an additional log |S||A| term accounts for state-action space size. This analysis completes the


                                                            18


---

sample complexity proof while demonstrating alignment with established OT and RL theory Genevay, A., Peyré, G., &
Cuturi, M. (2018, March) and Peyré, G., & Cuturi, M. (2019).




                                                       19


---

7   Conclusion
Your conclusion here

Acknowledgments
This was was supported in part by......

References
[1] George Kour and Raid Saabne. Real-time segmentation of on-line handwritten arabic script. In Frontiers in
    Handwriting Recognition (ICFHR), 2014 14th International Conference on, pages 417–422. IEEE, 2014.
[2] George Kour and Raid Saabne. Fast classification of handwritten on-line arabic characters. In Soft Computing and
    Pattern Recognition (SoCPaR), 2014 6th International Conference of, pages 312–318. IEEE, 2014.
[3] Guy Hadash, Einat Kermany, Boaz Carmeli, Ofer Lavi, George Kour, and Alon Jacovi. Estimate and replace: A
    novel approach to integrating deep neural networks with existing applications. arXiv preprint arXiv:1804.09028,
    2018.




                                                        25


---
