# AE6106

## Mathematical Concepts, Formulas, Equations, and Methods in Learning Mixed Strategies in Trajectory Games

This paper introduces novel methods for solving trajectory games, focusing on reducing computational burden and enabling more competitive mixed strategies. It leverages various mathematical concepts from game theory, optimization, and machine learning. The core contributions involve an offline training phase and a 'lifted game' formulation that allows players to optimize over multiple candidate trajectories.

### I. Introduction

- The paper begins by highlighting the limitations of traditional 'predict then plan' approaches in multi-agent settings and the computational challenges of current game-theoretic planning methods. It introduces the concept of 'mixed strategies' as a solution to non-existence issues in simple static games like rock-paper-scissors, where players choose actions based on a probability distribution rather than a single deterministic action [1].
- **Pure vs. Mixed Strategies:**
  - **Pure Strategy:** A single, deterministic action chosen by a player. In continuous trajectory games, representing mixed strategies can be difficult, often leading to regularization of player objectives to encourage pure solutions [1].
  - **Mixed Strategy:** A distribution over multiple possible actions (or trajectories in this context), allowing players to choose an action at random from this distribution [1]. The paper's approach lifts the strategy space to learn more competitive mixed strategies, represented as distributions over multiple trajectory candidates per player [2].

### III. Formulation

This section details the mathematical formulation of the trajectory game, starting with a standard representation and then introducing the paper's novel 'lifted' formulation.

#### A. Trajectory Game Formulation

- **Equation (1a) and (1b): Coupled Optimization Problems for Two Players**
  - **Equation (1a):** `OPT1(T2, X1):= arg min T1 f1 (T1, T2) s.t. T1 ∈ K1(x1)`
  - **Equation (1b):** `OPT2(T1,X2) := arg min T2 f2(T1, T2) s.t. T2 ∈ K2(x2)`
  - **Variables and Symbols:**
    - `OPT1`, `OPT2`: Optimal control problems for Player 1 and Player 2, respectively.
    - `T1`, `T2`: Decision variables representing discrete-time state-control trajectories for Player 1 and Player 2, starting from initial configurations `x1` and `x2` [3].
    - `f1(T1, T2)`, `f2(T1, T2)`: Differentiable cost functions (objectives) for Player 1 and Player 2, respectively. These functions can depend on both players' trajectories and encode preferences like goal-reaching and collision-avoidance [3].
    - `K1(x1)`, `K2(x2)`: Constraint sets for Player 1 and Player 2, respectively. These sets represent all dynamically-feasible trajectories satisfying dynamic constraints, control limits, etc. They are independent of each other's trajectory [3].
    - `arg min`: The argument that minimizes the subsequent expression.
  - **Intuition:** This formulation describes a noncooperative game where each player tries to minimize their own cost function, subject to their own constraints. The costs, however, depend on the actions of all players, making them coupled optimization problems [3].
  - **Structure:** These are standard representations of finite-horizon trajectory games, where agents plan future decisions while accounting for strategic reactions from others [3].

- **Equation (2): Nash Equilibrium Definition**
  - `T1 ∈ OPT1(T2, x1)` and `T2 ∈ OPT2(T1, x2)`
  - **Variables and Symbols:**
    - `(T1, T2)`: A pair of trajectories representing a Nash equilibrium.
  - **Intuition:** A Nash equilibrium is a state where no player can unilaterally improve their outcome by changing their strategy, assuming the other players' strategies remain fixed. In this context, it means that `T1` is optimal for Player 1 given `T2`, and `T2` is optimal for Player 2 given `T1` [4].
  - **Framework:** Nash equilibrium is a fundamental concept in game theory, encoding rational strategic play [4]. The paper notes that finding Nash equilibria can be intractable, and they may not always exist, especially in non-convex problems [5].

#### B. Lifted Trajectory Game Formulation (Reducing Run-time Computation)

- **Equation (3a) and (3b): Reformulated Coupled Optimization Problems with Reference Variables**
  - **Equation (3a):** `OPT1 (§2, X1, X2) := arg min ξ1 f1 (T1, T2)`
  - **Equation (3b):** `OPT2(§1, X1, X2) := arg min ξ2 f2(T1, T2)`
  - **Variables and Symbols:**
    - `§1`, `§2`: Auxiliary trajectory references for Player 1 and Player 2, respectively. These are the new decision variables [6].
    - `T1`, `T2`: Trajectory variables, now defined as `Ti = TRAJi(ģi, xi)` [6].
  - **Intuition:** This reformulation shifts the optimization from directly over trajectories `Ti` to optimizing over reference variables `§i`. The goal is to offload computational complexity to an offline training phase, making online computation more efficient [6]. The paper proves that no stationary points are 'lost' in this reformulation, meaning the equilibria of (1) correspond to equilibria of (3) [6].

- **Equation (4): TRAJ Function for Trajectory Generation**
  - `TRAJi (Ei, Xi) := arg min T 1/2 ||GiT - Ei||^2 + 1/2 ||HiT||^2 s.t. T ∈ Ki(xi)`
  - **Variables and Symbols:**
    - `TRAJi`: A function that generates a trajectory `T` for player `i` given a reference `§i` and initial state `xi`.
    - `Gi`: A matrix that enables `§i` to serve as a reference for `T`. For example, if `T` includes state and control variables, `Gi` could be `[0 I]` to interpret `§i` as a control reference signal [6].
    - `Hi`: A matrix for regularization. The term `1/2 ||HiT||^2` allows for regularization of the trajectory, which may be needed if the reference and constraint sets are insufficient to isolate solutions [6].
    - `T`: The trajectory.
    - `Ki(xi)`: The constraint set for player `i`.
  - **Intuition:** This is a trajectory optimization problem where the agent seeks a trajectory `T` that is 'close' to the reference `§i` (first term) while also satisfying some regularization (second term) and dynamic/physical constraints [6].
  - **Structure:** This is a quadratic program if `Gi` and `Hi` are constant and `Ki(xi)` defines linear constraints, as is the case in the tag game example [7].

- **Equation (5a) and (5b): Offline Training of Reference Generators**
  - **Equation (5a):** `GEN1(02, D) := arg min 01 1/d Σk=1 to d f1 (T1k, T2k)`
  - **Equation (5b):** `GEN2(01, D) := arg min 02 1/d Σk=1 to d f2 (T1k, T2k)`
  - **Variables and Symbols:**
    - `GEN1`, `GEN2`: Reference generators for Player 1 and Player 2.
    - `01`, `02`: Parameters of the reference generators `πθ1` and `πθ2` (e.g., weights of a multi-layer perceptron) [6].
    - `D`: A dataset of initial MPGP configurations `{(x1k, x2k)}k=1 to d` [6].
    - `T1k`, `T2k`: Trajectories corresponding to the k-th sample in the dataset.
  - **Intuition:** This describes an offline training phase where reference generators `πθi(x1, x2)` are learned. These generators map initial states `(x1, x2)` to reference variables `§i`. The training minimizes the average cost over a dataset of initial configurations [6]. This pre-training reduces online computational burden.

- **Equation (6): Trajectories as Functions of Generator Parameters**
  - `T1k = TRAJ1 (πθ1 (x1k, x2k), x1k)`
  - `T2k = TRAJ2 (πθ2 (x1k, x2k), x2k)`
  - **Intuition:** This explicitly shows how the trajectories `T1k` and `T2k` used in the offline training (Equation 5) are generated by the `TRAJ` function using the output of the reference generators `πθi` and the initial states [6].

- **Equation (7): Gradient Updates for Reference Generator Parameters**
  - `δθ1 = 1/d Σk=1 to d ∂f1/∂θ1 (TRAJ1(θ1, xk), TRAJ2(θ2, xk))`
  - `δθ2 = 1/d Σk=1 to d ∂f2/∂θ2 (TRAJ1(θ1, xk), TRAJ2(θ2, xk))`
  - **Variables and Symbols:**
    - `δθ1`, `δθ2`: Gradient updates for `θ1` and `θ2`.
    - `xk`: Shorthand for `(x1k, x2k)` [6].
  - **Intuition:** These equations describe how the parameters `θ1` and `θ2` of the reference generators are updated using simultaneous gradient descent. The gradients are computed by differentiating the player's objective functions with respect to the generator parameters [6]. This process requires differentiating through the trajectory optimization step (Equation 4), which is handled using implicit differentiation [6].
  - **Framework:** This uses gradient-based optimization, specifically simultaneous gradient descent, which is common in adversarial machine learning [6].

#### C. Lifted Trajectory Games

- **Equation (9a) and (9b): Lifted Game Formulation for Mixed Strategies**
  - **Equation (9a):** `OPT1lifted (§2, X1,X2) := arg min ξ1 L1(§1, §2)`
  - **Equation (9b):** `OPT2lifted (§1, x1, x2) := arg min ξ2 L2(§1, §2)`
  - **Variables and Symbols:**
    - `§1`, `§2`: Collections of trajectory references `(§1, ..., §n1)` and `(§2, ..., §n2)` for Player 1 and Player 2, respectively, where `n1` and `n2` are the number of trajectory candidates [8].
    - `L1`, `L2`: Loss functions for Player 1 and Player 2, which now depend on the collections of references [8].
  - **Intuition:** This is the 'lifted' game formulation. Instead of a single reference `§i`, each player now optimizes over a *collection* of `n` trajectory candidates. This allows for the representation of mixed strategies, where players choose from a distribution over these candidates [8].
  - **Connection:** If `n1 = n2 = 1`, this formulation reduces to Game (3), which is the single-reference game [8].

- **Equation (10a) - (10e): Relationships for Loss Functions in Lifted Game**
  - **Equation (10a):** `L1 = q1Aq2`, `L2 = qBq2`
  - **Equation (10b):** `Ai,j = f1(T1i, T2j)`, `Bi,j = f2(T1i, T2j)`
  - **Equation (10c):** `T1i = TRAJ1(§1i, x1), i ∈ N1`
  - **Equation (10d):** `T2j = TRAJ2(§2j, x2), j ∈ N2`
  - **Equation (10e):** `(q1, q2) = BMG(A, B)`
  - **Variables and Symbols:**
    - `q1`, `q2`: Mixed equilibrium strategies (probability distributions) for Player 1 and Player 2, respectively. These are vectors where each element represents the probability of selecting a particular trajectory candidate [8].
    - `A`, `B`: Cost matrices for Player 1 and Player 2. `Ai,j` is the cost for Player 1 if they choose trajectory `T1i` and Player 2 chooses `T2j`. Similarly for `Bi,j` [8].
    - `N1`, `N2`: Index sets `{1, ..., n1}` and `{1, ..., n2}` for the number of trajectory candidates for Player 1 and Player 2 [8].
    - `BMG(A, B)`: A function that maps cost matrices `A` and `B` to mixed equilibrium strategies `(q1, q2)` for a bimatrix game [8].
  - **Intuition:** These equations define how the loss functions `L1` and `L2` are computed in the lifted game. First, individual trajectories `T1i` and `T2j` are generated from their respective reference candidates `§1i` and `§2j` (10c, 10d). Then, these trajectories are used to form cost matrices `A` and `B` (10b), which represent the costs for all possible pairings of candidate trajectories. Finally, a bimatrix game solver (`BMG`) is used to find the mixed equilibrium strategies `q1` and `q2` (10e) based on these cost matrices. The overall loss for each player (10a) is then the expected cost given these mixed strategies [8].
  - **Framework:** The `BMG` function finds a Nash equilibrium for a bimatrix game, which is known to exist with finite support for separable games [8].

- **Equation (11): Nash Equilibrium Conditions for Bimatrix Game**
  - `(q1)T A q2 ≤ (q1)T A q2, ∀q1 ∈ ∆n1-1`
  - `(q1)T B q2 ≤ (q1)T B q2, ∀q2 ∈ ∆n2-1`
  - **Variables and Symbols:**
    - `∆k`: The k-simplex, representing the space of valid parameters for a categorical distribution over `k+1` elements [8]. This means `q1` and `q2` are probability distributions (non-negative and sum to 1).
  - **Intuition:** These are the standard conditions for a Nash equilibrium in a bimatrix game. It states that for the equilibrium strategies `(q1, q2)`, no player can improve their expected payoff by unilaterally changing their mixed strategy `q1` or `q2` [8].

### APPENDIX A. Equivalence of Game (1) and Game (3)

This appendix provides a formal proof of the equivalence between the original trajectory game (1) and the reformulated game (3).

- **Equation (12): Stationary Point Definition for Game (1)**
  - `d^T ∇T1 fi(T1, T2) ≥ 0, ∀d ∈ TK1 (T1)`
  - **Variables and Symbols:**
    - `∇T1 fi(T1, T2)`: Gradient of player `i`'s cost function with respect to `T1`.
    - `TK1 (T1)`: The set of linearized feasible directions with respect to the constraint set `K1(x1)` at `T1`. This is equivalent to the tangent cone at `T1` [9].
  - **Intuition:** This is the first-order optimality condition for a stationary point. It states that for a given trajectory `T2`, `T1` is optimal if there is no feasible direction `d` along which Player 1's cost `f1` can be decreased [9].

- **Equation (13): Tangent Cone for Constraints**
  - `TK1 (T) := {d : d^T ∇gi,j(T) ≥ 0, j ∈ IL(T), d^T ∇gi,j (T) ≤ 0, j ∈ IU(T)}`
  - **Variables and Symbols:**
    - `gi,j(T)`: The `j`-th constraint function for player `i`'s trajectory `T`.
    - `IL(T)`: Set of indices `j` for active lower bound constraints (`gi,j(T) = lbj`).
    - `IU(T)`: Set of indices `j` for active upper bound constraints (`gi,j(T) = ubj`).
  - **Intuition:** This defines the tangent cone at a feasible point `T` as the set of directions `d` that do not immediately violate any active constraints. It's used to characterize feasible deviations from `T` [9].

- **Equation (14): Stationary Point Definition for Game (3)**
  - `(∇§i Ti · d)^T ∇Ti fi (T1, T2) ≥ 0, ∀d`
  - **Variables and Symbols:**
    - `∇§i Ti`: The Jacobian of `Ti` with respect to `§i`.
    - `(∇§i Ti · d)`: The directional derivative of `TRAJi(§i, xi)` with respect to changes in `§i` in direction `d` [9].
  - **Intuition:** This is the first-order optimality condition for the reformulated game (3). It states that the cost `fi` cannot be decreased by changing the reference `§i` in any direction `d` [9].
  - **Connection:** The paper proves that conditions (14) imply (12) under certain assumptions, establishing the equivalence of stationary points between the two formulations [9].

- **Equation (15): Quadratic Program for Directional Derivative**
  - `min e 1/2 e^T Q1 e + d^T Q2 e s.t. e ∈ Ck1 (Ti)`
  - **Variables and Symbols:**
    - `e`: The directional derivative.
    - `Q1`, `Q2`: Matrices derived from the problem (e.g., `Q1 = I` and `Q2 = -I` in some cases) [9].
    - `Ck1 (Ti)`: The critical cone to the constraint set `gi(T)` at `Ti` [9].
  - **Intuition:** This quadratic program is used to define the directional derivative `e` of `TRAJi(§i, xi)` with respect to changes of `§i` in direction `d` [9].

- **Equation (18): Regularization Scheme for Game (3)**
  - `fi(T1, T2) + ||(g(§i) – ub)+ + (lb – g(§i))+||^2`
  - **Variables and Symbols:**
    - `(·)+`: `max(·, 0)`.
    - `g(§i)`: Constraint functions for the reference `§i`.
    - `ub`, `lb`: Upper and lower bounds for `g(§i)`.
  - **Intuition:** This regularization term is added to the objective function of Game (3) to eliminate stationary points where the reference `§i` is outside its feasible set `Ki(xi)`. It penalizes constraint violations of `§i`. This ensures that any stationary point found for (3) corresponds to a valid stationary point for (1) where `§i` satisfies its constraints [9].
  - **Constraint:** This regularization is exact and ensures that if `§i` is feasible, the regularization term is zero and has no effect on the un-regularized game [9].

### APPENDIX B. Differentiating Through BMG

This appendix describes how to differentiate through the BMG (Bimatrix Game) function, which is crucial for gradient-based learning.

- **Equation (19): Linear Complementarity Problem (LCP) for BMG**
  - `find P1, P2 s.t. P1 ≥ 0, A^T P2 ≥ 1, P2 ≥ 0, B^T P1 ≥ 1`
  - **Variables and Symbols:**
    - `P1`, `P2`: Variables representing the solutions to the LCP.
    - `A`, `B`: Cost matrices from the bimatrix game.
  - **Intuition:** The problem of finding `q1`, `q2` (mixed strategies) that satisfy the Nash equilibrium conditions (11) can be equivalently expressed as a Linear Complementarity Problem (LCP) [10]. Solving this LCP yields `P1` and `P2`, from which `q1` and `q2` can be derived.
  - **Framework:** The LCP is a mathematical problem that generalizes linear programming and quadratic programming.

- **Equation (20): Relationship between LCP Solution and Mixed Strategies**
  - `(q1)i = (P1)i / Σk(P1)k`
  - `(q2)i = (P2)i / Σk(P2)k`
  - **Intuition:** This shows how the mixed strategies `q1` and `q2` are obtained by normalizing the solutions `P1` and `P2` from the LCP (19). Each element of `q1` (or `q2`) represents the probability of choosing a specific trajectory candidate [10].

- **Equation (23): Derivatives of LCP Solution with Respect to Problem Data**
  - This equation provides the formulas for the derivatives of `P1` and `P2` with respect to the elements of the cost matrices `A` and `B`.
  - **Intuition:** These derivatives are essential for back-propagation in the learning process. They allow the system to calculate how changes in the cost matrices (which depend on trajectories and thus generator parameters) affect the mixed strategies, and consequently, the overall game value [10]. This enables the gradient-based optimization of the reference generators.

In summary, the paper introduces a sophisticated framework that combines trajectory optimization, game theory, and machine learning. It addresses the computational challenges of multi-agent trajectory games by introducing an offline training phase for reference generators and by lifting the strategy space to allow for competitive mixed strategies. The mathematical rigor is maintained through careful formulation of optimization problems, Nash equilibrium conditions, and the use of implicit differentiation for end-to-end learning.
