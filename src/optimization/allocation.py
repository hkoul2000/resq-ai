"""Uncertainty-aware emergency resource allocation optimization.

Formulation:
- Sets: demand zones i in I, depots/facilities j in J
- Random demand D_i(omega) from Monte Carlo flood scenarios
- Decision variables: x_ij (resources from j to i), y_j (open depot j), u_i(omega) (unmet demand)
- Objectives: minimize expected unmet demand + transport cost, CVaR, chance-constrained

Solvers: OR-Tools (CP-SAT), Pyomo with HiGHS/CBC
"""

import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import linprog

try:
    import pyomo.environ as pyo
    HAS_PYOMO = True
except ImportError:
    HAS_PYOMO = False
    pyo = None

logger = logging.getLogger(__name__)

@dataclass
class AllocationProblem:
    zones: List[str]
    depots: List[str]
    capacities: Dict[str, float]
    travel_times: Dict[Tuple[str, str], float]
    scenarios: Dict[str, Dict[str, float]]
    scenario_probs: Dict[str, float]
    max_response_time: float = 60.0
    transport_cost_weight: float = 0.01

@dataclass
class AllocationSolution:
    x_ij: Dict[Tuple[str, str], float]
    y_j: Dict[str, int]
    objective_value: float
    unmet_demand: Dict[str, float]  # Expected unmet demand per zone
    solve_time: float
    status: str


def _solve_deterministic_scipy(problem: AllocationProblem) -> AllocationSolution:
    """Solves deterministic mean-demand LP via SciPy HiGHS."""
    start_time = time.time()
    zones = problem.zones
    depots = problem.depots
    n_z = len(zones)
    n_d = len(depots)

    # Compute mean demand across scenarios
    mean_demand = {z: 0.0 for z in zones}
    for s, prob in problem.scenario_probs.items():
        for z in zones:
            mean_demand[z] += prob * problem.scenarios[s].get(z, 0.0)

    # Variables: x_ij for (i, j) in I x J (n_z * n_d), u_i for i in I (n_z)
    n_vars = n_z * n_d + n_z
    c = np.zeros(n_vars, dtype=np.float64)

    # Transport cost objective coefficients
    for i_idx, z in enumerate(zones):
        for j_idx, d in enumerate(depots):
            var_idx = i_idx * n_d + j_idx
            c[var_idx] = problem.transport_cost_weight * problem.travel_times.get((z, d), 9999.0)

    # Unmet demand objective coefficients
    for i_idx in range(n_z):
        c[n_z * n_d + i_idx] = 1.0

    # Bounds: x_ij in [0, inf) or [0, 0] if travel time > max_response_time
    bounds = []
    for i_idx, z in enumerate(zones):
        for j_idx, d in enumerate(depots):
            t_ij = problem.travel_times.get((z, d), 9999.0)
            if t_ij > problem.max_response_time:
                bounds.append((0.0, 0.0))
            else:
                bounds.append((0.0, None))
    for _ in range(n_z):
        bounds.append((0.0, None))

    # Constraints:
    # 1. Demand rows: -sum_j x_ij - u_i <= -mean_demand[i]
    # 2. Capacity rows: sum_i x_ij <= capacities[j]
    n_constr = n_z + n_d
    A_ub = np.zeros((n_constr, n_vars), dtype=np.float64)
    b_ub = np.zeros(n_constr, dtype=np.float64)

    for i_idx, z in enumerate(zones):
        for j_idx in range(n_d):
            A_ub[i_idx, i_idx * n_d + j_idx] = -1.0
        A_ub[i_idx, n_z * n_d + i_idx] = -1.0
        b_ub[i_idx] = -mean_demand[z]

    for j_idx, d in enumerate(depots):
        row_idx = n_z + j_idx
        for i_idx in range(n_z):
            A_ub[row_idx, i_idx * n_d + j_idx] = 1.0
        b_ub[row_idx] = problem.capacities.get(d, 0.0)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    solve_time = time.time() - start_time

    x_ij = {}
    y_j = {}
    unmet_demand = {}
    if res.success:
        for i_idx, z in enumerate(zones):
            for j_idx, d in enumerate(depots):
                x_ij[(z, d)] = max(0.0, float(res.x[i_idx * n_d + j_idx]))
            unmet_demand[z] = max(0.0, float(res.x[n_z * n_d + i_idx]))
        for j_idx, d in enumerate(depots):
            total_depot_alloc = sum(x_ij[(z, d)] for z in zones)
            y_j[d] = 1 if total_depot_alloc > 1e-4 else 0
        obj_val = float(res.fun)
        status = "optimal"
    else:
        obj_val = 0.0
        status = "infeasible"

    return AllocationSolution(x_ij, y_j, obj_val, unmet_demand, solve_time, status)


def _solve_stochastic_scipy(problem: AllocationProblem) -> AllocationSolution:
    """Solves two-stage SAA stochastic LP via SciPy HiGHS."""
    start_time = time.time()
    zones = problem.zones
    depots = problem.depots
    scenarios = list(problem.scenarios.keys())
    n_z = len(zones)
    n_d = len(depots)
    n_s = len(scenarios)

    # Variables: x_ij (n_z * n_d) followed by u_is (n_z * n_s)
    n_vars = n_z * n_d + n_z * n_s
    c = np.zeros(n_vars, dtype=np.float64)

    for i_idx, z in enumerate(zones):
        for j_idx, d in enumerate(depots):
            var_idx = i_idx * n_d + j_idx
            c[var_idx] = problem.transport_cost_weight * problem.travel_times.get((z, d), 9999.0)

    u_offset = n_z * n_d
    for s_idx, s in enumerate(scenarios):
        prob = problem.scenario_probs.get(s, 1.0 / n_s)
        for i_idx in range(n_z):
            c[u_offset + s_idx * n_z + i_idx] = prob

    bounds = []
    for i_idx, z in enumerate(zones):
        for j_idx, d in enumerate(depots):
            t_ij = problem.travel_times.get((z, d), 9999.0)
            if t_ij > problem.max_response_time:
                bounds.append((0.0, 0.0))
            else:
                bounds.append((0.0, None))
    for _ in range(n_z * n_s):
        bounds.append((0.0, None))

    n_constr = n_s * n_z + n_d
    A_ub = np.zeros((n_constr, n_vars), dtype=np.float64)
    b_ub = np.zeros(n_constr, dtype=np.float64)

    row = 0
    for s_idx, s in enumerate(scenarios):
        scen_demand = problem.scenarios[s]
        for i_idx, z in enumerate(zones):
            for j_idx in range(n_d):
                A_ub[row, i_idx * n_d + j_idx] = -1.0
            A_ub[row, u_offset + s_idx * n_z + i_idx] = -1.0
            b_ub[row] = -scen_demand.get(z, 0.0)
            row += 1

    for j_idx, d in enumerate(depots):
        for i_idx in range(n_z):
            A_ub[row, i_idx * n_d + j_idx] = 1.0
        b_ub[row] = problem.capacities.get(d, 0.0)
        row += 1

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    solve_time = time.time() - start_time

    x_ij = {}
    y_j = {}
    unmet_demand = {}
    if res.success:
        for i_idx, z in enumerate(zones):
            for j_idx, d in enumerate(depots):
                x_ij[(z, d)] = max(0.0, float(res.x[i_idx * n_d + j_idx]))
            exp_u_z = 0.0
            for s_idx, s in enumerate(scenarios):
                prob = problem.scenario_probs.get(s, 1.0 / n_s)
                exp_u_z += prob * max(0.0, float(res.x[u_offset + s_idx * n_z + i_idx]))
            unmet_demand[z] = exp_u_z

        for j_idx, d in enumerate(depots):
            total_depot_alloc = sum(x_ij[(z, d)] for z in zones)
            y_j[d] = 1 if total_depot_alloc > 1e-4 else 0
        obj_val = float(res.fun)
        status = "optimal"
    else:
        obj_val = 0.0
        status = "infeasible"

    return AllocationSolution(x_ij, y_j, obj_val, unmet_demand, solve_time, status)


def _solve_cvar_scipy(problem: AllocationProblem, beta: float = 0.90) -> AllocationSolution:
    """Solves Rockafellar-Uryasev CVaR LP via SciPy HiGHS."""
    start_time = time.time()
    zones = problem.zones
    depots = problem.depots
    scenarios = list(problem.scenarios.keys())
    n_z = len(zones)
    n_d = len(depots)
    n_s = len(scenarios)

    # Variables: x_ij (n_z * n_d), u_is (n_z * n_s), nu (1), v_s (n_s)
    n_vars = n_z * n_d + n_z * n_s + 1 + n_s
    c = np.zeros(n_vars, dtype=np.float64)

    for i_idx, z in enumerate(zones):
        for j_idx, d in enumerate(depots):
            var_idx = i_idx * n_d + j_idx
            c[var_idx] = problem.transport_cost_weight * problem.travel_times.get((z, d), 9999.0)

    nu_idx = n_z * n_d + n_z * n_s
    v_offset = nu_idx + 1

    c[nu_idx] = 1.0
    for s_idx, s in enumerate(scenarios):
        prob = problem.scenario_probs.get(s, 1.0 / n_s)
        c[v_offset + s_idx] = prob / max(1e-4, 1.0 - beta)

    bounds = []
    for i_idx, z in enumerate(zones):
        for j_idx, d in enumerate(depots):
            t_ij = problem.travel_times.get((z, d), 9999.0)
            if t_ij > problem.max_response_time:
                bounds.append((0.0, 0.0))
            else:
                bounds.append((0.0, None))
    for _ in range(n_z * n_s):
        bounds.append((0.0, None))
    bounds.append((0.0, None))  # nu >= 0
    for _ in range(n_s):
        bounds.append((0.0, None))  # v_s >= 0

    n_constr = n_s + n_s * n_z + n_d
    A_ub = np.zeros((n_constr, n_vars), dtype=np.float64)
    b_ub = np.zeros(n_constr, dtype=np.float64)

    u_offset = n_z * n_d
    row = 0
    # CVaR surplus constraint: sum_i u_is - nu - v_s <= 0
    for s_idx in range(n_s):
        for i_idx in range(n_z):
            A_ub[row, u_offset + s_idx * n_z + i_idx] = 1.0
        A_ub[row, nu_idx] = -1.0
        A_ub[row, v_offset + s_idx] = -1.0
        b_ub[row] = 0.0
        row += 1

    # Demand constraint: -sum_j x_ij - u_is <= -d_is
    for s_idx, s in enumerate(scenarios):
        scen_demand = problem.scenarios[s]
        for i_idx, z in enumerate(zones):
            for j_idx in range(n_d):
                A_ub[row, i_idx * n_d + j_idx] = -1.0
            A_ub[row, u_offset + s_idx * n_z + i_idx] = -1.0
            b_ub[row] = -scen_demand.get(z, 0.0)
            row += 1

    # Capacity constraint: sum_i x_ij <= C_j
    for j_idx, d in enumerate(depots):
        for i_idx in range(n_z):
            A_ub[row, i_idx * n_d + j_idx] = 1.0
        b_ub[row] = problem.capacities.get(d, 0.0)
        row += 1

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    solve_time = time.time() - start_time

    x_ij = {}
    y_j = {}
    unmet_demand = {}
    if res.success:
        for i_idx, z in enumerate(zones):
            for j_idx, d in enumerate(depots):
                x_ij[(z, d)] = max(0.0, float(res.x[i_idx * n_d + j_idx]))
            exp_u_z = 0.0
            for s_idx, s in enumerate(scenarios):
                prob = problem.scenario_probs.get(s, 1.0 / n_s)
                exp_u_z += prob * max(0.0, float(res.x[u_offset + s_idx * n_z + i_idx]))
            unmet_demand[z] = exp_u_z

        for j_idx, d in enumerate(depots):
            total_depot_alloc = sum(x_ij[(z, d)] for z in zones)
            y_j[d] = 1 if total_depot_alloc > 1e-4 else 0
        obj_val = float(res.fun)
        status = "optimal"
    else:
        obj_val = 0.0
        status = "infeasible"

    return AllocationSolution(x_ij, y_j, obj_val, unmet_demand, solve_time, status)


class DeterministicAllocation:
    """Solves using point-estimate (mean) demand via HiGHS."""
    
    def __init__(self, solver_name: str = 'highs', timeout: int = 300):
        self.solver_name = solver_name
        self.timeout = timeout
        
    def solve(self, problem: AllocationProblem) -> AllocationSolution:
        return _solve_deterministic_scipy(problem)


class StochasticAllocation:
    """Sample Average Approximation with scenario-based optimization."""
    def __init__(self, solver_name: str = 'highs', timeout: int = 300):
        self.solver_name = solver_name
        self.timeout = timeout
        
    def solve(self, problem: AllocationProblem) -> AllocationSolution:
        return _solve_stochastic_scipy(problem)


class CVaRAllocation:
    """Minimizes CVaR_beta of unmet demand + transport costs."""
    def __init__(self, beta: float = 0.90, solver_name: str = 'highs', timeout: int = 300):
        self.beta = beta
        self.solver_name = solver_name
        self.timeout = timeout
        
    def solve(self, problem: AllocationProblem) -> AllocationSolution:
        return _solve_cvar_scipy(problem, beta=self.beta)

class ChanceConstrainedAllocation:
    """P(total_unmet <= epsilon) >= 1-alpha."""
    def __init__(self, epsilon: float = 100.0, alpha: float = 0.05, solver_name: str = 'appsi_highs', timeout: int = 300):
        self.epsilon = epsilon
        self.alpha = alpha
        self.solver_name = solver_name
        self.timeout = timeout
        
    def solve(self, problem: AllocationProblem) -> AllocationSolution:
        start_time = time.time()
        
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=problem.zones)
        m.J = pyo.Set(initialize=problem.depots)
        m.S = pyo.Set(initialize=problem.scenarios.keys())
        
        m.x = pyo.Var(m.I, m.J, domain=pyo.NonNegativeReals)
        m.y = pyo.Var(m.J, domain=pyo.Binary)
        m.u = pyo.Var(m.I, m.S, domain=pyo.NonNegativeReals)
        
        # Indicator variable: 1 if unmet > epsilon for scenario s
        m.z = pyo.Var(m.S, domain=pyo.Binary)
        M_val = sum(max(scen.values()) for scen in problem.scenarios.values()) * len(problem.zones)
        
        def obj_rule(m):
            exp_unmet = sum(problem.scenario_probs[s] * m.u[i, s] for i in m.I for s in m.S)
            trans_cost = problem.transport_cost_weight * sum(
                problem.travel_times.get((i, j), 9999) * m.x[i, j] for i in m.I for j in m.J
            )
            return exp_unmet + trans_cost
        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        
        def demand_rule(m, i, s):
            return sum(m.x[i, j] for j in m.J) + m.u[i, s] >= problem.scenarios[s].get(i, 0.0)
        m.demand_constr = pyo.Constraint(m.I, m.S, rule=demand_rule)
        
        def chance_rule(m, s):
            return sum(m.u[i, s] for i in m.I) <= self.epsilon + M_val * m.z[s]
        m.chance_constr = pyo.Constraint(m.S, rule=chance_rule)
        
        def chance_prob_rule(m):
            return sum(problem.scenario_probs[s] * m.z[s] for s in m.S) <= self.alpha
        m.chance_prob_constr = pyo.Constraint(rule=chance_prob_rule)
        
        def capacity_rule(m, j):
            return sum(m.x[i, j] for i in m.I) <= problem.capacities[j] * m.y[j]
        m.capacity_constr = pyo.Constraint(m.J, rule=capacity_rule)
        
        def max_time_rule(m, i, j):
            if problem.travel_times.get((i, j), 9999) > problem.max_response_time:
                return m.x[i, j] == 0
            return pyo.Constraint.Skip
        m.max_time_constr = pyo.Constraint(m.I, m.J, rule=max_time_rule)
        
        try:
            solver = pyo.SolverFactory(self.solver_name)
            if self.solver_name == 'appsi_highs':
                solver.options['time_limit'] = self.timeout
            results = solver.solve(m, tee=False)
            status = str(results.solver.status)
        except Exception as e:
            logger.error(f"Solver error: {e}")
            status = "error"
            
        solve_time = time.time() - start_time
        
        x_ij = {}
        y_j = {}
        unmet_demand = {}
        obj_val = 0.0
        
        if status in ['ok', 'optimal']:
            x_ij = {(i, j): pyo.value(m.x[i, j]) for i in m.I for j in m.J}
            y_j = {j: int(round(pyo.value(m.y[j]))) for j in m.J}
            unmet_demand = {i: sum(problem.scenario_probs[s] * pyo.value(m.u[i, s]) for s in m.S) for i in m.I}
            obj_val = pyo.value(m.obj)
            
        return AllocationSolution(x_ij, y_j, obj_val, unmet_demand, solve_time, status)

class GreedyAllocation:
    """Baseline - assign each zone to nearest open depot."""
    def solve(self, problem: AllocationProblem) -> AllocationSolution:
        start_time = time.time()
        
        mean_demand = {z: 0.0 for z in problem.zones}
        for s, prob in problem.scenario_probs.items():
            for z in problem.zones:
                mean_demand[z] += prob * problem.scenarios[s].get(z, 0.0)
                
        x_ij = {(i, j): 0.0 for i in problem.zones for j in problem.depots}
        y_j = {j: 1 for j in problem.depots}  # Open all by default for baseline
        
        capacities = problem.capacities.copy()
        
        for i in problem.zones:
            demand = mean_demand[i]
            # sort depots by travel time
            sorted_depots = sorted(problem.depots, key=lambda j: problem.travel_times.get((i, j), 9999))
            for j in sorted_depots:
                if problem.travel_times.get((i, j), 9999) > problem.max_response_time:
                    continue
                if demand <= 0:
                    break
                alloc = min(demand, capacities[j])
                x_ij[(i, j)] = alloc
                capacities[j] -= alloc
                demand -= alloc
                
        unmet_demand = {}
        for i in problem.zones:
            received = sum(x_ij[(i, j)] for j in problem.depots)
            unmet_demand[i] = max(0.0, mean_demand[i] - received)
            
        obj_val = sum(unmet_demand.values()) + problem.transport_cost_weight * sum(
            problem.travel_times.get((i, j), 9999) * x_ij[(i, j)] for i in problem.zones for j in problem.depots
        )
        
        solve_time = time.time() - start_time
        return AllocationSolution(x_ij, y_j, obj_val, unmet_demand, solve_time, "optimal")

class ProportionalAllocation:
    """Baseline - allocate proportional to demand."""
    def solve(self, problem: AllocationProblem) -> AllocationSolution:
        start_time = time.time()
        
        mean_demand = {z: 0.0 for z in problem.zones}
        total_demand = 0.0
        for s, prob in problem.scenario_probs.items():
            for z in problem.zones:
                val = prob * problem.scenarios[s].get(z, 0.0)
                mean_demand[z] += val
                total_demand += val
                
        x_ij = {(i, j): 0.0 for i in problem.zones for j in problem.depots}
        y_j = {j: 1 for j in problem.depots}
        
        if total_demand > 0:
            for j in problem.depots:
                cap = problem.capacities[j]
                for i in problem.zones:
                    if problem.travel_times.get((i, j), 9999) <= problem.max_response_time:
                        x_ij[(i, j)] = cap * (mean_demand[i] / total_demand)
                        
        unmet_demand = {}
        for i in problem.zones:
            received = sum(x_ij[(i, j)] for j in problem.depots)
            unmet_demand[i] = max(0.0, mean_demand[i] - received)
            
        obj_val = sum(unmet_demand.values()) + problem.transport_cost_weight * sum(
            problem.travel_times.get((i, j), 9999) * x_ij[(i, j)] for i in problem.zones for j in problem.depots
        )
        
        solve_time = time.time() - start_time
        return AllocationSolution(x_ij, y_j, obj_val, unmet_demand, solve_time, "optimal")

class OracleAllocation:
    """Uses ground-truth flood extent for allocation (upper bound)."""
    def __init__(self, ground_truth_scenario: str, solver_name: str = 'appsi_highs', timeout: int = 300):
        self.gt_scenario = ground_truth_scenario
        self.solver_name = solver_name
        self.timeout = timeout
        
    def solve(self, problem: AllocationProblem) -> AllocationSolution:
        # Just use DeterministicAllocation but with only the ground truth scenario
        gt_prob = AllocationProblem(
            zones=problem.zones,
            depots=problem.depots,
            capacities=problem.capacities,
            travel_times=problem.travel_times,
            scenarios={self.gt_scenario: problem.scenarios[self.gt_scenario]},
            scenario_probs={self.gt_scenario: 1.0},
            max_response_time=problem.max_response_time,
            transport_cost_weight=problem.transport_cost_weight
        )
        solver = DeterministicAllocation(self.solver_name, self.timeout)
        return solver.solve(gt_prob)
