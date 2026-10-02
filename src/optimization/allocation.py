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
import pyomo.environ as pyo

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

class DeterministicAllocation:
    """Solves using point-estimate (mean) demand. Uses Pyomo with HiGHS."""
    
    def __init__(self, solver_name: str = 'appsi_highs', timeout: int = 300):
        self.solver_name = solver_name
        self.timeout = timeout
        
    def _compute_mean_demand(self, problem: AllocationProblem) -> Dict[str, float]:
        mean_demand = {z: 0.0 for z in problem.zones}
        for s, prob in problem.scenario_probs.items():
            for z in problem.zones:
                mean_demand[z] += prob * problem.scenarios[s].get(z, 0.0)
        return mean_demand

    def solve(self, problem: AllocationProblem) -> AllocationSolution:
        start_time = time.time()
        mean_demand = self._compute_mean_demand(problem)
        
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=problem.zones)
        m.J = pyo.Set(initialize=problem.depots)
        
        m.x = pyo.Var(m.I, m.J, domain=pyo.NonNegativeReals)
        m.y = pyo.Var(m.J, domain=pyo.Binary)
        m.u = pyo.Var(m.I, domain=pyo.NonNegativeReals)
        
        def obj_rule(m):
            return sum(m.u[i] for i in m.I) + problem.transport_cost_weight * sum(
                problem.travel_times.get((i, j), 9999) * m.x[i, j] for i in m.I for j in m.J
            )
        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        
        def demand_rule(m, i):
            return sum(m.x[i, j] for j in m.J) + m.u[i] >= mean_demand[i]
        m.demand_constr = pyo.Constraint(m.I, rule=demand_rule)
        
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
            unmet_demand = {i: pyo.value(m.u[i]) for i in m.I}
            obj_val = pyo.value(m.obj)
            
        return AllocationSolution(x_ij, y_j, obj_val, unmet_demand, solve_time, status)

class StochasticAllocation:
    """Sample Average Approximation with scenario-based optimization."""
    def __init__(self, solver_name: str = 'appsi_highs', timeout: int = 300):
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

class CVaRAllocation:
    """Minimizes CVaR_beta of unmet demand."""
    def __init__(self, beta: float = 0.95, solver_name: str = 'appsi_highs', timeout: int = 300):
        self.beta = beta
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
        m.total_u = pyo.Var(m.S, domain=pyo.NonNegativeReals)
        
        # CVaR auxiliary variables
        m.alpha = pyo.Var(domain=pyo.Reals)
        m.v = pyo.Var(m.S, domain=pyo.NonNegativeReals)
        
        def obj_rule(m):
            cvar = m.alpha + (1.0 / (1.0 - self.beta)) * sum(problem.scenario_probs[s] * m.v[s] for s in m.S)
            trans_cost = problem.transport_cost_weight * sum(
                problem.travel_times.get((i, j), 9999) * m.x[i, j] for i in m.I for j in m.J
            )
            return cvar + trans_cost
        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        
        def demand_rule(m, i, s):
            return sum(m.x[i, j] for j in m.J) + m.u[i, s] >= problem.scenarios[s].get(i, 0.0)
        m.demand_constr = pyo.Constraint(m.I, m.S, rule=demand_rule)
        
        def total_u_rule(m, s):
            return m.total_u[s] == sum(m.u[i, s] for i in m.I)
        m.total_u_constr = pyo.Constraint(m.S, rule=total_u_rule)
        
        def cvar_rule(m, s):
            return m.v[s] >= m.total_u[s] - m.alpha
        m.cvar_constr = pyo.Constraint(m.S, rule=cvar_rule)
        
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
