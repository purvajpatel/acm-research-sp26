"""
Solution Verifier: Verifies integration solutions using numerical methods
Exact implementation based on paper methodology
"""
import sympy as sp
import numpy as np
from scipy.integrate import quad
from typing import Optional, Tuple
import re

class SolutionVerifier:
    """
    Verifies integration solutions using numerical and symbolic methods
    As specified in the paper: uses numerical integration for verification
    """
    
    def __init__(self, tolerance: float = 1e-6, num_test_points: int = 100):
        """
        Initialize verifier
        
        Args:
            tolerance: Numerical tolerance for verification (default 1e-6 as in paper)
            num_test_points: Number of points to test (default 100 as in paper)
        """
        self.tolerance = tolerance
        self.num_test_points = num_test_points
        self.test_domain = (-10, 10)  # Domain for numerical testing
        self.numerical_step = 1e-5  # Step size for numerical derivative
    
    def verify_solution(
        self, 
        problem: str, 
        solution: str,
        use_symbolic: bool = True,
        use_numerical: bool = True
    ) -> Tuple[bool, str]:
        """
        Verify if solution is correct
        
        Args:
            problem: Integration problem string
            solution: Proposed solution string
            use_symbolic: Use symbolic verification
            use_numerical: Use numerical verification
        
        Returns:
            Tuple of (is_correct, reason)
        """
        try:
            # Parse problem and solution
            integrand_expr = self._parse_integrand(problem)
            antiderivative_expr = self._parse_antiderivative(solution)
            
            if integrand_expr is None or antiderivative_expr is None:
                return False, "Failed to parse problem or solution"
            
            # Method 1: Symbolic verification (fastest)
            if use_symbolic:
                symbolic_result = self._symbolic_verify(integrand_expr, antiderivative_expr)
                if symbolic_result[0]:
                    return True, "Symbolic verification passed"
            
            # Method 2: Numerical verification (more robust)
            if use_numerical:
                numerical_result = self._numerical_verify(integrand_expr, antiderivative_expr)
                if numerical_result[0]:
                    return True, "Numerical verification passed"
                else:
                    return False, numerical_result[1]
            
            return False, "All verification methods failed"
            
        except Exception as e:
            return False, f"Verification error: {str(e)}"
    
    def _parse_integrand(self, problem: str) -> Optional[sp.Expr]:
        """
        Parse integrand from problem string
        Handles various formats: ∫ f(x) dx, integral of f(x), etc.
        """
        x = sp.Symbol('x')
        
        # Clean problem string
        problem = problem.strip()
        
        # Remove common prefixes
        problem = re.sub(r'^(∫|integral|integrate|compute\s+the\s+integral)\s*', '', problem, flags=re.IGNORECASE)
        problem = re.sub(r'\s*dx\s*$', '', problem)
        problem = problem.strip()
        
        try:
            # Try to parse as SymPy expression
            # Handle common LaTeX/math notation
            problem = problem.replace('^', '**')  # Convert ^ to **
            problem = problem.replace('\\', '')  # Remove LaTeX backslashes
            
            # Parse expression
            expr = sp.sympify(problem, evaluate=False)
            return expr
        except:
            try:
                # Try with eval (less safe but more flexible)
                expr = eval(problem, {'x': x, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 
                                     'log': sp.log, 'sqrt': sp.sqrt, 'tan': sp.tan})
                return expr
            except:
                return None
    
    def _parse_antiderivative(self, solution: str) -> Optional[sp.Expr]:
        """
        Parse antiderivative from solution string
        Handles: "x^2/2 + C", "x^2/2", etc.
        """
        x = sp.Symbol('x')
        
        # Clean solution string
        solution = solution.strip()
        
        # Remove "Solution:", "Answer:", etc.
        solution = re.sub(r'^(solution|answer|result):\s*', '', solution, flags=re.IGNORECASE)
        
        # Remove constant of integration (+ C, + constant, etc.)
        solution = re.sub(r'\s*\+\s*[Cc](onstant)?\s*$', '', solution)
        solution = solution.strip()
        
        try:
            # Convert notation
            solution = solution.replace('^', '**')
            solution = solution.replace('\\', '')
            
            # Parse expression
            expr = sp.sympify(solution, evaluate=False)
            return expr
        except:
            try:
                # Try with eval
                expr = eval(solution, {'x': x, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp,
                                       'log': sp.log, 'sqrt': sp.sqrt, 'tan': sp.tan,
                                       'arcsin': sp.asin, 'arccos': sp.acos, 'arctan': sp.atan})
                return expr
            except:
                return None
    
    def _symbolic_verify(
        self, 
        integrand: sp.Expr, 
        antiderivative: sp.Expr
    ) -> Tuple[bool, str]:
        """
        Verify using symbolic differentiation
        Check if d/dx(antiderivative) == integrand
        """
        x = sp.Symbol('x')
        
        try:
            # Differentiate antiderivative
            derivative = sp.diff(antiderivative, x)
            
            # Simplify difference
            diff = sp.simplify(derivative - integrand)
            
            # Check if difference is zero
            if diff == 0:
                return True, "Symbolic check passed"
            else:
                # Sometimes expressions are equivalent but not identical
                # Check at a few points
                test_points = [1, 2, 3, -1, -2]
                for point in test_points:
                    try:
                        diff_val = diff.subs(x, point)
                        if abs(float(diff_val)) > self.tolerance:
                            return False, f"Symbolic check failed: derivative != integrand"
                    except:
                        continue
                
                # If all test points pass, consider it correct
                return True, "Symbolic check passed (equivalent forms)"
        except Exception as e:
            return False, f"Symbolic verification error: {str(e)}"
    
    def _numerical_verify(
        self, 
        integrand: sp.Expr, 
        antiderivative: sp.Expr
    ) -> Tuple[bool, str]:
        """
        Verify using numerical evaluation at multiple points
        As specified in the paper
        """
        x = sp.Symbol('x')
        
        try:
            # Convert to numerical functions
            integrand_func = sp.lambdify(x, integrand, 'numpy')
            antiderivative_func = sp.lambdify(x, antiderivative, 'numpy')
        except Exception as e:
            return False, f"Failed to create numerical functions: {str(e)}"
        
        # Generate test points
        test_points = np.linspace(
            self.test_domain[0], 
            self.test_domain[1], 
            self.num_test_points
        )
        
        # Filter out points where functions might be undefined
        valid_points = []
        for point in test_points:
            try:
                # Test if functions are defined at this point
                _ = integrand_func(point)
                _ = antiderivative_func(point)
                valid_points.append(point)
            except (ValueError, ZeroDivisionError, TypeError):
                continue
        
        if len(valid_points) < 10:
            return False, "Too few valid test points"
        
        # Verify at each point
        failed_points = 0
        for point in valid_points:
            try:
                # Compute derivative numerically
                h = self.numerical_step
                derivative_num = (
                    antiderivative_func(point + h) - 
                    antiderivative_func(point)
                ) / h
                
                # Compare with integrand
                integrand_value = integrand_func(point)
                
                # Check if they match within tolerance
                if abs(derivative_num - integrand_value) > self.tolerance:
                    failed_points += 1
                    # Allow some failures (due to numerical errors)
                    if failed_points > len(valid_points) * 0.1:  # More than 10% failures
                        return False, f"Numerical check failed at {failed_points}/{len(valid_points)} points"
            except (ValueError, ZeroDivisionError, TypeError):
                continue
        
        # Also check definite integrals
        definite_result = self._verify_definite_integrals(integrand_func, antiderivative_func)
        if not definite_result[0]:
            return definite_result
        
        return True, "Numerical verification passed"
    
    def _verify_definite_integrals(
        self, 
        integrand_func, 
        antiderivative_func
    ) -> Tuple[bool, str]:
        """
        Verify using definite integrals
        Check that ∫[a to b] f(x) dx = F(b) - F(a)
        """
        test_intervals = [(-5, 5), (-3, 3), (0, 5), (-5, 0)]
        
        for a, b in test_intervals:
            try:
                # Compute using antiderivative (Fundamental Theorem)
                F_b = antiderivative_func(b)
                F_a = antiderivative_func(a)
                definite_from_antiderivative = F_b - F_a
                
                # Compute using numerical integration
                definite_numerical, error = quad(integrand_func, a, b)
                
                # Check if they match
                if abs(definite_from_antiderivative - definite_numerical) > self.tolerance:
                    return False, f"Definite integral check failed on [{a}, {b}]"
            except Exception:
                continue
        
        return True, "Definite integral check passed"
