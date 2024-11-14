import numpy as np

class PoissonProcess:
    """Implements Poisson process"""
    def simulate(self, rate, T, N):
        """
        Simulate Poisson process
        
        Args:
            rate: Average rate of events
            T: Final time
            N: Number of steps
        
        Returns:
            Time points and process values
        """
        dt = T/N
        t = np.linspace(0, T, N)
        X = np.zeros(N)
        X[0] = 0
        for i in range(1, N):
            # Generate Poisson increment
            dN = np.random.poisson(rate * dt)
            X[i] = X[i-1] + dN
        return t, X

class WienerProcess:
    """Implements Wiener process (Brownian motion)"""
    def simulate(self, T, N):
        """
        Simulate Wiener process
        T: final time
        N: number of steps
        """
        dt = T/N
        dW = np.random.normal(0, np.sqrt(dt), N)
        W = np.cumsum(dW)
        return np.linspace(0, T, N), W

class OrnsteinUhlenbeck:
    """Implements Ornstein-Uhlenbeck process"""
    def simulate(self, params, T, N):
        """
        Simulate OU process
        params: (theta, mu, sigma) parameters
        T: final time
        N: number of steps
        """
        theta, mu, sigma = params
        dt = T/N
        t = np.linspace(0, T, N)
        X = np.zeros(N)
        X[0] = mu
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt))
            X[i] = X[i-1] + theta*(mu - X[i-1])*dt + sigma*dW
        return t, X

class GeometricBrownianMotion:
    """Implements Geometric Brownian Motion (GBM)"""
    def simulate(self, params, T, N):
        """
        Simulate GBM process
        params: (mu, sigma) drift and volatility parameters
        T: final time
        N: number of steps
        """
        mu, sigma = params
        dt = T/N
        t = np.linspace(0, T, N)
        X = np.zeros(N)
        X[0] = 1.0
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt))
            X[i] = X[i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)
        return t, X

class CoxIngersollRoss:
    """Implements Cox-Ingersoll-Ross (CIR) process"""
    def simulate(self, params, T, N):
        """
        Simulate CIR process
        params: (kappa, theta, sigma) mean reversion, long-term mean, and volatility
        T: final time
        N: number of steps
        """
        kappa, theta, sigma = params
        dt = T/N
        t = np.linspace(0, T, N)
        X = np.zeros(N)
        X[0] = theta
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt))
            X[i] = X[i-1] + kappa*(theta - X[i-1])*dt + sigma*np.sqrt(max(X[i-1], 0))*dW
        return t, X

class VasicekModel:
    """Implements Vasicek interest rate model"""
    def simulate(self, params, T, N):
        """
        Simulate Vasicek process
        params: (kappa, theta, sigma) mean reversion, long-term rate, and volatility
        T: final time
        N: number of steps
        """
        kappa, theta, sigma = params
        dt = T/N
        t = np.linspace(0, T, N)
        r = np.zeros(N)
        r[0] = theta
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt))
            r[i] = r[i-1] + kappa*(theta - r[i-1])*dt + sigma*dW
        return t, r