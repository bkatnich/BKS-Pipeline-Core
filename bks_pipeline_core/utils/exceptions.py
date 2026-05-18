class TrendFetchAbortedError(Exception):
    """Raised when the circuit breaker trips due to consecutive external API failures."""

    def __init__(self, message: str, consecutive_failures: int = 0):
        super().__init__(message)
        self.consecutive_failures = consecutive_failures
