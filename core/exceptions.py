class EWSEError(Exception):
    """Project base exception."""


class ValidationError(EWSEError):
    """Input data is outside the supported simulation domain."""


class PhysicsError(EWSEError):
    """A physics calculation failed or reached an invalid state."""


class RegistryError(EWSEError):
    """A simulation module cannot be registered or resolved."""


class ApiError(EWSEError):
    """The local API could not process a request."""


class FrontendError(EWSEError):
    """The desktop frontend could not build or render a scene."""


EWSEngineError = EWSEError
PhysicsComputationError = PhysicsError
UIConfigurationError = FrontendError
