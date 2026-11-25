"""Lazy access to the optional conic solver used by PhaseLift and PhaseMax."""

BACKEND = 'SCS'


class UnsupportedSize(ValueError):
    pass


class BackendUnavailable(RuntimeError):
    pass


def load_scs(method):
    try:
        import cvxpy
        import scs
    except ImportError as error:
        raise BackendUnavailable(f'Install tmbench[convex] to use the {method} backend') from error
    if BACKEND not in cvxpy.installed_solvers():
        raise BackendUnavailable('SCS is unavailable in this CVXPY installation')
    return cvxpy, {'cvxpy': cvxpy.__version__, 'scs': scs.__version__}
