REGULARIZER_REGISTRY = {}

NO_REGULARIZER_ERR = 'Regularizer {} not in REGULARIZER_REGISTRY! Available regularizers are {}'

def RegisterRegularizer(regularizer_name):
    """Registers a regularizer."""

    def decorator(f):
        REGULARIZER_REGISTRY[regularizer_name] = f
        return f

    return decorator

def get_regularizer(regularizer_name):
    """Get regularizer from REGULARIZER_REGISTRY based on regularizer_name."""

    if not regularizer_name in REGULARIZER_REGISTRY:
        raise Exception(NO_REGULARIZER_ERR.format(
            regularizer_name, REGULARIZER_REGISTRY.keys()))

    regularizer = REGULARIZER_REGISTRY[regularizer_name]

    return regularizer