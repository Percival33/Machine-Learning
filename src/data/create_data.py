import Orange


def create_data(data):
    target_var_name = "y"
    target_var = data.domain[target_var_name]
    feature_vars = [var for var in data.domain.variables if var is not target_var]

    new_domain = Orange.data.Domain(feature_vars, target_var)
    return Orange.data.Table(new_domain, data)