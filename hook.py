
import tensorflow as tf



def print_variables(variables, every_n_iter=100):

    return tf.train.LoggingTensorHook(
        variables,
        every_n_iter=every_n_iter,
        formatter=format_variable(variables))


def format_variable(keys):

    def format(values):
        result = []
        for key in keys:
                result.append(f"{key} = {values[key]}")

        try:
            return '\n - '.join(result)
        except:
            pass

    return format
