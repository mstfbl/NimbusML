# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
Models.DatasetTransformerEx
"""


from ..utils.entrypoints import EntryPoint
from ..utils.utils import try_set, unlist


def models_datasettransformerex(
        data,
        transform_model,
        output_data=None,
        **params):
    """
    **Description**
        Transform a dataset that may have a predictor model
        
    :param data: Input dataset (inputs).
    :param transform_model: Transform model (inputs).
    :param output_data: Transformed dataset (outputs).
    """

    entrypoint_name = 'Models.DatasetTransformerEx'
    inputs = {}
    outputs = {}

    if data is not None:
        inputs['Data'] = try_set(
            obj=data,
            none_acceptable=False,
            is_of_type=str)
    if transform_model is not None:
        inputs['TransformModel'] = try_set(
            obj=transform_model,
            none_acceptable=False,
            is_of_type=str)
    if output_data is not None:
        outputs['OutputData'] = try_set(
            obj=output_data,
            none_acceptable=False,
            is_of_type=str)

    input_variables = {
        x for x in unlist(inputs.values())
        if isinstance(x, str) and x.startswith("$")}
    output_variables = {
        x for x in unlist(outputs.values())
        if isinstance(x, str) and x.startswith("$")}

    entrypoint = EntryPoint(
        name=entrypoint_name, inputs=inputs, outputs=outputs,
        input_variables=input_variables,
        output_variables=output_variables)
    return entrypoint
