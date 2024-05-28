import typing

from ixai.utils.wrappers.base import Wrapper


class ActionSelectionWrapper(Wrapper):
    def __init__(
        self,
        action_selection_function: typing.Callable,
        feature_names: typing.Optional[list] = None,
    ):
        """
        Args:
            action_selection_function: (Callable): The function linking from the model input to the output.
            feature_names (list[str], optional): A ordered list of feature names what features should be provided.
        """
        super().__init__(action_selection_function, feature_names)

    def __call__(
        self, x: typing.Union[typing.List[dict], dict]
    ) -> typing.Union[dict, typing.List[dict]]:
        """
        Args:
            x (Union[list[dict], dict]): Input features in the form of a dict (1d-input) mapping from feature names to
            feature values or a list of such dicts.

        Returns:
            (Union[list[dict], dict]): The model output as a dictionary following river conventions.
        """

        x_input = self.convert_1d_input_to_arr(x)
        output = self._prediction_function(x_input)
        return self.convert_arr_output_to_dict(output)
