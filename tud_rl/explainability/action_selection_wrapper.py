import typing
import numpy as np

from ixai.utils.wrappers.base import Wrapper


class ActionSelectionWrapperIPdp(Wrapper):
    def __init__(
        self,
        action_selection_function: typing.Callable,
        feature_names: typing.Optional[list] = None,
    ):
        super().__init__(action_selection_function, feature_names)

    def __call__(
        self, x: typing.Union[typing.List[dict], dict]
    ) -> typing.Union[dict, typing.List[dict]]:
        x_input = self.convert_1d_input_to_arr(x)
        output = self._prediction_function(x_input)
        return self.convert_arr_output_to_dict(output)


class ActionSelectionWrapperAlibi:
    def __init__(self, action_selection_function) -> None:
        self.action_selection_function = action_selection_function

    def __call__(self, x: np.ndarray) -> typing.Any:
        output = np.zeros((np.shape(x)[0], 1))
        for i in range(np.shape(x)[0]):
            output[i, :] = self.action_selection_function(x[i, :])
        return output
