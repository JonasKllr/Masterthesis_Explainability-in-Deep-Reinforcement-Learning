import typing

from ixai.utils.wrappers.base import Wrapper


class ActionSelectionWrapper(Wrapper):
    def __init__(
            self,
            action_selection_function: typing.Union["Module", typing.Callable],
            feature_names: typing.Optional[list] = None,
            device: str = 'cpu'
            ):
        """
        Args:
            link_function (Union[torch.nn.Module, Callable]): The function linking from the model input to the output.
            device: (str): Torch device flag where the model is running. Defaults to `'cpu'`.
            feature_names (list[str], optional): A ordered list of feature names what features should be provided.
        """
        super().__init__(action_selection_function, feature_names)
        self._device: str = device

    def __call__(self, x: typing.Union[typing.List[dict], dict]) -> typing.Union[dict, typing.List[dict]]:
        """
        Args:
            x (Union[list[dict], dict]): Input features in the form of a dict (1d-input) mapping from feature names to
            feature values or a list of such dicts.

        Returns:
            (Union[list[dict], dict]): The model output as a dictionary following river conventions.
        """
        
        x_input = self.convert_1d_input_to_arr(x)
        x_input = torch.tensor(x_input, device=self._device, dtype=torch.float32)
        output = self._prediction_function(x_input).detach().cpu().numpy()
        return self.convert_arr_output_to_dict(output)
        

        # x_input = self.convert_2d_input_to_arr(x)
        # x_input = torch.tensor(x_input, device=self._device, dtype=torch.float32)
        # y_predictions = self._prediction_function(x_input).detach().cpu().numpy()
        # y_prediction = [self.convert_arr_output_to_dict(y_predictions[i]) for i in range(len(y_predictions))]
        # return y_prediction