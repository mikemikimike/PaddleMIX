import paddle
import paddlenlp
from paddlenlp.transformers import PretrainedConfig
from ppdiffusers.transformers.model_utils import PretrainedModel 
from typing import Optional, Tuple, Union
# from .modeling_aria import AriaPretrainedModel
from paddlenlp.transformers.model_outputs import BaseModelOutput, BaseModelOutputWithPooling
import os
from paddle.nn import Layer, Linear, Conv2D, LayerList
from paddle.nn import functional as F
from paddle.nn.layer.norm import LayerNorm
from paddlenlp.transformers.auto.configuration import AutoConfig
from paddlemix.models.minicpm_v.modeling_navit_siglip import SigLipAttention, SigLipVisionConfig



class AriaVisionConfig(SigLipVisionConfig):
    """Configuration class for AriaVisionModel."""
    model_type = 'aria_vision_model'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class IdentityOp(paddle.nn.Layer):
    """
    An identity operation that returns the input unchanged.

    This can be used as a placeholder or to maintain architectural consistency
    when a specific operation is not needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x



class AriaVisionTransformer(SigLipAttention):
    """
    Aria Vision Transformer model based on Idefics2VisionTransformer.

    This class extends the original Idefics2VisionTransformer by removing the post-layernorm operation.
    """

    def __init__(self, config: AriaVisionConfig):
        
        super().__init__(config)
        self.post_layernorm = IdentityOp()  



class AriaVisionModel(PretrainedModel):
    """
    Aria Vision Model extends SiglipVisionModel to support pixel_mask.

    The pixel_mask is a 2D boolean tensor that indicates which pixels in the input
    image are actual content and which are padding. It has the same height and width
    as the input image, where:
    - True (1) values represent pixels from the original image
    - False (0) values represent padding pixels

    This mask helps the model focus on the relevant parts of the image during processing.
    """
    config_class = AriaVisionConfig
    main_input_name = 'pixel_values'
    _supports_sdpa = False
    def __init__(self, config: AriaVisionConfig):
        super().__init__(config)
        self.vision_model = AriaVisionTransformer(config)
        self.post_init()

    def forward(self, pixel_values: paddle.Tensor, pixel_mask: Optional[
        paddle.Tensor]=None, output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None, return_dict: Optional[
        bool]=None) ->Union[Tuple, BaseModelOutputWithPooling]:
        """
        Forward pass of the AriaVisionModel.

        Args:
            pixel_values (torch.Tensor): The pixel values of the input images.
            pixel_mask (Optional[torch.BoolTensor]): Mask for the pixel values.
            output_attentions (Optional[bool]): Whether to output attentions.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            return_dict (Optional[bool]): Whether to return a ModelOutput object.

        Returns:
            Union[Tuple, BaseModelOutputWithPooling]: The model's output.
        """
        return_dict = (return_dict if return_dict is not None else self.
            config.use_return_dict)
        patch_attention_mask = self._create_patch_attention_mask(pixel_mask)
        vit_oup = self.vision_model(pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask, output_attentions=
            output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        image_atts = self._create_image_attention_mask(patch_attention_mask)
        return vit_oup, image_atts

    def _create_patch_attention_mask(self, pixel_mask):
        if pixel_mask is None:
            return None
        patches_subgrid = pixel_mask.unfold(axis=1, size=self.vision_model.
            config.patch_size, step=self.vision_model.config.patch_size
            ).unfold(axis=2, size=self.vision_model.config.patch_size, step
            =self.vision_model.config.patch_size)
        return (patches_subgrid.sum(axis=(-1, -2)) > 0).astype(dtype='bool')

    def _create_image_attention_mask(self, patch_attention_mask):
        if patch_attention_mask is None:
            return None
        flattened_mask = patch_attention_mask.flatten(start_axis=1)
        return paddle.logical_not(x=flattened_mask)

