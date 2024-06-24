# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import io
import functools
import json
import time
import types
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
import warnings
import transformers
import numpy as np
from typing import Callable, Optional, Sequence, Union
import log_utils

makedirs = functools.partial(os.makedirs, exist_ok=True)
logger = log_utils.get_logger(__name__)

def alleq(l: Sequence, f: Optional[Callable] = lambda x, y: x == y):
    """Check all arguments in a sequence are equal according to a given criterion.

    Args:
        f: A bi-variate boolean function.
        l: A list/tuple.

    Returns:
        True if everything is equal; otherwise False.
    """
    return all(f(l[0], li) for li in l[1:])

def zip_(*args: Sequence):
    """Assert sequences of same length before zipping."""
    if len(args) == 0:
        return []
    assert alleq(args, lambda x, y: len(x) == len(y))
    return zip(*args)

Numeric = Union[int, float]
def mean(*seqs: Sequence[Numeric]) -> Union[Numeric, Sequence[Numeric]]:
    singleton = len(seqs) == 1
    means = [float(np.mean(seq)) for seq in seqs]
    return means[0] if singleton else means

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            makedirs(f_dirname)
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def jdump(obj: Union[str, dict, list], f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def jdumps(obj, indent=4, default=str):
    return json.dumps(obj, indent=indent, default=default)

def stable_resize_token_embeddings_and_tokenizer(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    special_tokens_dict: dict,
):
    """Resize tokenizer and embedding together.

    For new tokens, the embedding value is the average of all old embedding vectors.
    """
    tokenizer.add_special_tokens(special_tokens_dict)
    stable_resize_token_embeddings(model, len(tokenizer))


def stable_resize_token_embeddings(model: transformers.PreTrainedModel, target_size: int, jitter_new_embeddings=False):
    num_new_tokens = target_size - model.get_input_embeddings().weight.size(0)
    model.resize_token_embeddings(target_size)

    if num_new_tokens > 0:

        @torch.inference_mode()
        def stable_init(embedding):
            embedding_data = embedding.weight.data
            embedding_avg = embedding_data[:-num_new_tokens].mean(dim=0, keepdim=True)
            embedding_data[-num_new_tokens:] = embedding_avg
            if jitter_new_embeddings:
                embedding_std = embedding_data[:-num_new_tokens].std(dim=0, keepdim=True)
                embedding_data[-num_new_tokens:] += torch.randn_like(embedding_data[-num_new_tokens:]) * embedding_std

        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        for embeddings in (input_embeddings, output_embeddings):
            stable_init(embeddings)


class staggered_object_creation(object):
    """
    Objection creation in a distributed setting could be very RAM-intensive.

    This function staggers the creation of objects on odd and even ranks, so that not all objects
    are created at once.

    Assumes local_rank == -1 means no distributed training.
    """

    def __init__(self, local_rank: int, world_size: int):
        super().__init__()
        self.local_rank = local_rank
        self.world_size = world_size

    def __enter__(self, *args, **kwargs):
        del args, kwargs
        if self.world_size > 1 and self.local_rank % 2 == 0:
            dist.barrier()
        return self

    def __exit__(self, *args, **kwargs):
        del args, kwargs
        if self.world_size > 1:
            if self.local_rank % 2 == 1:
                dist.barrier()
            dist.barrier()  # Final safety barrier.

    def __call__(self, func):
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator


def let_model_save_mem_when_zero_grad(model: torch.nn.Module):
    def new_zero_grad(self, set_to_none: bool = True) -> None:
        r"""Sets gradients of all model parameters to zero. See similar function
        under :class:`torch.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
        if getattr(self, "_is_replica", False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead."
            )

        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    # Make zero_grad `set_to_none=True` by default.
    # Need this runtime method patching, since self is used within zero_grad.
    model.zero_grad = types.MethodType(new_zero_grad, model)
    return model


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, give_rw_access=True, rank0_only=True
):
    """Collects the state dict and dump to disk."""
    now = time.perf_counter()

    # if trainer.fsdp is not None:
    if trainer.is_fsdp_enabled:
        # NOTE(rtaori): technically should be rank0_only=True (otherwise duplicates model in RAM),
        # but currently there seems to be a bug in FSDP that causes it to hang.
        # Migration to Pytorch 2 should fix this.
        # Once we migrate, we can also implement more efficient loading:
        # https://github.com/pytorch/pytorch/blob/master/torch/distributed/fsdp/api.py#L286-L295
        # NOTE(tianyi): tested on sphinx6, seems to work fine with rank0_only=False
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
        with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, cfg):
            # print(trainer.model)
            # check whether the model is trained using mixed precision

            # model = BetterTransformer.reverse(trainer.model)
            
            state_dict = trainer.model.state_dict()
            if trainer.args.should_save:
                trainer._save(output_dir, state_dict=state_dict)  # noqa

    # elif trainer.deepspeed is not None:
    elif trainer.is_deepspeed_enabled:
        # --- The stuff below is almost a copy from transformers.trainer.Trainer.save_model (transformers==4.27.3) ---
        # this takes care of everything as long as we aren't under zero3
        if trainer.args.should_save:
            trainer._save(output_dir)

    else:  # Also support saving for non-FSDP models.
        # NOTE(lxuechen): Saving and loading T5 has weird pickle issues due to device map.
        #  Wasn't able to exactly pinpoint. But saving to and loading from CPU seems to work.
        #  In principle, trainer.save_model() should do the same thing, but breaks in practice.
        #  We drop T5 support.
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

    if trainer.args.should_save:
        if give_rw_access:
            try:
                os.system(f"chmod -R a+xwr {output_dir}")
            except Exception as e:
                logger.fatal(f"Failed to give read-write access to {output_dir}: {e}")
        logger.warning(f"Saving model took {time.perf_counter() - now:.2f} seconds.")
