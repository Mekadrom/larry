import torch
# noinspection PyProtectedMember
import torch._dynamo
from torch import nn

from larry.common.config.training.trainer_configs import CompileConfig


class ModelCompiler:
    def __init__(self, config: CompileConfig) -> None:
        self.config = config

    @staticmethod
    def repeated_blocks(model: nn.Module) -> list[nn.Module]:
        selected: dict[str, nn.Module] = {}
        seen: set[int] = set()
        for name, module in model.named_modules():
            if not isinstance(module, nn.ModuleList):
                continue

            for index, block in enumerate(module):
                path = f"{name}.{index}" if name else str(index)
                if id(block) in seen or any(path.startswith(f"{outer}.") for outer in selected):
                    continue

                seen.add(id(block))
                selected[path] = block
        return list(selected.values())

    def targets(self, model: nn.Module) -> list[nn.Module]:
        if self.config.mode == "model":
            return [model]

        if self.config.mode == "blocks":
            return self.repeated_blocks(model)

        return []

    def compile_forward(self, module: nn.Module) -> None:
        if module.__dict__.get("_larry_compiled", False):
            return

        module.forward = torch.compile(
            module.forward,
            backend=self.config.backend,
            mode=self.config.compile_mode,
            dynamic=self.config.dynamic,
        )
        module.__dict__["_larry_compiled"] = True

    def apply(self, model: nn.Module) -> int:
        targets = self.targets(model)
        if not targets:
            return 0

        # noinspection PyProtectedMember
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, self.config.recompile_limit)
        for module in targets:
            self.compile_forward(module)

        return len(targets)
