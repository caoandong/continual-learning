from __future__ import annotations

from .core import Scenario, SplitEvaluation, StageConfig
from .tasks import (
    AdditionTask,
    ECARolloutTask,
    ExpressionTask,
    LinearCombinationTask,
    MultiplicationTask,
    ValueAssignmentTask,
)


COMMON_SPLITS = (
    SplitEvaluation("id", 256),
    SplitEvaluation("ood_length", 256),
    SplitEvaluation("ood_template", 256),
)


def three_stage_arith_v1() -> Scenario:
    stages = [
        StageConfig(
            name="stage_1_addition",
            task=AdditionTask(),
            train_examples=1024,
            batch_size=32,
            checkpoints=(0, 128, 256, 512, 1024),
            eval_splits=COMMON_SPLITS,
            train_seed=11,
        ),
        StageConfig(
            name="stage_2_multiplication",
            task=MultiplicationTask(),
            train_examples=1024,
            batch_size=32,
            checkpoints=(0, 128, 256, 512, 1024),
            eval_splits=COMMON_SPLITS,
            train_seed=22,
        ),
        StageConfig(
            name="stage_3_linear_combination",
            task=LinearCombinationTask(),
            train_examples=1024,
            batch_size=32,
            checkpoints=(0, 128, 256, 512, 1024),
            eval_splits=COMMON_SPLITS,
            train_seed=33,
        ),
    ]
    return Scenario(
        name="three_stage_arith_v1",
        stages=stages,
        description=(
            "Core arithmetic continual-learning track: addition -> multiplication -> "
            "linear combination. Designed to test positive transfer and composition."
        ),
    )


def extended_arith_v1() -> Scenario:
    stages = [
        StageConfig(
            name="stage_1_addition",
            task=AdditionTask(),
            train_examples=1024,
            batch_size=32,
            checkpoints=(0, 128, 256, 512, 1024),
            eval_splits=COMMON_SPLITS,
            train_seed=101,
        ),
        StageConfig(
            name="stage_2_multiplication",
            task=MultiplicationTask(),
            train_examples=1024,
            batch_size=32,
            checkpoints=(0, 128, 256, 512, 1024),
            eval_splits=COMMON_SPLITS,
            train_seed=202,
        ),
        StageConfig(
            name="stage_3_linear_combination",
            task=LinearCombinationTask(),
            train_examples=1024,
            batch_size=32,
            checkpoints=(0, 128, 256, 512, 1024),
            eval_splits=COMMON_SPLITS,
            train_seed=303,
        ),
        StageConfig(
            name="stage_4_expression",
            task=ExpressionTask(),
            train_examples=1024,
            batch_size=32,
            checkpoints=(0, 128, 256, 512, 1024),
            eval_splits=COMMON_SPLITS,
            train_seed=404,
        ),
    ]
    return Scenario(
        name="extended_arith_v1",
        stages=stages,
        description=(
            "Extended arithmetic continual-learning track: addition -> multiplication -> "
            "linear combination -> nested expression evaluation."
        ),
    )


def warmup_transfer_v1() -> Scenario:
    stages = [
        StageConfig(
            name="stage_0_value_assignment",
            task=ValueAssignmentTask(),
            train_examples=1024,
            batch_size=32,
            checkpoints=(0, 128, 256, 512, 1024),
            eval_splits=COMMON_SPLITS,
            train_seed=7,
        ),
        StageConfig(
            name="stage_1_eca_rollout",
            task=ECARolloutTask(),
            train_examples=1024,
            batch_size=32,
            checkpoints=(0, 128, 256, 512, 1024),
            eval_splits=COMMON_SPLITS,
            train_seed=8,
        ),
        StageConfig(
            name="stage_2_addition",
            task=AdditionTask(),
            train_examples=1024,
            batch_size=32,
            checkpoints=(0, 128, 256, 512, 1024),
            eval_splits=COMMON_SPLITS,
            train_seed=9,
        ),
        StageConfig(
            name="stage_3_multiplication",
            task=MultiplicationTask(),
            train_examples=1024,
            batch_size=32,
            checkpoints=(0, 128, 256, 512, 1024),
            eval_splits=COMMON_SPLITS,
            train_seed=10,
        ),
        StageConfig(
            name="stage_4_linear_combination",
            task=LinearCombinationTask(),
            train_examples=1024,
            batch_size=32,
            checkpoints=(0, 128, 256, 512, 1024),
            eval_splits=COMMON_SPLITS,
            train_seed=11,
        ),
    ]
    return Scenario(
        name="warmup_transfer_v1",
        stages=stages,
        description=(
            "Optional non-arithmetic warmup track motivated by synthetic pre-pretraining ideas: "
            "value assignment -> cellular automaton rollout -> addition -> multiplication -> "
            "linear combination."
        ),
    )
