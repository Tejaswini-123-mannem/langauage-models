"""
Curriculum sampler: shift the task mix from easy to hard during training.

Run 5. Runs 1-4 drew training examples uniformly, so every task was
equally represented at every step. The error analysis says the tasks are
not equally hard and, more importantly, that they are not INDEPENDENT:

    numeric_relation   learned first and fastest -- given two numbers,
                       say which is bigger
    numeric_pair       the same comparison, but the answer is a name
    chain_*            the same comparison repeated and composed, with
                       half the premises stated in the "less" direction

Run 3's worked answers made the dependency explicit in the target text,
and the numeric tasks jumped while the chain tasks stayed at chance --
the models could compare, and could not chain. A curriculum is the
natural next lever: spend the early steps where the basic comparison is
learned, and the late steps where it has to be composed.

Weights are interpolated linearly from start to end over the run, so
early batches are mostly easy examples and late batches mostly hard ones,
with every task present throughout. Nothing is ever excluded: a task
dropped entirely early is a task the model can forget by the end.

Resumability
------------
The sampler's whole state is (seed, step). Each batch is drawn from a
generator seeded by both, so resuming at step n redraws exactly the batch
an uninterrupted run would have drawn -- the same guarantee
PermutationSampler gives by storing (seed, epoch, position).

Sampling is WITH replacement, unlike PermutationSampler. Under changing
weights there is no fixed epoch to permute, and with 8,000 examples over
6,000 batches of 16 the difference is immaterial.
"""

import numpy as np


class CurriculumSampler:
    """Weighted example sampler whose weights move from easy to hard."""

    def __init__(self, tasks, batch_size, max_steps, easy_tasks, hard_tasks,
                 easy_start=3.0, easy_end=1.0, hard_start=1.0, hard_end=3.0,
                 seed=42, step=0):
        """
        tasks       task name per training example, in dataset order
        easy_tasks  task names weighted easy_start -> easy_end
        hard_tasks  task names weighted hard_start -> hard_end

        A task in neither list keeps weight 1.0 throughout.
        """

        self.tasks = list(tasks)
        self.batch_size = int(batch_size)
        self.max_steps = max(1, int(max_steps))
        self.seed = int(seed)
        self.step = int(step)

        self.easy = np.array([t in set(easy_tasks) for t in self.tasks])
        self.hard = np.array([t in set(hard_tasks) for t in self.tasks])
        self.easy_start, self.easy_end = float(easy_start), float(easy_end)
        self.hard_start, self.hard_end = float(hard_start), float(hard_end)

    @property
    def batches_per_epoch(self):
        return max(1, len(self.tasks) // self.batch_size)

    def weights(self, step):
        """Per-example sampling probabilities at one step."""

        progress = min(1.0, step / max(1, self.max_steps - 1))

        easy = self.easy_start + progress * (self.easy_end - self.easy_start)
        hard = self.hard_start + progress * (self.hard_end - self.hard_start)

        weights = np.ones(len(self.tasks), dtype=np.float64)
        weights[self.easy] = easy
        weights[self.hard] = hard

        return weights / weights.sum()

    def mix(self, step):
        """Share of the batch drawn from easy and from hard tasks, for logging."""

        weights = self.weights(step)

        return float(weights[self.easy].sum()), float(weights[self.hard].sum())

    def next_batch(self):
        """Indices for the next batch, drawn under this step's weights."""

        rng = np.random.default_rng([self.seed, self.step])
        batch = rng.choice(len(self.tasks), size=self.batch_size, replace=False,
                           p=self.weights(self.step))
        self.step += 1

        return batch

    def state_dict(self):
        return {"seed": self.seed, "step": self.step, "kind": "curriculum"}

    def load_state_dict(self, state):
        self.seed = int(state["seed"])
        self.step = int(state["step"])

    def __repr__(self):
        return (f"CurriculumSampler(examples={len(self.tasks)}, "
                f"batch={self.batch_size}, step={self.step})")
